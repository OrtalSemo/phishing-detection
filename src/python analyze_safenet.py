import os
import zipfile
import csv
import re
import time
from datetime import datetime
from urllib.parse import urlparse
import concurrent.futures

import torch
from torch.nn.functional import softmax
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from PIL import Image
import pytesseract

# ========== CONFIGURATION ==========
MODEL_PATH = "models/saved_model/saved_model_v4"
TOKENIZER_PATH = "tokenizer"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
PSM_MODE = r'--oem 3 --psm 11'  # OCR Engine Mode + Page Segmentation Mode
PHISHING_THRESHOLD = 0.92
CHUNK_SIZE = 512  # Max number of tokens per input to DistilBERT
# ===================================

# ZIP file path (configurable)
zip_path = r'C:\Users\Owner\Downloads\archive (15).zip'

# --- Reused Detection Functions from safenet_monitor.py ---

def correct_ocr_errors(text):
    """Simple replacements for common OCR mistakes"""
    corrections = [
        (r'\b0([A-Za-z])', r'O\1'),  # 0 at start of word to O
        (r'([A-Za-z])0\b', r'\1O'),  # 0 at end of word to O
        (r'\b1([A-Za-z])', r'l\1'),  # 1 at start of word to l
        (r'([A-Za-z])1\b', r'\1l'),  # 1 at end of word to l
        (r'\b5([A-Za-z])', r'S\1'),  # 5 at start of word to S
        (r'([A-Za-z])5\b', r'\1S'),  # 5 at end of word to S
        (r'\b([A-Za-z])l\b', r'\1I'), # l as I
    ]
    for pattern, repl in corrections:
        text = re.sub(pattern, repl, text)
    return text

def clean_text(text):
    """Remove non-printable characters and normalize whitespace"""
    import string
    printable = set(string.printable)
    cleaned = ''.join(c if c in printable else ' ' for c in text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def extract_urls(text):
    """Extract URLs from text using regex"""
    url_pattern = re.compile(
        r'\b('
        r'(?:https?://|www\.)[^\s<>"]+'
        r'|'
        r'(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,10}(?:/[^\s<>"]*)?'
        r')\b'
    )
    matches = url_pattern.findall(text)
    cleaned = []
    for url in matches:
        url = url.strip(".,:;!?()[]{}<>\"'")
        # Avoid matching filenames (e.g., main.py, file.txt)
        if re.match(r'.+\.[a-zA-Z]{2,4}$', url) and not url.startswith(('http', 'www')):
            continue
        # Add http:// if missing
        if not url.startswith(('http://', 'https://')):
            url = "http://" + url
        cleaned.append(url)
    return list(set(cleaned))

def is_suspicious_url(url):
    """Check if URL is suspicious based on heuristics"""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    score = score_url_suspicion(url)

    if score < 3:
        return False
    if score == 3 and contains_uuid(path):
        return False
    return True

def score_url_suspicion(url):
    """Score URL suspicion level (0-5)"""
    suspicious_keywords = [
        'login', 'verify', 'update', 'secure', 'account', 'reset',
        'signin', 'password', 'bank', 'confirm', 'validate', 'webscr'
    ]
    bad_tlds = ['.zip', '.tk', '.cn', '.gq', '.ml', '.ru', '.xyz']
    known_shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'buff.ly']
    
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    score = 0
    
    if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain):
        score += 1
    if any(re.search(rf'\b{k}\b', url.lower()) for k in suspicious_keywords):
        score += 1
    if any(domain.endswith(tld) for tld in bad_tlds):
        score += 1
    if any(short in domain for short in known_shorteners):
        score += 1
    if domain.count('.') > 3:
        score += 1
    return score

def contains_uuid(path):
    """Check if path contains UUID"""
    uuid_regex = re.compile(
        r'[a-fA-F0-9]{8}-'
        r'[a-fA-F0-9]{4}-'
        r'[a-fA-F0-9]{4}-'
        r'[a-fA-F0-9]{4}-'
        r'[a-fA-F0-9]{12}'
    )
    return bool(uuid_regex.search(path))

def minimal_ocr_line_filter(text):
    """Filter OCR text to remove noise"""
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue
        if re.match(r'^[^a-zA-Z]*$', line):
            continue
        filtered_lines.append(line)
    return '\n'.join(filtered_lines)

def classify_text_chunks(text, tokenizer, model):
    """Classify text using DistilBERT in chunks"""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=CHUNK_SIZE)["input_ids"][0]
    chunks = [tokens[i:i+CHUNK_SIZE] for i in range(0, len(tokens), CHUNK_SIZE)]

    def classify_chunk(chunk):
        if chunk.size(0) > CHUNK_SIZE:
            chunk = chunk[:CHUNK_SIZE]
        
        input_dict = {"input_ids": chunk.unsqueeze(0)}
        with torch.no_grad():
            outputs = model(**input_dict)
        probs = softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        conf = probs[pred].item()
        return pred, conf

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(classify_chunk, chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    max_confidence = 0
    final_prediction = 0
    for pred, conf in results:
        if conf > max_confidence:
            max_confidence = conf
            final_prediction = pred

    return final_prediction, max_confidence

def process_image_with_detection(image_file, tokenizer, model):
    """Process image using full detection logic from safenet_monitor.py"""
    try:
        # Open and process image
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using OCR
        text = pytesseract.image_to_string(image, config=PSM_MODE, lang='eng')
        
        if not text or len(text.strip()) < 10:
            return None, "No significant text found", text.strip()[:100]
        
        # Apply detection logic from safenet_monitor.py
        raw_text = text
        
        # Filter and clean text
        text = minimal_ocr_line_filter(text)
        text = correct_ocr_errors(text)
        text = clean_text(text)
        
        # Extract URLs and keywords
        urls = extract_urls(text)
        
        # Define phishing keywords
        suspicious_keywords = [
            'login', 'verify', 'update', 'secure', 'account', 'reset',
            'signin', 'password', 'bank', 'confirm', 'validate', 'webscr'
        ]
        phishing_keywords_extra = [
            'urgent', 'suspend', 'unauthorized', 'action required', 'limited time',
            'reset', 'security alert', 'reactivate', 'access', 'billing', 'invoice',
            'credentials', 'verify identity'
        ]
        all_phishing_keywords = suspicious_keywords + phishing_keywords_extra
        
        words = text.lower().split()
        found_keywords = [word for word in words if word in all_phishing_keywords]
        suspicious_urls = [u for u in urls if is_suspicious_url(u)]
        
        # Classify text using model
        final_prediction, max_confidence = classify_text_chunks(text, tokenizer, model)
        
        # Final decision logic: Require ALL indicators (same as safenet_monitor.py)
        if (
            final_prediction == 1
            and (max_confidence >= PHISHING_THRESHOLD)
            and found_keywords
            and suspicious_urls
        ):
            prediction_label = "phishing"
        else:
            prediction_label = "legit"
        
        # Create preview text
        preview = raw_text[:200].replace('\n', ' ').strip()
        
        return prediction_label, f"Confidence: {max_confidence:.3f}, Keywords: {found_keywords}, URLs: {suspicious_urls}", preview
        
    except Exception as e:
        return None, f"Error: {str(e)}", ""

def analyze_zip_images(zip_path, target_folder='phishIRIS_DL_Dataset/val/other', output_file='phish_eval_results.csv'):
    """Main function to analyze images in ZIP file"""
    print("🔄 Starting evaluation of website screenshots...")
    print(f"📁 Processing ZIP file: {zip_path}")
    print(f"📊 Target folder: {target_folder}")
    print("-" * 60)
    
    # Load model and tokenizer
    print("🤖 Loading DistilBERT model and tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    
    # Initialize counters
    processed = 0
    correct = 0
    false_positives = []
    failed_files = []
    
    print(f"🔍 Looking for images in folder: {target_folder}")
    
    # Create CSV file
    with open(output_file, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "True_Label", "Predicted_Label", "Text_Preview"])
        
        with zipfile.ZipFile(zip_path, 'r') as archive:
            # Get all files and show available folders
            all_files = archive.namelist()
            folders = set()
            for f in all_files:
                if '/' in f:
                    folders.add(f.split('/')[0])
            print(f"📁 Available top-level folders: {sorted(folders)}")
            
            # Find target images
            found_images = []
            for file in all_files:
                if file.startswith(target_folder) and file.endswith(('.png', '.jpg', '.jpeg')):
                    found_images.append(file)
            
            print(f"🖼️  Found {len(found_images)} images in target folder")
            print("-" * 60)
            
            # Process each image
            for i, file in enumerate(found_images):
                if i % 50 == 0:  # Progress every 50 files
                    print(f"📄 Processing {i+1}/{len(found_images)}: {file}")
                
                with archive.open(file) as image_file:
                    prediction, details, preview = process_image_with_detection(image_file, tokenizer, model)
                    
                    if prediction is not None:
                        processed += 1
                        true_label = "legit"  # All files in 'other' folder should be legitimate
                        
                        # Write to CSV
                        writer.writerow([file, true_label, prediction, preview])
                        
                        # Track accuracy
                        if prediction == true_label:
                            correct += 1
                        else:
                            false_positives.append((file, details, preview))
                    else:
                        failed_files.append((file, details))
                        # Still write to CSV with blank prediction
                        writer.writerow([file, "legit", "", preview])
    
    # Print results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"✅ Results saved to: {output_file}")
    print(f"🖼️  Total images found: {len(found_images)}")
    print(f"🔍 Successfully processed: {processed}")
    print(f"❌ Failed to process: {len(failed_files)}")
    
    if processed > 0:
        accuracy = (correct / processed) * 100
        print(f"🎯 Accuracy on legit samples: {correct}/{processed} ({accuracy:.2f}%)")
        
        if false_positives:
            print(f"\n⚠️  False Positives ({len(false_positives)}):")
            for i, (file, details, preview) in enumerate(false_positives[:10]):  # Show first 10
                print(f"   {i+1}. {file}")
                print(f"      Details: {details}")
                print(f"      Preview: {preview[:100]}...")
                print()
    else:
        print("❌ No images were successfully processed.")
    
    if failed_files:
        print(f"\n🚫 Failed Files ({len(failed_files)}):")
        for file, reason in failed_files[:5]:  # Show first 5
            print(f"   - {file}: {reason}")

def main():
    """Main execution function"""
    # Optional: allow command line argument for zip path
    import sys
    global zip_path
    
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
    
    # Check if ZIP file exists
    if not os.path.exists(zip_path):
        print(f"❌ Error: ZIP file not found: {zip_path}")
        return
    
    # Run the analysis
    analyze_zip_images(zip_path)

if __name__ == "__main__":
    main()
