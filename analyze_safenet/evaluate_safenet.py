# analyze_safenet/evaluate_safenet.py
"""
evaluate_safenet.py: Comprehensive Evaluation of the SafeNet Phishing Detection System

Purpose:
This script performs a thorough evaluation of the SafeNet phishing detection system
using a labeled dataset of screenshots. It is designed to mirror the exact
processing logic of the main SafeNet application (main.py) to ensure that the
evaluation results accurately reflect the system's real-world performance.

Inputs:
1. Ground-Truth Labels CSV:
   - A CSV file containing the filenames of screenshots and their corresponding
     ground-truth labels ("phishing" or "safe").
   - Expected path: C:/Users/Owner/Pictures/evaluation_dataset/labels.csv

2. Screenshots Directory:
   - A folder containing all the screenshot images referenced in the labels CSV.
   - Expected path: C:/Users/Owner/Pictures/evaluation_dataset/screenshots/

Outputs:
1. Detailed Evaluation Report (CSV):
   - A comprehensive CSV file ('evaluation_report.csv') that logs the results for
     every processed screenshot. It includes raw OCR text, filtered text, detected
     URLs, model predictions, confidence scores, and error analysis.

2. Misclassified Cases Report (CSV):
   - A separate CSV file ('misclassified_cases.csv') that isolates all cases
     where the system's prediction did not match the ground-truth label,
     allowing for targeted analysis of failures.

3. Console Summary:
   - A printed summary of key performance metrics, including accuracy, precision,
     recall, F1-score, and a confusion matrix.

How to Run:
1. Ensure the paths to the dataset, screenshots, and model are correctly set in the
   CONFIGURATION section.
2. Make sure all required libraries (pandas, scikit-learn, torch, etc.) are
   installed in your environment.
3. Run the script from the command line:
   python analyze_safenet/evaluate_safenet.py
"""

import os
import re
import csv
import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from PIL import Image
import pytesseract
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI
import tldextract

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
load_dotenv()

# ========== CONFIGURATION ==========
MODEL_PATH = "models/saved_model/saved_model_v4"
TOKENIZER_PATH = "tokenizer"
LABELS_FILE_PATH = r"C:\Users\Owner\Pictures\evaluation_dataset\labels.xlsx"
SCREENSHOTS_DIR = r"C:\Users\Owner\Pictures\evaluation_dataset\screenshots"
OUTPUT_DIR = "evaluation"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
PSM_MODE = r'--oem 3 --psm 11'

# --- Logic-defining constants (mirrored from main.py) ---
PHISHING_THRESHOLD = 0.92
CHUNK_SIZE = 512
BACKUP_THRESHOLD = 0.75 # Only trigger fallback just below the main threshold
suspicious_keywords = [
    'login', 'verify', 'update', 'secure', 'reset', 'phishing',
    'signin', 'password', 'bank', 'confirm', 'validate', 'webscr',
    'wallet', 'security', 'locked', 'unusual activity', 'deactivate',
    'payment', 'alert', 'click here', 'unlock', 'invoice', 'urgent',
    'credential', 'session expired', 'verify identity', 'your account',
    'limited access', 'unrecognized', 'recovery', 'restricted', 'authentication',
    'customs duty', 'customs', 'parcel', 'reschedule', 'refund', 'refunds',
    'funds', 'refunded', 'refunding', 'shipping fee', 'shipping fees',
    'duty', 'tracking', 'transfer', 'transferred',
    'secure-', 'account-', 'login-', 'update-', 'verify-', 'reset-',
    'confirm-', 'validate-', 'webscr-', 'wallet-', 'security-', 'locked-',
    'unusual activity-', 'deactivate-', 'payment-', 'alert-', 'click here-',
    'unlock-', 'invoice-', 'urgent-', 'credential-', 'session expired-',
    'verify identity-', 'your account-', 'limited access-', 'unrecognized-',
    'recovery-', 'restricted-', 'authentication-', 'customs duty-', 'customs-',
    'parcel-', 'reschedule-', 'refund-', 'refunds-', 'funds-', 'refunded-',
    'refunding-', 'shipping fee-', 'shipping fees-', 'duty-', 'tracking-',
    'transfer-', 'transferred-',
    'billing', '-online', 'log in', 'account access', 'email or mobile',
    'enter password', 'sign in to your account',
    'act now', 'verify now', 'click to verify', 'login now', 'update now',
    'dear customer', 'dear user', 'verify your email', 'claim your reward',
    'you have won', 'apple lottery', 'lotto', 'selected', 'winner',
    'security update', 'important notice', 'take action', 'you are eligible',
    'final notice', 'time sensitive', 'response required', 'limited time',
    'reset your credentials', 'access denied', 'payment required',
    'unauthorized login', 'security alert', 'bank account', 'money transfer',
    'urgent payment', 'suspended', 'claim now', 'email verification',
    'sign in required', 'temporary hold', 'account suspended','invoice','blocked','flagged','warning','back to safety','enter site','trust'
]

bad_tlds = [
    '.zip', '.tk', '.cn', '.gq', '.ml', '.ru', '.xyz',
    '.top', '.cf', '.work', '.click', '.support', '.fit', '.review',
    '.country', '.stream', '.biz', '.monster', '.su', '.pw', '.info',
    '.loan', '.rest', '.cam', '.men', '.party', '.host'
]

known_shorteners = [
    'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'buff.ly',
    'rebrand.ly', 'shorte.st', 'cutt.ly', 'rb.gy', 'is.gd', 'short.cm',
    'v.gd', 'soo.gd', 'bl.ink', 't.ly', 'clck.ru', 'lnkd.in',
    's.id', 'qr.ae', 'adf.ly'
]

soft_blacklist_topics = {
    "shipping": [
        "parcel", "tracking", "reschedule", "customs", "shipping", "delivery",
        "warehouse", "package", "post office", "delivery failed", "arrived",
        "track", "shipment", "custom fee"
    ],
    "finance": [
        "refund", "payment", "invoice", "fee", "bank", "wallet", "billing",
        "charge", "amount", "funds", "refunded", "refunding", "shipping fee",
        "shipping fees", "duty", "tracking", "transfer", "transferred"
    ],
    "account_update": [
        "verify", "update", "confirm", "identity", "secure", "reset",
        "credentials", "authentication", "validate identity", "locked account",
        "account recovery", "login issues", "restore access"
    ],
     "site_warning": [
        "warning", "suspicious site", "suspected phishing", "phishing site",
        "site ahead", "flagged", "avoid this link", "unsafe", "blocked website",
        "blocked by your organization", "contact your administrator", "red screen",
        "security warning", "review your flagged website", "back to safety"
    ]
}

# ===================================

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========== LOGIC FROM main.py (for 100% parity) ==========

def mask_sensitive_info(text):
    # Mask credit card numbers (13-16 digits)
    text = re.sub(r'\b\d{13,16}\b', '[CREDIT_CARD]', text)
    # Mask password fields (e.g., password: something)
    text = re.sub(r'(password\s*[:=]\s*)(\S+)', r'\1[PASSWORD]', text, flags=re.IGNORECASE)
    return text

def analyze_with_chatgpt(text):
    filtered_text = mask_sensitive_info(text)
    prompt = (
        "You are an AI security assistant. Analyze the following text and determine if it is likely to be a phishing attempt. "
        "Reply with 'PHISHING' or 'SAFE' and a brief explanation.\n\n"
        f"Text:\n{filtered_text}"
    )
    try:
        response = client.chat.completions.create(
             model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": ( "You're a cybersecurity and phishing detection expert. Each incoming message is OCR output from a screenshot — it may contain noise, formatting errors, or random characters.\n\n"
                                "Your task: Determine whether the message is part of a phishing attempt. Be strict — do not flag a message as phishing unless at least TWO of the following conditions are met:\n\n"
                                "1. Manipulative or urgent language (e.g. 'your account will be suspended', 'act now')\n"
                                "2. Request for credentials, personal or financial information\n"
                                "3. Suspicious, masked, or malformed link\n"
                                "4. Use of a well-known brand name in a misleading or fake context\n\n"
                                "If only one of these conditions is present, or if the text is unclear or incomplete, classify as SAFE.\n\n"
                                "Ignore noise caused by OCR (random characters, poor formatting, etc.). Focus only on the semantic meaning.\n\n"
                                "Examples of SAFE content:\n"
                                "- No clear phishing intent\n"
                                "- Legitimate services like microsoft.com or mail.google.com\n"
                                "- General messages without manipulation or links\n\n"
                                "If you're unsure or lack context, return: 'Inconclusive – more context needed'."
                            ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [ChatGPT ERROR] {type(e).__name__}: {e}")
        return "Unable to analyze with ChatGPT"

def correct_ocr_errors(text):
    corrections = [
        (r'\b0([A-Za-z])', r'O\1'), (r'([A-Za-z])0\b', r'\1O'),
        (r'\b1([A-Za-z])', r'l\1'), (r'([A-Za-z])1\b', r'\1l'),
        (r'\b5([A-Za-z])', r'S\1'), (r'([A-Za-z])5\b', r'\1S'),
        (r'\b([A-Za-z])l\b', r'\1I'),
    ]
    for pattern, repl in corrections:
        text = re.sub(pattern, repl, text)
    return text

def clean_text(text):
    import string
    text = ''.join(c if c in string.printable else ' ' for c in text)
    text = re.sub(r'[^A-Za-z0-9\s.,:;!?@%&$/\-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def compute_noise_ratio(text):
    tokens = text.split()
    non_words = [t for t in tokens if not re.search(r'[a-zA-Z]', t)]
    if not tokens:
        return 1.0
    return len(non_words) / len(tokens)

def extract_urls(text):
    # Normalize common OCR errors
    text = text.lower()
    text = text.replace('hxxp', 'http')  # Trick avoidance
    text = text.replace('．', '.').replace('：', ':')
    text = re.sub(r'(?<=\w)(dot)(?=\w)', '.', text)  # "exampledotcom"
    text = re.sub(r'(?<=\w)(@)(?=\w)', '.com', text)  # email OCR issues

    # Remove unwanted characters often added by OCR
    text = re.sub(r'[\[\]<>|{}]', '', text)

    # Original URL regex, improved slightly
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
        if re.match(r'.+\.[a-zA-Z]{2,4}$', url) and not url.startswith(('http', 'www')):
            continue
        if not url.startswith(('http://', 'https://')):
            url = "http://" + url
        cleaned.append(url)

    return list(set(cleaned))

def is_suspicious_url(url: str) -> bool:
    suspicious_keywords = ["login", "secure", "wallet", "verify", "update", "auth", "authentication"]
    suspicious_tlds = ["tk", "ml", "cf", "gq", "xyz", "support", "zip", "click", "top", "cam", "fit"]
    score = 0

    try:
        ext = tldextract.extract(url)
        domain_parts = f"{ext.subdomain}.{ext.domain}".lower()
        
        # Keyword check in domain and subdomain
        if any(word in domain_parts for word in suspicious_keywords):
            score += 1

        # TLD check
        if ext.suffix in suspicious_tlds:
            score += 1
            
    except Exception:
        # In case of any parsing error, assume not suspicious
        return False

    return score >= 2


def minimal_ocr_line_filter(text):
    lines = text.splitlines()
    # If only one line, keep as is
    if len(lines) <= 1:
        filtered = text
    else:
        filtered = '\n'.join([line for line in lines if len(line.strip()) >= 4])
    return filtered

from typing import List, Tuple

keyword_scores = {
    "high": {'verify': 2, 'unauthorized': 2, 'security alert': 2},
    "medium": {'login': 1, 'password': 1, 'update': 1,'limited time': 1},
    "low": {'account': 0.5, 'access': 0.5, 'security': 0.5}
}

def analyze_text_topics(text: str) -> Tuple[int, List[str]]:
    text_lower = text.lower()
    matched_topics = []
    score = 0
    for topic, keywords in soft_blacklist_topics.items():
        if any(keyword in text_lower for keyword in keywords):
            matched_topics.append(topic)
            score += 1
    return score, list(set(matched_topics))

def calculate_keyword_score(text: str) -> Tuple[float, List[str]]:
    text_lower = text.lower()
    score = 0
    found_keywords = []
    
    all_tiers = {**keyword_scores["high"], **keyword_scores["medium"], **keyword_scores["low"]}
    sorted_keywords = sorted(all_tiers.keys(), key=len, reverse=True)
    
    for kw in sorted_keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, text_lower):
            score += all_tiers[kw]
            found_keywords.append(kw)
            text_lower = text_lower.replace(kw, "")
            
    return score, list(set(found_keywords))


def process_image_content(text, tokenizer, model):
    """
    Processes the extracted text from a single image to determine if it's phishing.
    This function contains the core classification logic duplicated from main.py.
    """
    raw_text = text
    # chatgpt_used and chatgpt_result are initialized later
    
    # --- Text Processing ---
    filtered_text = minimal_ocr_line_filter(text)
    corrected_text = correct_ocr_errors(filtered_text)
    cleaned_text = clean_text(corrected_text)

    noise_ratio = compute_noise_ratio(cleaned_text)
    if noise_ratio > 0.6:
        print("  [INFO] Text too noisy, skipping classification.")
        return {
            "prediction": "safe",
            "confidence": 0.0,
            "noise_ratio": noise_ratio,
            "raw_ocr_text": raw_text,
            "filtered_text": cleaned_text,
            "detected_keywords": [],
            "urls": [],
            "suspicious_urls": [],
            "model_prediction_flag": False,
            "keyword_flag": False,
            "suspicious_url_flag": False,
            "chatgpt_used": False,
            "chatgpt_result": "Skipped due to high noise",
            "topic_score": 0,
            "matched_topics": []
        }

    # --- Feature Extraction ---
    urls = extract_urls(cleaned_text)
    keyword_score, found_keywords = calculate_keyword_score(cleaned_text)
    topic_score, matched_topics = analyze_text_topics(cleaned_text)
    suspicious_urls = [u for u in urls if is_suspicious_url(u)]

    # --- Model Prediction ---
    tokens = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=CHUNK_SIZE)["input_ids"][0]
    chunks = [tokens[i:i + CHUNK_SIZE] for i in range(0, len(tokens), CHUNK_SIZE)]
    
    max_confidence = 0
    final_prediction_idx = 0

    if chunks:
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
        
        for pred, conf in results:
            if conf > max_confidence:
                max_confidence = conf
                final_prediction_idx = pred

    # --- Final Decision Logic with Weighted Scoring ---
    is_phishing_predicted_by_model = (final_prediction_idx == 1)
    prediction = "safe"
    chatgpt_used = False
    chatgpt_result = "N/A"

    # --- Component Scoring ---
    model_score = 0
    if is_phishing_predicted_by_model and max_confidence >= PHISHING_THRESHOLD:
        model_score = 2
    elif is_phishing_predicted_by_model and max_confidence >= BACKUP_THRESHOLD:
        model_score = 1

    confidence_bonus = 1 if max_confidence >= 0.9 else 0

    url_score = 0
    if len(suspicious_urls) >= 1:
        url_score += 1
    if any(urlparse(url).netloc.endswith(tld) for url in suspicious_urls for tld in bad_tlds):
        url_score += 1

    total_score = model_score + confidence_bonus + keyword_score + url_score + topic_score

    print(f"  [DEBUG] Suspicion Score: {total_score} (Model: {model_score}, Confidence Bonus: {confidence_bonus}, Keywords: {keyword_score}, URLs: {url_score}, Topics: {topic_score})")

    # --- Final Prediction ---
    if total_score >= 4:
        prediction = "phishing"
    elif total_score >= 3:
        # Trigger ChatGPT only if there are some non-weak signals
        if keyword_score >= 1 or url_score >= 1:
            prediction = "suspected"
        else:
            print(f"  [INFO] Score ({total_score}) in suspected range, but keyword/URL signals are weak. Classifying as safe.")

    if prediction == "suspected":
        log_reason = f"Score: {total_score} (Keywords: {keyword_score}, URLs: {url_score})"
        print(f"  [INFO] Medium suspicion, falling back to ChatGPT. Reason: {log_reason}")
        chatgpt_used = True
        chatgpt_result = analyze_with_chatgpt(cleaned_text)
        print(f"  [INFO] ChatGPT analysis result: {chatgpt_result}")
        
        is_chatgpt_safe = "SAFE" in chatgpt_result.upper()

        if is_chatgpt_safe and is_phishing_predicted_by_model:
            prediction = "safe"
        elif "PHISHING" in chatgpt_result.upper():
            prediction = "phishing"
        else:
            prediction = "safe"

    print("\n[DEBUG] Final Decision:")
    print(f"  Model Prediction: {'phishing' if is_phishing_predicted_by_model else 'safe'}")
    print(f"  Confidence: {max_confidence:.4f}")
    print(f"  Keyword Score: {keyword_score}")
    print(f"  Suspicious URLs Found: {bool(suspicious_urls)}")
    print(f"  Has Trigger: {bool(keyword_score > 0 or suspicious_urls)}")
    print(f"  Final Classification: {prediction}")
    
    return {
        "prediction": prediction,
        "confidence": max_confidence,
        "noise_ratio": noise_ratio,
        "raw_ocr_text": raw_text,
        "filtered_text": cleaned_text,
        "detected_keywords": found_keywords,
        "urls": urls,
        "suspicious_urls": suspicious_urls,
        "topic_score": topic_score,
        "matched_topics": matched_topics,
        "model_prediction_flag": is_phishing_predicted_by_model,
        "keyword_flag": keyword_score > 0,
        "suspicious_url_flag": bool(suspicious_urls),
        "chatgpt_used": chatgpt_used,
        "chatgpt_result": chatgpt_result
    }

def get_error_type(prediction, ground_truth):
    """Determines the type of error (FP, FN, TP, TN)."""
    if prediction == 'phishing' and ground_truth == 'phishing':
        return 'TP'
    if prediction == 'safe' and ground_truth == 'safe':
        return 'TN'
    if prediction == 'phishing' and ground_truth == 'safe':
        return 'FP'
    if prediction == 'safe' and ground_truth == 'phishing':
        return 'FN'
    return 'Unknown'


def main():
    """Main function to run the evaluation."""
    print("Starting SafeNet Evaluation Script...")

    # --- Load Model and Tokenizer ---
    print(f"Loading model from: {MODEL_PATH}")
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model or tokenizer: {e}")
        return

    # --- Load Dataset ---
    print(f"Loading labels from: {LABELS_FILE_PATH}")
    try:
        labels_df = pd.read_excel(LABELS_FILE_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Labels file not found at: {LABELS_FILE_PATH}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to read labels file: {e}")
        print("Please ensure 'openpyxl' is installed if you are using an Excel file (pip install openpyxl).")
        return

    # --- Prepare for evaluation ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    total_images = len(labels_df)

    print(f"Found {total_images} images to evaluate.")
    print("-" * 30)

    # --- Main Evaluation Loop ---
    for index, row in labels_df.iterrows():
        filename = row['filename']
        ground_truth = row['label']
        image_path = os.path.join(SCREENSHOTS_DIR, filename)
        
        print(f"Processing ({index + 1}/{total_images}): {filename}")

        if not os.path.exists(image_path):
            print(f"  [WARNING] Screenshot not found, skipping: {image_path}")
            continue

        try:
            # --- OCR Extraction ---
            image = Image.open(image_path).convert("RGB")
            ocr_text = pytesseract.image_to_string(image, config=PSM_MODE, lang='eng')
            
            if not ocr_text.strip():
                print("  [INFO] OCR returned empty text.")
                result = {
                    "prediction": "safe", "confidence": 0.0, "noise_ratio": 1.0, "raw_ocr_text": "",
                    "filtered_text": "", "detected_keywords": [], "urls": [],
                    "suspicious_urls": [], "model_prediction_flag": False,
                    "keyword_flag": False, "suspicious_url_flag": False,
                    "chatgpt_used": False, "chatgpt_result": "N/A",
                    "topic_score": 0, "matched_topics": []
                }
            else:
                # --- Process Content ---
                result = process_image_content(ocr_text, tokenizer, model)

            # --- Store Results ---
            result['filename'] = filename
            result['ground_truth'] = ground_truth
            result['error_type'] = get_error_type(result['prediction'], ground_truth)
            results.append(result)

        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}: {e}")
            results.append({
                'filename': filename, 'ground_truth': ground_truth,
                'prediction': 'error', 'confidence': 0.0, 'noise_ratio': -1.0, 'error_type': 'Processing Error',
                'raw_ocr_text': '', 'filtered_text': '', 'detected_keywords': [],
                'urls': [], 'suspicious_urls': [], 'chatgpt_used': False, 'chatgpt_result': 'N/A',
                'topic_score': -1, 'matched_topics': []
            })

    # --- Analyze and Report Results ---
    if not results:
        print("\nNo results to analyze. Exiting.")
        return
        
    results_df = pd.DataFrame(results)
    
    # Reorder columns for clarity
    report_cols = [
        'filename', 'ground_truth', 'prediction', 'error_type', 'confidence',
        'noise_ratio', 'topic_score', 'matched_topics',
        'model_prediction_flag', 'keyword_flag', 'suspicious_url_flag',
        'chatgpt_used', 'chatgpt_result',
        'raw_ocr_text', 'filtered_text', 'detected_keywords', 'urls', 'suspicious_urls'
    ]
    results_df = results_df[report_cols]


    # --- Save Reports to CSV ---
    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.csv')
    misclassified_path = os.path.join(OUTPUT_DIR, 'misclassified_cases.csv')
    
    results_df.to_csv(report_path, index=False, encoding='utf-8')
    print(f"\nFull evaluation report saved to: {report_path}")
    
    misclassified_df = results_df[results_df['error_type'].isin(['FP', 'FN'])]
    misclassified_df.to_csv(misclassified_path, index=False, encoding='utf-8')
    print(f"Misclassified cases report saved to: {misclassified_path}")

    # --- Calculate and Print Metrics ---
    # Filter out processing errors for metric calculations
    valid_results_df = results_df[results_df['prediction'] != 'error'].copy()

    # Normalize ground_truth labels and filter for binary classification
    valid_results_df['ground_truth_norm'] = valid_results_df['ground_truth'].str.lower().str.strip()
    binary_df = valid_results_df[valid_results_df['ground_truth_norm'].isin(['safe', 'phishing'])].copy()

    if binary_df.empty:
        print("\n[WARNING] No valid 'safe' or 'phishing' labels found for metric calculation.")
        return

    y_true = binary_df['ground_truth_norm']
    y_pred = binary_df['prediction'].str.lower().str.strip()
    labels = ['safe', 'phishing']

    # Remap 'suspected' to 'safe' for binary classification metrics
    y_pred_for_metrics = y_pred.replace({'suspected': 'safe'})

    accuracy = accuracy_score(y_true, y_pred_for_metrics)
    precision = precision_score(y_true, y_pred_for_metrics, pos_label='phishing', zero_division=0)
    recall = recall_score(y_true, y_pred_for_metrics, pos_label='phishing', zero_division=0)
    f1 = f1_score(y_true, y_pred_for_metrics, pos_label='phishing', zero_division=0)
    cm = confusion_matrix(y_true, y_pred_for_metrics, labels=labels)

    print("\n" + "="*30)
    print("      EVALUATION METRICS")
    print("="*30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix (treating 'suspected' as 'safe'):")
    print("             Predicted")
    print(f"         Safe  Phishing")
    print(f"Actual Safe:  {cm[0][0]:>4}  {cm[0][1]:>5}")
    print(f"Actual Phish: {cm[1][0]:>4}  {cm[1][1]:>5}")
    print("="*30)
    
    # --- Error Analysis Summary ---
    fp_count = len(misclassified_df[misclassified_df['error_type'] == 'FP'])
    fn_count = len(misclassified_df[misclassified_df['error_type'] == 'FN'])
    suspected_count = len(valid_results_df[valid_results_df['prediction'] == 'suspected'])

    print("\nError Analysis Summary:")
    print(f"False Positives (FP): {fp_count}")
    print(f"False Negatives (FN): {fn_count}")
    print(f"Flagged as 'Suspected' before ChatGPT: {suspected_count}")
    
    if fn_count > 0:
        print("\nCommon causes for False Negatives (Phishing missed):")
        fn_df = misclassified_df[misclassified_df['error_type'] == 'FN']
        if not all(fn_df['model_prediction_flag']): print("- Model confidence was high, but other heuristics (keywords/URLs) failed.")
        if not all(fn_df['keyword_flag']): print("- Required phishing keywords were not detected.")
        if not all(fn_df['suspicious_url_flag']): print("- Extracted URLs were not flagged as suspicious.")
    
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main() 