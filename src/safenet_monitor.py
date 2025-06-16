# main.py – SafeNet (Tesseract + DistilBERT with chunked classification)

# ========== RUN MODES ==========
# The script supports two run modes: "production" and "debug".
#
# 1. Production Mode:
#    - Continuous, event-driven monitoring using keyboard, mouse, and window events.
#    - To enable, set the environment variable: SAFENET_MODE=production
#
# 2. Debug Mode:
#    - Runs a single, one-off scan and then exits. Useful for testing.
#    - To enable, set the environment variable: SAFENET_MODE=debug
#
# If SAFENET_MODE is not set, the script will prompt the user to choose a mode
# at runtime, assuming they have 'admin' privileges.
# ===============================

import time
from datetime import datetime
import os
import re
import csv
import threading
import difflib
import concurrent.futures

from urllib.parse import urlparse
import pyautogui
import pytesseract
from PIL import Image
import imagehash
import torch
from torch.nn.functional import softmax
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pygetwindow as gw
import pynput
from dotenv import load_dotenv
from openai import OpenAI  # Only if you use GPT backup
from event_exposure import filter_event, format_event, export_for_splunk, get_user_role
import anomaly_detection as anomaly  # <-- NEW IMPORT

load_dotenv()

# ========== CONFIGURATION ==========
MODEL_PATH = "models/saved_model/saved_model_v4"
SCREENSHOT_DIR = r"C:\Users\Owner\Pictures\SafeNetAI"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
PSM_MODE = r'--oem 3 --psm 11'  # OCR Engine Mode + Page Segmentation Mode
PHISHING_THRESHOLD = 0.92
CHUNK_SIZE = 512  # Max number of tokens per input to DistilBERT
BACKUP_THRESHOLD = 0.75  # Set as appropriate
# ===================================

# print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Sensitive Info Masking Function ---
def mask_sensitive_info(text):
    # Mask credit card numbers (13-16 digits)
    text = re.sub(r'\b\d{13,16}\b', '[CREDIT_CARD]', text)
    # Mask password fields (e.g., password: something)
    text = re.sub(r'(password\s*[:=]\s*)(\S+)', r'\1[PASSWORD]', text, flags=re.IGNORECASE)
    return text

# --- ChatGPT Backup Analysis Function ---
def analyze_with_chatgpt(text):
    filtered_text = mask_sensitive_info(text)
    prompt = (
        "You are an AI security assistant. Analyze the following text and determine if it is likely to be a phishing attempt. "
        "Reply with 'PHISHING' or 'SAFE' and a brief explanation.\n\n"
        f"Text:\n{filtered_text}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cybersecurity and phishing detection expert. All incoming messages are texts extracted from screenshots using OCR technology. "
                        "These texts may be noisy, fragmented, or contain formatting errors due to the OCR process. Do not flag a message as phishing based solely on unclear structure, randomness, or visual artifacts.\n\n"
                        "Your job is to determine whether the content is likely to be part of a phishing attempt. Focus on meaningful phishing indicators such as:\n"
                        "• Urgent or manipulative language\n"
                        "• Requests for personal/sensitive information\n"
                        "• Suspicious or mismatched links\n"
                        "• Imitation of known services or brands\n\n"
                        "When uncertain, respond that the result is inconclusive and explain what additional context is needed. Be careful not to misclassify messages due to noise introduced by OCR."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ChatGPT ERROR] {type(e).__name__}: {e}")
        return "Unable to analyze with ChatGPT"

# --- Global Constants for Heuristics ---
suspicious_keywords = [
    'login', 'verify', 'update', 'secure', 'account', 'reset',
    'signin', 'password', 'bank', 'confirm', 'validate', 'webscr'
]
bad_tlds = ['.zip', '.tk', '.cn', '.gq', '.ml', '.ru', '.xyz']
known_shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'buff.ly']

# --- OCR Correction Function ---
def correct_ocr_errors(text):
    # Simple replacements for common OCR mistakes
    corrections = [
        (r'\b0([A-Za-z])', r'O\1'),  # 0 at start of word to O
        (r'([A-Za-z])0\b', r'\1O'),  # 0 at end of word to O
        (r'\b1([A-Za-z])', r'l\1'),  # 1 at start of word to l
        (r'([A-Za-z])1\b', r'\1l'),  # 1 at end of word to l
        (r'\b5([A-Za-z])', r'S\1'),  # 5 at start of word to S
        (r'([A-Za-z])5\b', r'\1S'),  # 5 at end of word to S
        (r'\b([A-Za-z])l\b', r'\1I'), # l as I
        # Add more as needed
    ]
    for pattern, repl in corrections:
        text = re.sub(pattern, repl, text)
    return text

# --- Improved Text Cleaning Function ---
def clean_text(text):
    import string
    # Remove only non-printable/control characters, keep all normal English and punctuation
    printable = set(string.printable)
    cleaned = ''.join(c if c in printable else ' ' for c in text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

# --- Improved URL Extraction Function ---
def extract_urls(text):
    # Strict URL regex: http(s), www, or bare domain with TLD (2-10 chars)
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

# --- Heuristic Check: Is URL Suspicious? ---
def is_suspicious_url(url):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    score = score_url_suspicion(url)

    if score < 3:
        return False
    if score == 3 and contains_uuid(path):
        print(f"[INFO] UUID detected in path → skipping suspicious flag for: {url}")
        return False
    # score > 3, or score == 3 and no UUID
    return True

# --- Heuristic Scoring Function for URLs ---
def score_url_suspicion(url):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
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
    return score  # Score from 0 to 5

def contains_uuid(path):
    # Standard UUID regex (8-4-4-4-12 hex digits)
    uuid_regex = re.compile(
        r'[a-fA-F0-9]{8}-'
        r'[a-fA-F0-9]{4}-'
        r'[a-fA-F0-9]{4}-'
        r'[a-fA-F0-9]{4}-'
        r'[a-fA-F0-9]{12}'
    )
    return bool(uuid_regex.search(path))

# --- OCR Line Filtering Function ---
def filter_ocr_lines(text):
    import re

    ui_words = [
        "search", "menu", "settings", "sign in", "sign up", "q ", "pm", "am",
        "goodreads", "linkedin", "bookmark", "profile", "202"
    ]
    date_time_patterns = [
        r'\b\d{1,2}:\d{2}\b',                # 12:34
        r'\b\d{1,2}:\d{2}\s?(am|pm)\b',      # 12:34 am
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',      # 12/31/2023
        r'\b\d{4}-\d{2}-\d{2}\b',            # 2023-12-31
        r'\b\d{1,2}\s?(am|pm)\b',            # 3 pm
        r'\b\d{4}\b',                        # 2023
    ]
    english_word_pattern = re.compile(r'[a-zA-Z]{2,}')

    filtered_lines = []
    for line in text.splitlines():
        line_stripped = line.strip()
        line_lower = line_stripped.lower()

        # Remove empty lines
        if not line_stripped:
            print(f"[FILTER] Removed (empty line): '{line}'")
            continue

        # Remove lines that are mostly numbers or special chars
        if len(line_stripped) >= 4:
            num_ratio = len(re.sub(r'[^0-9]', '', line_stripped)) / len(line_stripped)
            special_ratio = len(re.sub(r'[\w]', '', line_stripped)) / len(line_stripped)
            if num_ratio > 0.7:
                print(f"[FILTER] Removed (mostly numbers): '{line}'")
                continue
            if special_ratio > 0.7:
                print(f"[FILTER] Removed (mostly special chars): '{line}'")
                continue

        # Remove lines with UI/menu words
        if any(word in line_lower for word in ui_words):
            print(f"[FILTER] Removed (UI/menu word): '{line}'")
            continue

        # Remove lines with date/time patterns
        if any(re.search(pattern, line_lower) for pattern in date_time_patterns):
            print(f"[FILTER] Removed (date/time pattern): '{line}'")
            continue

        # Remove lines that are too short (less than 4 chars)
        if len(line_stripped) < 4:
            print(f"[FILTER] Removed (too short): '{line}'")
            continue

        # Allow lines with at least one English word, or if not obvious noise
        if english_word_pattern.search(line_stripped):
            filtered_lines.append(line_stripped)
        else:
            print(f"[FILTER] Removed (no English word): '{line}'")

    return '\n'.join(filtered_lines)

def minimal_ocr_line_filter(text):
    lines = text.splitlines()
    # If only one line, keep as is
    if len(lines) <= 1:
        filtered = text
    else:
        filtered = '\n'.join([line for line in lines if len(line.strip()) >= 4])
    return filtered

def process_text(text, timestamp, tokenizer, model, softmax, suspicious_keywords, PHISHING_THRESHOLD, CHUNK_SIZE, BACKUP_THRESHOLD):
    raw_text = text  # Save the original, unfiltered text for logging

    # --- Minimal OCR line filtering ---
    text = minimal_ocr_line_filter(text)
    print(f"\n[DEBUG] Filtered text to be sent to model:\n{text}\n")
    # --- OCR Correction ---
    text = correct_ocr_errors(text)
    # --- Clean Text ---
    text = clean_text(text)

    # --- Extract and Analyze URLs from Text ---
    urls = extract_urls(text)
    phishing_keywords_extra = [
        'urgent', 'suspend', 'unauthorized', 'action required', 'limited time',
        'reset', 'security alert', 'reactivate', 'access', 'billing', 'invoice',
        'credentials', 'verify identity'
    ]
    all_phishing_keywords = suspicious_keywords + phishing_keywords_extra
    words = text.lower().split()
    found_keywords = [word for word in words if word in all_phishing_keywords]
    suspicious_urls = [u for u in urls if is_suspicious_url(u)]

    # --- Count unique domains ---
    unique_domains = set()
    for url in urls:
        try:
            domain = urlparse(url).netloc.lower()
            if domain:
                unique_domains.add(domain)
        except Exception:
            continue

    print(f"\n[DEBUG] URLs found: {urls}")
    print(f"[DEBUG] Suspicious URLs: {suspicious_urls}")
    print(f"[DEBUG] Keywords found: {found_keywords}")

    # --- Classify Text using DistilBERT in Chunks ---
    print(f"\n[DEBUG] Text sent to model:\n{text}\n")
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=CHUNK_SIZE)["input_ids"][0]
    chunks = [tokens[i:i+CHUNK_SIZE] for i in range(0, len(tokens), CHUNK_SIZE)]

    def classify_chunk(chunk):
        # Ensure chunk doesn't exceed max length
        if chunk.size(0) > CHUNK_SIZE:
            print(f"[WARNING] Chunk size {chunk.size(0)} exceeds maximum {CHUNK_SIZE}, truncating...")
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

    print(f"[DEBUG] Model confidence: {max_confidence:.4f}")

    # --- Final Decision Logic: Require ALL indicators ---
    print("\nPrediction Result:")
    if (
        final_prediction == 1
        and (max_confidence >= PHISHING_THRESHOLD)
        and found_keywords
        and suspicious_urls
    ):
        print("PHISHING DETECTED!")
        print(f"  Triggering keywords: {found_keywords}")
        print(f"  Triggering suspicious URLs: {suspicious_urls}")
        print(f"Max Confidence: {max_confidence * 100:.2f}%")
        label = "phishing"
    else:
        print("SAFE (Did not meet all phishing criteria: model, keyword, and suspicious URL).")
        print(f"  Found keywords: {found_keywords}")
        print(f"  Found suspicious URLs: {suspicious_urls}")
        print(f"Max Confidence: {max_confidence * 100:.2f}%")
        label = "safe"

    # --- Log for retraining (ALWAYS log, both classes) ---
    log_for_retraining(raw_text, label, timestamp)

    # --- Anomaly Detection: Collect and send metadata ---
    scan_time = time.time()  # You can use a more precise scan duration if available
    anomaly.add_scan_metadata(
        timestamp=timestamp,
        num_unique_urls=len(set(urls)),
        num_keywords=len(found_keywords),
        num_domains=len(unique_domains),
        scan_time=scan_time
    )
    is_anomaly, anomaly_reason = anomaly.is_last_anomaly()

    # --- Exposure logic for display/export ---
    event = {
        "timestamp": timestamp,
        "phishing_status": "PHISHING DETECTED" if label == "phishing" else "SAFE",
        "suspicious_urls": suspicious_urls,
        "summary": f"Keywords: {found_keywords}, Confidence: {max_confidence:.2f}",
        "model_confidence": max_confidence,
        "detected_keywords": found_keywords,
        "all_urls": urls,
        "full_text": text,
        "anomaly": is_anomaly,
        "anomaly_reason": anomaly_reason,
    }

    # --- Real-time anomaly alert ---
    role = get_user_role()
    if is_anomaly:
        if role == "admin":
            print("\n[ANOMALY ALERT - ADMIN]")
            print(f"Anomaly detected at {timestamp}!\nReason: {anomaly_reason}")
            print(f"Metadata: URLs={len(set(urls))}, Keywords={len(found_keywords)}, Domains={len(unique_domains)}, ScanTime={scan_time}")
        else:
            print("\n[ANOMALY ALERT]")
            print("Unusual activity detected on your computer. If this wasn't you, please check your security settings.")

    # To display for the current user:
    print(format_event(event))

    # To export to Splunk:
    splunk_data = export_for_splunk(event)
    # send splunk_data to Splunk

    # To save to CSV, you can use filter_event(event) to get the right fields

def log_for_retraining(text, label, source):
    os.makedirs('flagged_cases', exist_ok=True)
    csv_path = os.path.join('flagged_cases', 'phishing_log.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['text', 'label', 'source'])
        writer.writerow([
            text.replace('\n', ' ')[:5000],  # Cleaned OCR text, truncated for safety
            label,
            source
        ])

is_scanning = False

def run_phishing_detection(queue=None, mode="production"):
    """
    Main phishing detection logic.
    - In "debug" mode, runs one-time scan.
    - In "production" mode, runs in a time-limited loop (e.g., 15s).
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained("tokenizer")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    previous_hash = None
    previous_text = None
    last_capture_time = 0
    start_time = time.time()

    # In debug mode, run only once. In production, run for a limited time (e.g., 15s).
    run_duration = 0.1 if mode == "debug" else 15

    while time.time() - start_time < run_duration:
        now = time.time()
        if now - last_capture_time < 2:
            time.sleep(0.5)
            continue

        screenshot = pyautogui.screenshot()
        current_hash = imagehash.average_hash(screenshot)

        # OCR
        image = screenshot.convert("RGB")
        text = pytesseract.image_to_string(image, config=PSM_MODE, lang='eng')
        text = clean_text(text)

        if len(text.strip()) < 10:
            print("[INFO] OCR text too short/empty, skipping this capture.")
            previous_hash = current_hash
            time.sleep(0.5)
            continue

        similarity = 0
        if previous_text is not None:
            similarity = difflib.SequenceMatcher(None, text.strip(), previous_text.strip()).ratio()

        if (previous_hash is not None and current_hash == previous_hash) or \
           (previous_text is not None and (text.strip() == previous_text.strip() or similarity > 0.95)):
            print("[INFO] Screen or OCR unchanged (or nearly unchanged), skipping capture.")
            previous_hash = current_hash
            time.sleep(0.5)
            continue

        # No saving to disk here!
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Scanned at {timestamp}")

        previous_hash = current_hash
        previous_text = text
        last_capture_time = now

        process_text(
            text, timestamp, tokenizer, model, softmax,
            suspicious_keywords, PHISHING_THRESHOLD, CHUNK_SIZE, BACKUP_THRESHOLD
        )

        time.sleep(0.5)

        # In debug mode, exit after the first successful scan
        if mode == "debug":
            print("[INFO] Debug mode: single scan complete.")
            break

    # At the end, optionally put a log/result in the queue:
    if queue is not None:
        queue.put("Detection finished")  # Or put more detailed info if you want

last_scan_time = 0
COOLDOWN_SECONDS = 10  # Adjust as needed
last_active_window = None

last_event_time = time.time()
INACTIVITY_TIMEOUT = 15  # seconds

def on_user_activity(event_type=None):
    global is_scanning, last_event_time
    last_event_time = time.time()
    if is_scanning:
        print(f"[EVENT] {event_type} detected, but scan already in progress.")
        return
    print(f"[EVENT] {event_type} detected, running phishing detection.")
    is_scanning = True
    try:
        run_phishing_detection()
    finally:
        is_scanning = False

def monitor_active_window():
    global last_active_window
    while True:
        try:
            active_window = gw.getActiveWindowTitle()
            if active_window != last_active_window:
                last_active_window = active_window
                on_user_activity("Active window change")
        except Exception as e:
            pass
        time.sleep(5)  # Check every 5 seconds

# Keyboard listener
def on_press(key):
    on_user_activity("Key press")

def inactivity_fallback():
    global last_event_time, is_scanning, last_scan_time
    while True:
        now = time.time()
        if now - last_event_time > INACTIVITY_TIMEOUT and not is_scanning:
            print("[FALLBACK] No user activity detected for 15s, running phishing detection.")
            last_event_time = now
            last_scan_time = now
            is_scanning = True
            try:
                run_phishing_detection()
            finally:
                is_scanning = False
        time.sleep(5)

def get_run_mode():
    """
    Determines the run mode from an environment variable or prompts the user.
    If SAFENET_MODE is set to 'production' or 'debug', it runs automatically.
    Otherwise, it prompts the user to choose a mode at runtime.
    """
    mode = os.getenv("SAFENET_MODE", "").lower()
    if mode in ["production", "debug"]:
        print(f"[INFO] Running in '{mode}' mode (from SAFENET_MODE).")
        return mode

    # If the environment variable is not set, always prompt the user.
    while True:
        choice = input(
            "Choose run mode:\n"
            "1. Production (continuous monitoring)\n"
            "2. Debug (single scan)\n"
            "Enter choice (1 or 2): "
        )
        if choice == "1":
            print("[INFO] Running in 'production' mode.")
            return "production"
        elif choice == "2":
            print("[INFO] Running in 'debug' mode.")
            return "debug"
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    run_mode = get_run_mode()

    if run_mode == "debug":
        # Run a single, one-off scan for debugging
        print("\n[DEBUG MODE] Starting a single scan...")
        run_phishing_detection(mode="debug")
        print("[DEBUG MODE] Scan finished. Exiting.")
        os._exit(0)

    # --- Production Mode ---
    print("\n[PRODUCTION MODE] Starting continuous monitoring...")
    # Start the global timer
    def global_timeout_handler():
        print("\n[INFO] Global runtime limit (30 seconds) reached. Exiting.")
        os._exit(0)  # Immediately exit the process
    threading.Timer(30, global_timeout_handler).start()

    # Start keyboard listener and window monitor
    keyboard_listener = pynput.keyboard.Listener(on_press=on_press)
    keyboard_listener.start()

    window_thread = threading.Thread(target=monitor_active_window, daemon=True)
    window_thread.start()

    print("Event-driven phishing detection system started. Waiting for user activity...")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Exiting.")

    # Add inactivity_fallback thread as above and start it in __main__
    fallback_thread = threading.Thread(target=inactivity_fallback, daemon=True)
    fallback_thread.start()
