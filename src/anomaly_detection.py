import time
from collections import deque

# Number of recent scans to keep for baseline
HISTORY_SIZE = 20

# Each entry: dict with keys: timestamp, num_unique_urls, num_keywords, num_domains, scan_time
_scan_history = deque(maxlen=HISTORY_SIZE)
_last_anomaly = False
_last_reason = ""

def add_scan_metadata(timestamp, num_unique_urls, num_keywords, num_domains, scan_time):
    """
    Add metadata for a scan/session.
    """
    global _last_anomaly, _last_reason
    entry = {
        "timestamp": timestamp,
        "num_unique_urls": num_unique_urls,
        "num_keywords": num_keywords,
        "num_domains": num_domains,
        "scan_time": scan_time,
    }
    _scan_history.append(entry)
    _last_anomaly, _last_reason = detect_anomaly(entry)

def get_recent_history():
    """
    Return a list of recent scan metadata.
    """
    return list(_scan_history)

def detect_anomaly(current_entry):
    """
    Detect if the current scan/session is anomalous compared to recent history.
    Flags as anomalous if the number of unique URLs, suspicious keywords, or domains
    is more than 2x the recent average (excluding the current scan).
    Returns (is_anomaly: bool, reason: str)
    """
    if len(_scan_history) < 5:
        return False, "Not enough history for anomaly detection."

    # Exclude the current entry for baseline
    history = list(_scan_history)[:-1]
    avg_urls = sum(e["num_unique_urls"] for e in history) / len(history)
    avg_keywords = sum(e["num_keywords"] for e in history) / len(history)
    avg_domains = sum(e["num_domains"] for e in history) / len(history)

    urls_anomaly = current_entry["num_unique_urls"] > 2 * max(1, avg_urls)
    keywords_anomaly = current_entry["num_keywords"] > 2 * max(1, avg_keywords)
    domains_anomaly = current_entry["num_domains"] > 2 * max(1, avg_domains)

    if urls_anomaly:
        return True, "Unusual spike in suspicious URLs detected."
    if keywords_anomaly:
        return True, "Unusual spike in suspicious keywords detected."
    if domains_anomaly:
        return True, "Unusual spike in unique domains detected."
    return False, ""

def is_last_anomaly():
    """
    Return (is_anomaly: bool, reason: str) for the last scan/session.
    """
    return _last_anomaly, _last_reason
