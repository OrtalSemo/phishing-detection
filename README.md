# SafeNet – Real-Time Phishing Detection

SafeNet is a real-time phishing detection system that monitors the screen, extracts visible text using OCR (Tesseract), and analyzes it using a DistilBERT-based classifier to detect phishing attempts.  
When suspicious content is found, the system can alert the user or log the case for review.

---

## 🧠 How It Works

1. **Event Triggered**: The system detects user activity (e.g., scrolling, typing).
2. **Screenshot**: It captures the current screen.
3. **OCR**: Text is extracted using Tesseract.
4. **Analysis**: The text is chunked and sent to a DistilBERT model for phishing classification.
5. **Decision**: If phishing is detected (above a threshold), it logs or alerts.
6. **Optional backup**: Uncertain cases can be re-analyzed via OpenAI API.

---

## 📁 Key Files and Structure

| Path                           | Description                                          |
|--------------------------------|------------------------------------------------------|
| `src/main.py`                  | Entry point: real-time monitoring and detection      |
| `src/analyze_safenet.py`       | System test script    |
| `train_distilbert_phishing.py` | Train DistilBERT phishing model                      |
| `models/saved_model/`          | Directory for trained model files                    |
| `tokenizer/`                   | Tokenizer for DistilBERT                             |
| `src/anomaly_detection.py`     | Utilities for anomaly-based heuristics               |
| `src/event_exposure.py`        | Functions for logging & exposure tracking            |

---

## ⚙️ Requirements

- Python 3.8 or higher  
- Tesseract OCR installed at:  
  `C:\Program Files\Tesseract-OCR\tesseract.exe`
- cp env.example .env  # Edit to add OpenAI API key if needed
---
