# Loan Sphere — AI Loan Document Verification

A dark-themed (gold ✕ neon-blue) Flask web app that verifies loan applicant
documents. Upload a **bank statement** or a **payslip**, and the app will:

- Extract the text (PDF text extraction for bank statements, OCR for payslips)
- Classify every bank transaction into a spending category using a
  **BERT-family zero-shot classifier** (`typeform/distilbert-base-uncased-mnli`)
- Check payslip salary fields against a simple eligibility rule
- Render a **pie chart** and a **bar chart** of the results, themed to match
  the site

## Screens

1. **Login** (`/`) — gold/neon-blue glassmorphism login card.
2. **Document Verification** (`/upload`) — dropdown to choose *Payslip* or
   *Bank Statement*, a file input, and a "Run Verification" button. Results
   (charts, tables, eligibility) render on the same page.

## Project layout

```
LOAN_VERIFICATION/
├── app.py                 # Flask app: routes, OCR, charts, login/session
├── bert_model.py          # BERT-family zero-shot transaction classifier
├── generate_bank_data.py  # optional dev helper: makes a fake CSV statement
├── requirements.txt
├── static/
│   ├── css/style.css      # dark / gold / neon-blue theme
│   ├── js/script.js       # loader overlay, flash auto-dismiss, etc.
│   └── charts/            # generated chart PNGs land here (gitignored)
├── templates/
│   ├── login.html
│   └── upload.html
└── uploads/                # uploaded documents land here (gitignored)
```

## Setup

```bash
git clone <this-repo-url>
cd LOAN_VERIFICATION
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

### Demo login
```
username: admin
password: admin123
```
Change this in `DEMO_USERS` inside `app.py` before deploying anywhere real —
ideally swap it out for a real user database entirely.

## Notes

- The BERT classifier and EasyOCR both download their model weights the
  first time you run the app — this needs an internet connection and can
  take a few minutes.
- Bank statement PDFs must contain selectable text (not a scanned image) so
  `pdfplumber` can extract it — a scanned statement would need an OCR step
  added in `app.py`.
- Payslip field extraction looks for the labels "Basic Salary", "House Rent
  Allowance(s)" and "Professional Tax" — adjust the regular expressions in
  `_process_salary_slip()` inside `app.py` if your payslip format differs.
- Charts are generated as base64 PNGs and embedded directly in the page, so
  there's no risk of an old chart image being cached by the browser.

## Tech stack

Flask · pdfplumber · EasyOCR · Hugging Face Transformers (DistilBERT/MNLI) ·
Matplotlib · Werkzeug security helpers
