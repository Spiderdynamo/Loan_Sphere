"""
=================================================================================
 bert_model.py - AI transaction classifier
=================================================================================

WHAT THIS FILE DOES
--------------------
This module loads a pretrained transformer model and uses it to read a bank
transaction line (e.g. "UPI/DR/50231/Zomato Foods/450.00") and decide which
spending category it belongs to (Food, Travel, Shopping, Salary, etc.)
WITHOUT ever being specifically trained on bank statements. This technique is
called "zero-shot classification": we give the model a list of candidate
category names at run time, and it scores how well the transaction text
matches each one.

MODEL USED
----------
We use "typeform/distilbert-base-uncased-mnli", a DistilBERT model (a
smaller, faster distillation of BERT) that has been fine-tuned on the MNLI
(natural language inference) dataset. Hugging Face's `pipeline(task=
"zero-shot-classification")` uses that NLI fine-tuning to test how strongly
each candidate label is "entailed" by the transaction text. DistilBERT keeps
the download small (~260MB) and inference fast on a normal laptop CPU, which
matters here because every transaction line in a bank statement is
classified individually.

If you have a GPU and want higher accuracy at the cost of a larger download,
swap MODEL_NAME below for "facebook/bart-large-mnli".

WHY THIS IS SEPARATE FROM app.py
---------------------------------
Keeping the model-loading code in its own file means the (slow) model is
only loaded once, and app.py stays focused on the web/Flask logic instead of
being cluttered with AI plumbing.
=================================================================================
"""

from transformers import pipeline

# ---------------------------------------------------------------------------
# MODEL SETUP
# ---------------------------------------------------------------------------
# This line downloads the model the first time the app runs (it is then
# cached locally by the `transformers` library) and loads it into memory.
# This happens once, when the Flask app starts, not once per request.
MODEL_NAME = "typeform/distilbert-base-uncased-mnli"

print(f"[bert_model] Loading zero-shot classifier '{MODEL_NAME}' ...")
classifier = pipeline("zero-shot-classification", model=MODEL_NAME)
print("[bert_model] Classifier ready.")

# The fixed set of spending categories the model is allowed to choose from.
# Feel free to tailor this list to your own bank's statement format.
CATEGORIES = [
    "Food",
    "Shopping",
    "Travel",
    "Medical",
    "Fuel",
    "Entertainment",
    "Electricity",
    "Recharge",
    "Salary",
    "Investment",
    "Transfer",
    "Cash Withdrawal",
    "Other",
]


def classify_transaction(transaction_text):
    """
    Classify a single transaction line into one of CATEGORIES.

    Returns a dict:
        {
            "category":   the best-matching category (str),
            "confidence": how confident the model is, as a 0-100 percentage,
        }
    """
    result = classifier(transaction_text, candidate_labels=CATEGORIES)

    return {
        "category": result["labels"][0],       # highest scoring label
        "confidence": round(result["scores"][0] * 100, 2),
    }


def analyze_bank_statement(text):
    """
    Walk through every line of a bank statement's extracted text, skip
    headers/footers/blank lines, and run the AI classifier on anything that
    looks like an actual transaction (contains a payment-network keyword
    such as UPI, NEFT, IMPS, ATM, POS, etc).

    Returns a list of dicts: [{"transaction": ..., "category": ..., "confidence": ...}, ...]
    """
    output = []
    lines = text.split("\n")

    # Keywords that let us skip statement headers/footers/summary rows so we
    # only classify genuine transaction lines.
    SKIP_KEYWORDS = ("balance", "statement", "account", "branch", "ifsc", "page")

    # Keywords that indicate a line IS a genuine transaction.
    TRANSACTION_KEYWORDS = ("UPI", "ATM", "NEFT", "IMPS", "TRANSFER", "POS", "RTGS", "CHQ")

    for raw_line in lines:
        line = raw_line.strip()

        if len(line) < 3:
            continue

        if any(keyword in line.lower() for keyword in SKIP_KEYWORDS):
            continue

        if any(keyword in line.upper() for keyword in TRANSACTION_KEYWORDS):
            prediction = classify_transaction(line)
            output.append({
                "transaction": line,
                "category": prediction["category"],
                "confidence": prediction["confidence"],
            })

    return output


def summarize_document(text, max_sentences=5):
    """
    A lightweight, rule-based summary: just grabs the first few non-empty
    sentences from the extracted text. This is intentionally simple; swap
    in a dedicated summarization pipeline (e.g. "facebook/bart-large-cnn")
    here if you want a true abstractive summary later.
    """
    sentences = text.split(".")
    summary_parts = [s.strip() for s in sentences if s.strip()][:max_sentences]
    return ". ".join(summary_parts) + ("." if summary_parts else "")


def analyze_salary(text):
    """
    Small helper kept for future salary-slip intelligence: flags whether
    key salary-slip fields were found anywhere in the OCR'd text.
    """
    lowered = text.lower()
    return {
        "salary_found": "basic salary" in lowered,
        "tax_detected": "professional tax" in lowered,
    }
