"""
=================================================================================
 LOAN SPHERE - AI POWERED LOAN DOCUMENT VERIFICATION SYSTEM
=================================================================================

WHAT THIS FILE DOES
--------------------
This is the main Flask application (the "backend"). It is responsible for:

    1. Serving the dark-themed login page and checking the user's credentials.
    2. Serving the document verification (upload) page after a successful login.
    3. Receiving an uploaded PDF/image (either a "Bank Statement" or a
       "Payslip / Salary Slip") from the upload page.
    4. Reading the text out of that document (OCR / PDF text extraction).
    5. Running the extracted text through an AI model (see bert_model.py) that
       classifies each bank transaction into a spending category
       (Food, Shopping, Travel, Salary, etc.) using a BERT-family zero-shot
       classifier.
    6. Building a Pie Chart AND a Bar Chart for whichever document was
       uploaded, styled to match the site's dark / gold / neon-blue theme.
    7. Sending everything back to the upload page so the results render on
       the screen.

HOW THE PAGES ARE CONNECTED
----------------------------
    "/"        -> login.html   (attractive dark login screen)
    "/login"   -> POST target for the login form, validates credentials
    "/upload"  -> upload.html  (dropdown: Payslip / Bank Statement + file input)
    "/verify"  -> POST target for the upload form, runs OCR + AI + charts
    "/logout"  -> clears the session and sends the user back to the login page

Every route except "/" and "/login" is protected by a small `login_required`
decorator, so nobody can jump straight to the verification page without
logging in first.

RUNNING THE APP
----------------
    1. pip install -r requirements.txt
    2. python app.py
    3. Open http://127.0.0.1:5000 in your browser
    4. Log in with the demo credentials printed in the terminal on startup
       (username: admin / password: admin123 - change these in production!)
=================================================================================
"""

import os
import re
import io
import base64
import functools

import pdfplumber                     # extracts text from digital (non-scanned) PDFs
import easyocr                        # OCR engine used to read text out of payslip images/PDFs
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from werkzeug.security import generate_password_hash, check_password_hash

import matplotlib
matplotlib.use("Agg")                 # headless backend - needed because Flask has no display
import matplotlib.pyplot as plt

# Our own module that wraps the BERT-family zero-shot-classification model.
from bert_model import analyze_bank_statement, summarize_document


# ---------------------------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------------------------
app = Flask(__name__)

# The secret key is required so Flask can safely sign the session cookie
# that keeps a user "logged in" between page loads.  In a real deployment
# this MUST be a long random string kept outside of source control (for
# example loaded from an environment variable) - it is left as a plain
# string here only so the project runs immediately after a git clone.
app.secret_key = os.environ.get("LOAN_SPHERE_SECRET_KEY", "dev-only-change-me-before-deploying")

# Folder where uploaded documents are temporarily stored.
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Folder where the generated pie / bar chart PNGs are written to so the
# browser can display them with a normal <img> tag.
CHARTS_FOLDER = os.path.join(app.root_path, "static", "charts")
os.makedirs(CHARTS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}


# ---------------------------------------------------------------------------
# DEMO USER "DATABASE"
# ---------------------------------------------------------------------------
# For a real product you would replace this with a proper database table
# (e.g. SQLAlchemy + a Users model) and never store plaintext passwords.
# The password below is hashed with werkzeug's secure hashing helper so
# that even in this demo the raw password is not sitting in memory/plaintext.
DEMO_USERS = {
    "admin": generate_password_hash("admin123"),
}


# ---------------------------------------------------------------------------
# EASY-OCR READER (loaded once, at startup, so every upload is fast)
# ---------------------------------------------------------------------------
# gpu=False keeps this working on machines without a CUDA GPU (most laptops).
# Loading the model takes a few seconds the first time the app starts.
print("[Loan Sphere] Loading OCR engine, please wait...")
ocr_reader = easyocr.Reader(["en"], gpu=False)
print("[Loan Sphere] OCR engine ready.")


# ---------------------------------------------------------------------------
# TESSERACT CONFIGURATION (Windows only)
# ---------------------------------------------------------------------------
# pytesseract looks for the "tesseract" binary on the system PATH by default,
# which works fine on Linux/Mac once Tesseract is installed via the package
# manager. On Windows, Tesseract is not automatically added to PATH, so we
# point pytesseract at the default install location - but only if that file
# actually exists, so this code does not crash on Linux/Mac/CI.
try:
    import pytesseract
    _WINDOWS_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and os.path.exists(_WINDOWS_TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = _WINDOWS_TESSERACT_PATH
except ImportError:
    # pytesseract is optional in this build because easyocr already handles
    # the payslip OCR; it's kept as a soft dependency for anyone who wants
    # to extend the project with Tesseract-based extraction.
    pytesseract = None


# ---------------------------------------------------------------------------
# COLOR PALETTE FOR CHARTS - matches the site's dark / gold / neon-blue theme
# ---------------------------------------------------------------------------
CHART_BG = "#0b1220"          # deep navy-black background
CHART_TEXT = "#e8ecf4"        # near-white text
GOLD = "#e8c15a"
NEON_BLUE = "#33e1ff"
CHART_PALETTE = [
    "#e8c15a",  # gold
    "#33e1ff",  # neon blue
    "#8f6b1f",  # deep gold
    "#1c7ea8",  # deep blue
    "#f2d98a",  # pale gold
    "#7be7ff",  # pale neon blue
    "#c99b2e",  # bronze gold
    "#0fb6d6",  # teal blue
    "#ffe9a8",  # cream gold
    "#5fd1ea",  # light aqua blue
]


def _style_dark_axes(fig, ax):
    """Apply the dark/gold/neon-blue theme to a matplotlib figure + axes."""
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.title.set_color(GOLD)
    ax.xaxis.label.set_color(CHART_TEXT)
    ax.yaxis.label.set_color(CHART_TEXT)
    ax.tick_params(colors=CHART_TEXT)
    for spine in ax.spines.values():
        spine.set_color("#243049")


def _fig_to_base64(fig):
    """
    Convert a matplotlib figure straight into a base64 PNG string so it can
    be embedded directly in the HTML with <img src="data:image/png;base64,...">.
    This avoids any browser image-caching issues where an old chart would
    otherwise appear to "stick" after a new document is analyzed.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=140)
    plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def make_pie_chart(labels, values, title):
    """Build a themed pie chart and return it as a base64 data-URI string."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    _style_dark_axes(fig, ax)
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=CHART_PALETTE[: len(values)],
        textprops={"color": CHART_TEXT, "fontsize": 10},
        wedgeprops={"edgecolor": CHART_BG, "linewidth": 1.5},
    )
    for autotext in autotexts:
        autotext.set_color(CHART_BG)
        autotext.set_fontweight("bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    return _fig_to_base64(fig)


def make_bar_chart(labels, values, title, y_label="Amount (₹)"):
    """Build a themed bar chart and return it as a base64 data-URI string."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _style_dark_axes(fig, ax)
    bars = ax.bar(labels, values, color=CHART_PALETTE[: len(values)], edgecolor=NEON_BLUE, linewidth=1.2)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", color="#243049", linestyle="--", linewidth=0.6, alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    # Put the value on top of each bar for readability.
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color=CHART_TEXT,
            fontsize=8,
        )
    return _fig_to_base64(fig)


_AMOUNT_RE = re.compile(r"\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?")


def _extract_amount(description):
    """
    Pull the transaction amount out of a bank-statement line.

    The old implementation only matched a number sitting at the absolute
    end of the line, which fails on real statements where the amount is
    followed by a "Dr"/"Cr" indicator, a currency symbol, or a running
    balance (e.g. "UPI/DR/998877/BigBasket/1,899.00 Dr"). That caused every
    transaction to silently fall back to 0.0, which in turn made the pie
    chart crash with "All wedge sizes are zero".

    This version finds every comma/decimal-formatted number in the line and
    takes the LAST one, which is where the transaction amount normally sits
    (right before any trailing Dr/Cr marker or balance column). Thousands
    commas are stripped before conversion to float.
    """
    matches = _AMOUNT_RE.findall(description)
    if not matches:
        return 0.0
    return float(matches[-1].replace(",", ""))


def allowed_file(filename):
    """Only accept the document types this project knows how to read."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# LOGIN-REQUIRED DECORATOR
# ---------------------------------------------------------------------------
def login_required(view_func):
    """Redirects to the login page if there is no active session."""
    @functools.wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped_view


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/", methods=["GET"])
def login():
    """Show the dark-themed login page (this is the site's home page)."""
    # If the user is already logged in, skip straight to the dashboard.
    if session.get("logged_in"):
        return redirect(url_for("upload"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def do_login():
    """Validate the submitted credentials and start a session."""
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    stored_hash = DEMO_USERS.get(username)

    if stored_hash and check_password_hash(stored_hash, password):
        session["logged_in"] = True
        session["username"] = username
        return redirect(url_for("upload"))

    flash("Invalid username or password. Please try again.", "error")
    return redirect(url_for("login"))


@app.route("/logout")
def logout():
    """Clear the session and return to the login screen."""
    session.clear()
    return redirect(url_for("login"))


@app.route("/upload", methods=["GET"])
@login_required
def upload():
    """
    Show the document verification page. This page contains the dropdown
    where the user chooses "Payslip" or "Bank Statement" and the file input
    used to upload that document.
    """
    return render_template(
        "upload.html",
        username=session.get("username"),
        piechart=None,
        barchart=None,
        transaction_html="",
        summary="",
        status="",
        basic_salary="",
        hra_salary="",
        tax_amount="",
        doctype="",
    )


@app.route("/verify", methods=["POST"])
@login_required
def verify():
    """
    Handle the uploaded file: run OCR/text-extraction, classify the content
    with the BERT model, build the pie + bar charts, and re-render the
    upload page with the results filled in.
    """
    uploaded_file = request.files.get("file")
    doctype = request.form.get("doctype")

    if not uploaded_file or uploaded_file.filename == "":
        flash("Please choose a file before clicking Run Verification.", "error")
        return redirect(url_for("upload"))

    if not allowed_file(uploaded_file.filename):
        flash("Unsupported file type. Please upload a PDF, PNG or JPG.", "error")
        return redirect(url_for("upload"))

    filename = uploaded_file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(filepath)

    # -------------------------------------------------------------------
    # BANK STATEMENT PATH
    # -------------------------------------------------------------------
    if doctype == "bank":
        return _process_bank_statement(filepath)

    # -------------------------------------------------------------------
    # PAYSLIP / SALARY SLIP PATH
    # -------------------------------------------------------------------
    elif doctype == "salary":
        return _process_salary_slip(filepath)

    flash("Please select a document type from the dropdown.", "error")
    return redirect(url_for("upload"))


# -----------------------------------------------------------------------
# HELPER: BANK STATEMENT PROCESSING
# -----------------------------------------------------------------------
def _process_bank_statement(filepath):
    """
    1. Pull raw text out of the bank statement PDF with pdfplumber.
    2. Send that text to the BERT zero-shot classifier (bert_model.py) which
       labels every transaction line with a spending category.
    3. Total the amounts per category and build a pie chart + a bar chart.
    """
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if not text.strip():
        flash("Could not read any text from that PDF. Is it a scanned image?", "error")
        return redirect(url_for("upload"))

    # Run the AI classifier over every transaction line.
    ai_results = analyze_bank_statement(text)

    # A short, human-readable summary of the statement (first few lines).
    summary = summarize_document(text)

    # Build a (category, amount) list from the AI results so we can total
    # things up per category for the charts.
    transactions = []
    for item in ai_results:
        description = item["transaction"]
        category = item["category"]
        amount = _extract_amount(description)

        transactions.append((category, amount))

    category_totals = {}
    for category, amount in transactions:
        category_totals[category] = category_totals.get(category, 0) + amount

    # Build the HTML rows for the transaction table shown on the page.
    transaction_html = "".join(
        f"<tr><td>{category}</td><td>₹{amount:,.2f}</td></tr>"
        for category, amount in transactions
    )

    piechart = barchart = None
    # Guard against ax.pie()/ax.bar() crashing on an all-zero or empty
    # dataset (e.g. a statement whose amount formatting we still can't
    # parse). Better to render the page without charts than to 500.
    if category_totals and any(v > 0 for v in category_totals.values()):
        labels = list(category_totals.keys())
        values = list(category_totals.values())
        piechart = make_pie_chart(labels, values, "Bank Statement - Spending by Category")
        barchart = make_bar_chart(labels, values, "Bank Statement - Category Totals")
    elif category_totals:
        flash(
            "Transactions were found, but no amounts could be read from them, "
            "so the charts were skipped. The table below still shows what was detected.",
            "error",
        )

    return render_template(
        "upload.html",
        username=session.get("username"),
        piechart=piechart,
        barchart=barchart,
        transaction_html=transaction_html,
        summary=summary,
        status="",
        basic_salary="",
        hra_salary="",
        tax_amount="",
        doctype="bank",
    )


# -----------------------------------------------------------------------
# HELPER: SALARY SLIP / PAYSLIP PROCESSING
# -----------------------------------------------------------------------
def _process_salary_slip(filepath):
    """
    1. OCR the payslip image/PDF with EasyOCR.
    2. Pull out Basic Salary / HRA / Professional Tax using regular
       expressions.
    3. Decide loan eligibility with a simple rule and build a pie chart +
       bar chart of the salary breakdown.
    """
    ocr_lines = ocr_reader.readtext(filepath, detail=0)
    text = " ".join(ocr_lines)

    basic_match = re.search(r"Basic Salary\s*[:\-]?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    hra_match = re.search(r"House Rent Allowance[s]?\s*[:\-]?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    tax_match = re.search(r"Professional Tax\s*[:\-]?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)

    basic_salary = float(basic_match.group(1)) if basic_match else 0.0
    hra_salary = float(hra_match.group(1)) if hra_match else 0.0
    tax_amount = float(tax_match.group(1)) if tax_match else 0.0

    # Simple, easily-tunable eligibility rule for the demo.
    status = "ELIGIBLE ✅" if basic_salary > 20000 else "NOT ELIGIBLE ❌"

    labels, values = [], []
    if basic_salary > 0:
        labels.append("Basic Salary")
        values.append(basic_salary)
    if hra_salary > 0:
        labels.append("HRA")
        values.append(hra_salary)
    if tax_amount > 0:
        labels.append("Professional Tax")
        values.append(tax_amount)

    piechart = barchart = None
    if values:
        piechart = make_pie_chart(labels, values, "Payslip - Salary Breakdown")
        barchart = make_bar_chart(labels, values, "Payslip - Component Comparison")
    else:
        flash(
            "Could not detect salary fields automatically. "
            "Try a clearer scan, or check that the payslip uses the expected labels.",
            "error",
        )

    return render_template(
        "upload.html",
        username=session.get("username"),
        piechart=piechart,
        barchart=barchart,
        transaction_html="",
        summary="",
        status=status,
        basic_salary=f"{basic_salary:,.2f}",
        hra_salary=f"{hra_salary:,.2f}",
        tax_amount=f"{tax_amount:,.2f}",
        doctype="salary",
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" LOAN SPHERE - AI Loan Document Verification")
    print(" Demo login -> username: admin | password: admin123")
    print(" (Change DEMO_USERS in app.py before deploying anywhere real!)")
    print("=" * 70)
    app.run(debug=True)
