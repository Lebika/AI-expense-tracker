from flask import (
    Flask, Response, abort, render_template, request, redirect,
    send_from_directory, session, url_for
)
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
import ssl
import certifi
import re
import os
import csv
import json
import secrets
import uuid
import shutil
import difflib
import threading
import statistics
from datetime import datetime

from text_cleaning import clean_for_model
from dashboard import dashboard_bp
from auth import auth_bp, login_required
from recommend import recommend_bp
from settings import setting
from storage import store
from timeutil import now_local
from categories import CATEGORY_INFO, CATEGORY_NAMES, normalize_category

try:
    import cv2
except ImportError:        # OpenCV ships with EasyOCR, so this is rare
    cv2 = None


# ============================================================
# SSL FIX FOR EASYOCR
# ============================================================

ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# ------------------------------------------------------------
# Security: session secret + CSRF protection for every form POST
# ------------------------------------------------------------

def load_secret_key():
    """SECRET_KEY environment variable if set (use this when hosting),
    otherwise a random key kept in the file .secret_key (keep it private)."""

    key = os.environ.get("SECRET_KEY")

    if key:
        return key

    path = ".secret_key"

    if os.path.exists(path):
        with open(path) as file:
            return file.read().strip()

    key = secrets.token_hex(32)

    with open(path, "w") as file:
        file.write(key)

    return key


app.secret_key = load_secret_key()

ON_RENDER = bool(os.environ.get("RENDER"))          # Render sets this itself
USE_HTTPS = ON_RENDER or setting("FORCE_HTTPS") == "1"

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=USE_HTTPS,           # cookie only travels over HTTPS
    MAX_CONTENT_LENGTH=64 * 1024 * 1024,      # total size of one upload
)

if ON_RENDER:

    # Render sits in front of the app: trust its forwarded address / protocol
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

    # Render's disk is wiped on every deploy, so CSV storage would lose data
    if not store.name.startswith("Mongo"):
        raise RuntimeError(
            "On Render you must set STORAGE=mongo (and MONGODB_URI) in the "
            "Environment settings, otherwise all data is lost on each deploy."
        )


@app.after_request
def security_headers(response):

    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")

    if USE_HTTPS:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")

    return response


@app.before_request
def csrf_protect():

    if request.method == "POST":

        sent = request.form.get("csrf_token", "")
        expected = session.get("csrf_token", "")

        if not sent or not expected or not secrets.compare_digest(sent, expected):
            abort(400, "Your page expired. Please go back, refresh and try again.")


@app.context_processor
def inject_csrf_token():

    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(16)

    return {
        "csrf_token": session["csrf_token"],
        "contact_email": setting("CONTACT_EMAIL", "your-email@example.com"),
        "ocr_enabled": ENABLE_OCR,
    }


# ------------------------------------------------------------
# Public pages and files (no login needed)
# ------------------------------------------------------------

@app.route("/healthz")
def healthz():

    return Response("ok", mimetype="text/plain")


@app.route("/privacy")
def privacy():

    return render_template("privacy.html")


@app.route("/terms")
def terms():

    return render_template("terms.html")


@app.route("/sw.js")
def service_worker():
    """Served from the site root so the installable-app worker covers every page."""

    response = send_from_directory("static", "sw.js", mimetype="application/javascript")
    response.headers["Cache-Control"] = "no-cache"

    return response


# Pages: dashboard (dashboard.py), sign-in + profile (auth.py)
app.register_blueprint(dashboard_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(recommend_bp)     # AI tips at /recommendations

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"}

# Upscale + contrast boost before OCR (helps small / blurry shop bills)
USE_PREPROCESSING = True

# Write debug/<bill_id>.json for every bill: OCR boxes, rows, chosen
# extraction method. Send this file when a bill is extracted wrongly.
SAVE_DEBUG = True
DEBUG_DIR = "debug"

# width_ths: lower = EasyOCR keeps table columns apart instead of
# merging "Rice 1516204 1 172.00" into one box.
OCR_PARAMS = {"width_ths": 0.3}


# ============================================================
# BILL SCANNING (EasyOCR + trained model) IS OPTIONAL
#
# ENABLE_OCR=1  (default)  your own computer: scan bills + manual entry
# ENABLE_OCR=0             small free servers: manual entry only. EasyOCR and
#                          scikit-learn are then never loaded (saves memory).
# ============================================================

ENABLE_OCR = setting("ENABLE_OCR", "1").lower() not in ("0", "false", "no", "off")

# Categories come from categories.py (shared with training + dashboard).
# The manual dropdown shows CATEGORY_INFO as "name - hint".
CATEGORIES = CATEGORY_NAMES

vectorizer = model = reader = None

if ENABLE_OCR:

    import easyocr
    import joblib

    print("Loading expense category model...")

    vectorizer = joblib.load("expenseitems_tfidf_vectorizer.pkl")
    model = joblib.load("expenseitems_category_model.pkl")

    print("Expense category model loaded successfully!")
    print("Loading EasyOCR...")

    reader = easyocr.Reader(['en'])

    print("EasyOCR loaded successfully!")

else:

    print("Bill scanning is OFF (ENABLE_OCR=0): manual entry only.")


# Where bills are saved (CSV files or MongoDB Atlas) is decided in storage.py
# by STORAGE=csv|mongo in the .env file.

# Predictions below this confidence (%) are flagged for the user to check
REVIEW_THRESHOLD = 60.0


# ============================================================
# 1. OCR  (with bounding boxes, so table columns can be used)
#
# readtext(detail=0) returns plain text only and LOSES the column
# layout. We keep every box (text + x/y position) so we can find
# the "Items / Products / Description of Goods" column.
# ============================================================

MIN_OCR_CONFIDENCE = 0.10


def ocr_to_cells(result):
    """EasyOCR result [(bbox, text, conf), ...] -> list of cell dicts."""

    cells = []

    for bbox, text, conf in result:

        text = str(text).strip()

        if not text or float(conf) < MIN_OCR_CONFIDENCE:
            continue

        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]

        cells.append({
            "text": text,
            "x1": min(xs),
            "x2": max(xs),
            "y1": min(ys),
            "y2": max(ys),
            "xc": (min(xs) + max(xs)) / 2,
            "yc": (min(ys) + max(ys)) / 2,
            "conf": float(conf)
        })

    return cells


def preprocess_image(image_path):
    """Grayscale + upscale small images + local contrast (CLAHE).
    Returns an image array, or the path itself if OpenCV is unavailable."""

    if not USE_PREPROCESSING or cv2 is None:
        return image_path

    image = cv2.imread(image_path)

    if image is None:
        return image_path

    height, width = image.shape[:2]

    # text height must be large enough for OCR: aim for >= 1200 px wide
    if width < 1200:
        scale = 1200 / width
    elif max(height, width) > 4000:
        scale = 4000 / max(height, width)
    else:
        scale = 1.0

    if scale != 1.0:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def read_cells(image_path):

    return ocr_to_cells(
        reader.readtext(preprocess_image(image_path), **OCR_PARAMS)
    )


def group_into_rows(cells):
    """Group boxes that sit on the same horizontal line into rows,
    each row sorted left -> right."""

    if not cells:
        return []

    heights = sorted(c["y2"] - c["y1"] for c in cells)
    tolerance = max(4.0, 0.6 * heights[len(heights) // 2])

    rows = []

    for cell in sorted(cells, key=lambda c: c["yc"]):

        if rows:

            mean_y = sum(c["yc"] for c in rows[-1]) / len(rows[-1])

            if abs(cell["yc"] - mean_y) <= tolerance:
                rows[-1].append(cell)
                continue

        rows.append([cell])

    for row in rows:
        row.sort(key=lambda c: c["x1"])

    return rows


def row_text(row):

    return " ".join(c["text"] for c in row)


# ============================================================
# 2. SHOP NAME EXTRACTION
# ============================================================

SHOP_SKIP_RE = re.compile(
    r"\b(?:tax\s*invoice|invoice|bill|receipt|gstin|cash\s*memo|welcome|"
    r"customer\s*copy|original|duplicate|estimate|shop\s*no|phone|ph|mobile|"
    r"mob|cell|contact|file|date|gst\s*no|fssai|tin|dl\s*no|cashier|token|"
    r"table|state\s*name|email|website|www)\b",
    re.IGNORECASE
)

PINCODE_RE = re.compile(r"\b\d{6}\b")

# clear address words (kept short so food items like "Main Course" are safe)
ADDRESS_WORDS_RE = re.compile(
    r"\b(?:road|rd|street|nagar|salai|colony|avenue|opp|district|pincode|"
    r"tamil\s*nadu|door\s*no|plot\s*no|floor)\b",
    re.IGNORECASE
)


def is_shop_name_row(row):

    text = row_text(row)

    letters = letters_in(text)
    digits = len(re.findall(r"\d", text))

    if letters < 3 or digits > letters * 0.5:
        return False

    if SHOP_SKIP_RE.search(text) or ADDRESS_WORDS_RE.search(text):
        return False

    if PINCODE_RE.search(text) or re.search(r"\d{5,}", text):
        return False

    # a table header row (S.No | Particulars | Qty | Amount) is not a name
    header_cells = sum(
        1 for c in row if classify_header_cell(c["text"]) is not None
    )

    if header_cells >= 2:
        return False

    return True


def row_height(row):

    return statistics.median(c["y2"] - c["y1"] for c in row)


def extract_shop_name(rows):
    """The shop name is the first 'clean' line in the top rows, or a
    clearly bigger line there (shop names are printed large). Two short
    similar-sized lines are joined ("Madurai" + "Veg Restaurant")."""

    top = rows[:8]

    candidates = [i for i, row in enumerate(top) if is_shop_name_row(row)]

    if not candidates:
        return "Unknown"

    first = candidates[0]
    biggest = max(candidates, key=lambda i: (row_height(top[i]), -i))

    chosen = biggest if row_height(top[biggest]) >= 1.25 * row_height(top[first]) else first

    name = row_text(top[chosen])

    nxt = chosen + 1

    if (
        nxt in candidates
        and len(name.split()) <= 3
        and len(row_text(top[nxt]).split()) <= 4
        and abs(row_height(top[nxt]) - row_height(top[chosen])) <= 0.2 * row_height(top[chosen])
    ):
        name = name + " " + row_text(top[nxt])

    return name


# ============================================================
# 3. DATE EXTRACTION
# ============================================================

def extract_date(text):

    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",     # 25/09/2026
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",       # 2026/09/25
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b"    # 25 Sep 2026
    ]

    raw_date = None

    for pattern in date_patterns:

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            raw_date = match.group(0)
            break

    if not raw_date:
        return "Unknown"

    formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d",
        "%d %b %Y", "%d %B %Y", "%d/%m/%y", "%d-%m-%y"
    ]

    for fmt in formats:

        try:
            return datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")

        except ValueError:
            continue

    return "Unknown"


# ============================================================
# 4. NUMBER NORMALIZATION
# ============================================================

def normalize_number(number):

    number = number.strip()
    number = number.replace("₹", "").replace("Rs", "").replace("rs", "")
    number = number.strip()

    # 123,45 -> 123.45
    if re.match(r"^\d+,\d{2}$", number):
        number = number.replace(",", ".")

    else:
        # 1,250.50 -> 1250.50
        number = number.replace(",", "")

    return number


# ============================================================
# 5. TOTAL AMOUNT EXTRACTION
#
# Patterns are tried in priority order and the LAST match of the
# first pattern that hits is used. "Sub Total" is never mistaken
# for the final total.
# ============================================================

def extract_amount(text):

    money = r"\s*[:\-]?\s*(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d{1,2})?)"

    total_patterns = [
        r"Grand\s*Total" + money,
        r"Total\s*Amount" + money,
        r"Net\s*Amount" + money,
        r"Amount\s*Payable" + money,
        r"Balance\s*Due" + money,
        r"Bill\s*Amount" + money,
        r"(?<!sub )(?<!sub)(?<!sub-)Total" + money
    ]

    for pattern in total_patterns:

        found = re.findall(pattern, text, re.IGNORECASE)

        for value in reversed(found):

            try:
                return float(normalize_number(value))

            except ValueError:
                continue

    # Fallback: last decimal-looking number
    numbers = []

    for number in re.findall(r"\b\d+[.,]\d{2}\b", text):

        try:
            numbers.append(float(normalize_number(number)))

        except ValueError:
            continue

    if numbers:
        return numbers[-1]

    return None


# ============================================================
# 6. ITEM EXTRACTION
#
# Strategy (first one that finds items wins):
#
#   A) TABLE HEADER   find the header row containing
#                     Items / Products / Description of Goods /
#                     Particulars (+ S.No / Qty / Rate / Amount ...)
#                     and read ONLY that column below it
#                     (normally the 2nd column, right of S.No).
#   B) SERIAL ROWS    no readable header, but rows start with
#                     1, 2, 3 ... -> the text right of the number
#                     is the item.
#   C) PRICED LINES   plain receipts: "Chicken Biryani 2 360.00"
#                     (only lines that end with a price are items,
#                     so shop name / address are never items).
# ============================================================

ITEM_HEADER_WORDS = [
    "item", "items", "item name", "item names", "item description",
    "item details", "name of item", "name of items", "product",
    "products", "product name", "product description", "description",
    "description of goods", "description of goods and services",
    "description of services", "description of item", "particulars",
    "particular", "goods", "goods description"
]

SERIAL_HEADER_WORDS = [
    "sno", "s no", "sl no", "slno", "sr no", "srno", "si no", "sl",
    "sr", "no", "s n", "serial no", "serial number", "sl n"
]

OTHER_HEADER_WORDS = [
    "qty", "quantity", "qnty", "rate", "price", "mrp", "amount", "amt",
    "total", "value", "hsn", "hsn code", "hsn sac", "unit", "uom", "disc",
    "discount", "gst", "tax", "cgst", "sgst", "igst", "nos", "unit price",
    "net amount", "net", "size", "batch", "exp", "tax amount", "taxable"
]

# Lines/cells that mark the end of the item table (totals, tax, payment)
STOP_CELL_RE = re.compile(
    r"^\s*(?:sub\s*-?\s*total|grand\s*total|total|net\s*(?:amount|total)|"
    r"amount\s*payable|balance\s*due|bill\s*amount|round\s*(?:off|ing)|"
    r"discount|service\s*charge|[cs]?i?gst|tax|vat|thank\s*you|"
    r"amount\s*in\s*words|rupees|payment\s*modes?)\b",
    re.IGNORECASE
)

# Header/footer lines that are not purchased items
IGNORE_RE = re.compile(
    r"\b(?:address|gstin|fssai|invoice|cashier|customer|bill\s*no|"
    r"email|website|www)\b"
    r"|^\s*(?:phone|ph|mobile|mob|tel|telephone|contact|date|time)\b"
    r"\s*(?:no\.?)?\s*(?:[:\-#.]|\d)"
    r"|\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z0-9][A-Z0-9]\b"
    r"|\d{8,}",
    re.IGNORECASE
)

DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")

SERIAL_VALUE_RE = re.compile(r"^\d{1,3}[.)]?$")

UNIT_ONLY_RE = re.compile(
    r"^(?:nos?|pcs?|kgs?|gms?|grams?|ltrs?|pkts?|units?|each)\.?$",
    re.IGNORECASE
)

NUMERIC_TOKEN_RE = re.compile(
    r"^(?:₹|rs\.?|inr)?[\d,]+(?:[.,]\d{1,2})?$|^x\d+$|^\d+x$|^x$|^₹$|^rs\.?$",
    re.IGNORECASE
)


def is_price_token(token):
    """qty / price token. 6+ digit numbers (phone numbers, pincodes,
    GSTIN parts) are NOT prices."""

    if not NUMERIC_TOKEN_RE.match(token):
        return False

    digits = re.sub(r"\D", "", token)

    return not (re.fullmatch(r"[\d]+", token) and len(digits) >= 6)


def letters_in(text):

    return len(re.findall(r"[A-Za-z]", text))


def row_is_stop(row):

    if any(STOP_CELL_RE.match(c["text"]) for c in row):
        return True

    return bool(re.search(r"\bthank\s*you\b", row_text(row), re.IGNORECASE))


# ------------------------------------------------------------
# Item text cleaning + validation
# ------------------------------------------------------------

WEIGHT_TOKEN_RE = re.compile(
    r"^\d+(?:\.\d+)?\s*(?:kg|kgs|g|gm|gms|ml|l|ltr|ltrs|pcs|pc)$",
    re.IGNORECASE
)


def strip_numeric_noise(text):
    """Remove HSN / barcode numbers, GST percentages and trailing
    qty/rate numbers that OCR merged into the item cell, e.g.
    'Appalam Leaf 19059030 5% 2' -> 'Appalam Leaf'. Weights (250g,
    1Kg) are kept."""

    tokens = []

    for token in text.split():

        if WEIGHT_TOKEN_RE.match(token):
            tokens.append(token)
            continue

        if re.fullmatch(r"\d+(?:\.\d+)?%", token):
            continue

        if sum(ch.isdigit() for ch in token) >= 4:
            continue

        tokens.append(token)

    while tokens and re.fullmatch(r"[\d.,₹]+|rs\.?|x\d+", tokens[-1], re.IGNORECASE):
        tokens.pop()

    return " ".join(tokens)


def clean_item_text(text, strip_serial=False):

    cleaned = text.strip()

    # "1. Cashew" / "2) Rice"
    cleaned = re.sub(r"^\d{1,2}\s*[.)]\s*(?=[A-Za-z])", "", cleaned)

    # "1 Cashew" (only when a serial-number column is known to exist)
    if strip_serial:
        cleaned = re.sub(r"^\d{1,3}\s+(?=[A-Za-z])", "", cleaned)

    # trailing price like 320.00 / Rs 45.50 (repeat for "2 x 45.00")
    for _ in range(3):
        cleaned = re.sub(
            r"\s+(?:₹|Rs\.?)?\s*\d+[.,]\d{1,2}\s*$", "", cleaned,
            flags=re.IGNORECASE
        )

    # trailing "x2" / "x 2"
    cleaned = re.sub(r"\s+x\s*\d+\s*$", "", cleaned, flags=re.IGNORECASE)

    cleaned = strip_numeric_noise(cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip(" :-|.,")


# Footer / header / payment words: a line containing any of these is
# never a purchased item ("Payment Modes Cash Payable Cash Balance ...")
JUNK_ITEM_RE = re.compile(
    r"\b(?:payment|modes?|payable|balance|sales?|state\s*name|gstin|gst\s*no|"
    r"hsn|sac|cashier|bill|file|shop\s*no|total|paid|round|invoice|tender|"
    r"visit|thank|welcome|customer|signature|terms|conditions|goods\s*once|"
    r"subject\s*to|e\.?o\.?e|declaration|authori[sz]ed|bank|ifsc|a/c|cash|"
    r"upi|change|qty|quantity|rate|amount|amt|mrp|tax|cgst|sgst|igst|"
    r"discount|savings?|items?\s*count)\b",
    re.IGNORECASE
)


def is_valid_item(text):

    if len(text) < 2 or len(text) > 60 or len(text.split()) > 8:
        return False

    letters = letters_in(text)

    if letters < 2:
        return False

    # mostly symbols/digits = OCR garbage
    if letters / max(1, len(text.replace(" ", ""))) < 0.6:
        return False

    if (
        IGNORE_RE.search(text)
        or DATE_RE.search(text)
        or JUNK_ITEM_RE.search(text)
        or PINCODE_RE.search(text)
        or ADDRESS_WORDS_RE.search(text)
    ):
        return False

    if STOP_CELL_RE.match(text):
        return False

    return True


def add_item(items, text):

    if is_valid_item(text) and text.lower() not in [i.lower() for i in items]:
        items.append(text)


# ------------------------------------------------------------
# Header classification
# ------------------------------------------------------------

def normalize_header(text):

    text = re.sub(r"[^a-z ]", " ", text.lower())

    return re.sub(r"\s+", " ", text).strip()


def classify_header_cell(text):
    """Return 'item', 'serial', 'other' or None for a header cell."""

    raw = text.strip()

    if raw in {"#", "No", "No.", "S", "Sl", "Sr"}:
        return "serial"

    norm = normalize_header(raw)

    if not norm or len(norm.split()) > 5:
        return None

    # OCR often reads a capital I as a lowercase l: "ltem" -> "item"
    if norm.replace("l", "i") in ITEM_HEADER_WORDS:
        return "item"

    if norm in ITEM_HEADER_WORDS:
        return "item"

    if norm in SERIAL_HEADER_WORDS:
        return "serial"

    if norm in OTHER_HEADER_WORDS:
        return "other"

    # OCR typos: "ltem" (l instead of i), "Descripton", "Partculars" ...
    if len(norm) >= 4:

        if difflib.get_close_matches(norm, ITEM_HEADER_WORDS, n=1, cutoff=0.8):
            return "item"

        if difflib.get_close_matches(norm, OTHER_HEADER_WORDS, n=1, cutoff=0.85):
            return "other"

    return None


def find_header_row(rows):
    """First row that has an item-header cell AND at least one other
    header cell (S.No / Qty / Rate / Amount ...)."""

    for index, row in enumerate(rows):

        types = [classify_header_cell(c["text"]) for c in row]

        if "item" in types and any(t in ("serial", "other") for t in types):
            return index, row, types

    return None


# ------------------------------------------------------------
# A) Table header mode
# ------------------------------------------------------------

def extract_items_from_table(rows):

    header = find_header_row(rows)

    if header is None:
        return []

    header_index, cells, types = header

    item_pos = types.index("item")

    # Column boundaries come from the NEIGHBOURING recognised headers:
    #   left  = right edge of the S.No header (or other header on the left)
    #   right = left edge of the Qty / Rate / Amount header on the right
    left = 0.0

    for j in range(item_pos - 1, -1, -1):

        if types[j] in ("serial", "other"):
            left = cells[j]["x2"]
            break

    right = float("inf")

    for j in range(item_pos + 1, len(cells)):

        if types[j] in ("serial", "other"):
            right = cells[j]["x1"]
            break

    has_serial_header = "serial" in types[:item_pos]

    items = []
    last_was_item_row = False

    for row in rows[header_index + 1:]:

        if row_is_stop(row):
            break

        item_cells = [
            c for c in row
            if left <= c["xc"] < right
            and letters_in(c["text"]) >= 3
            and not UNIT_ONLY_RE.match(c["text"].strip())
        ]

        if not item_cells:
            last_was_item_row = False
            continue

        text = clean_item_text(
            " ".join(c["text"] for c in item_cells),
            strip_serial=has_serial_header
        )

        other_cells = [c for c in row if c not in item_cells]

        # Wrapped description: a row with item text but no S.No and no
        # numbers belongs to the previous item.
        if has_serial_header and not other_cells and items and last_was_item_row:

            items[-1] = clean_item_text(items[-1] + " " + text)
            continue

        before = len(items)

        add_item(items, text)

        last_was_item_row = len(items) > before

    return items


# ------------------------------------------------------------
# B) Serial-number rows mode (no readable header)
# ------------------------------------------------------------

def first_text_group(cells):
    """Join the first run of nearby text cells (an item name that OCR
    split into several boxes), stopping at numbers / units."""

    group = []

    for c in cells:

        is_text = (
            letters_in(c["text"]) >= 3
            and not UNIT_ONLY_RE.match(c["text"].strip())
        )

        if not is_text:

            if group:
                break

            continue

        if group:

            gap = c["x1"] - group[-1]["x2"]
            height = group[-1]["y2"] - group[-1]["y1"]

            if gap > 1.5 * height:
                break

        group.append(c)

    return " ".join(c["text"] for c in group)


def extract_items_from_serial_rows(rows):

    candidates = [
        i for i, row in enumerate(rows)
        if len(row) >= 2
        and SERIAL_VALUE_RE.match(row[0]["text"].strip())
        and any(letters_in(c["text"]) >= 3 for c in row[1:])
    ]

    if len(candidates) < 2:
        return []

    numbers = [
        int(re.sub(r"\D", "", rows[i][0]["text"])) for i in candidates
    ]

    # must look like a real serial column: 1,2,3...
    if numbers[0] > 1 or numbers[1] != numbers[0] + 1:
        return []

    items = []

    for i in range(candidates[0], len(rows)):

        row = rows[i]

        if row_is_stop(row):
            break

        if i in candidates:
            text = clean_item_text(first_text_group(row[1:]))
            add_item(items, text)

    return items


# ------------------------------------------------------------
# C) Priced-lines mode (plain receipts)
# ------------------------------------------------------------

def extract_items_from_lines(rows):

    items = []

    for row in rows:

        text = row_text(row)

        if row_is_stop(row):

            if items:
                break

            continue

        if IGNORE_RE.search(text) or DATE_RE.search(text):
            continue

        tokens = text.split()

        # drop a serial number that is its own cell: "1 | Chicken Biryani | 2 | 360.00"
        if (
            len(row) > 1
            and SERIAL_VALUE_RE.match(row[0]["text"].strip())
            and tokens
        ):
            tokens = tokens[1:]

        # peel qty / price off the end; a priced line must have at least one
        stripped = 0

        while tokens and is_price_token(tokens[-1]):
            tokens.pop()
            stripped += 1

        if stripped == 0 or not tokens:
            continue

        add_item(items, clean_item_text(" ".join(tokens)))

    return items


# ------------------------------------------------------------
# Entry point for item extraction
# ------------------------------------------------------------

def extract_items(rows):
    """Returns (items, method)."""

    for method, extractor in (
        ("table_header", extract_items_from_table),
        ("serial_rows", extract_items_from_serial_rows),
        ("priced_lines", extract_items_from_lines),
    ):

        items = extractor(rows)

        if items:
            return items, method

    return [], "none"


# ============================================================
# 7. TEXT CLEANING FOR THE MODEL
#
# clean_for_model() lives in text_cleaning.py and is shared with
# train_model.py, so training and prediction see identical text.
# ============================================================

# ============================================================
# 8. AI EXPENSE CATEGORY PREDICTION
# ============================================================

def predict_text(model_text):
    """Predict one cleaned text. Returns (category, confidence%) or
    (None, None) when none of its words exist in the training vocabulary
    (the model would otherwise just guess)."""

    vector = vectorizer.transform([model_text])

    if vector.getnnz() == 0:
        return None, None

    probabilities = model.predict_proba(vector)[0]
    best = int(probabilities.argmax())

    return (
        normalize_category(model.classes_[best]),
        round(float(probabilities[best]) * 100, 2)
    )


def predict_expense_category(items, shop_name):
    """Bill-level category from the ITEMS only.

    If no usable items were found, the shop name is used only when the
    model is confident about it. Raw OCR text is never used (it is full
    of addresses, GST lines and payment words). Otherwise the bill is
    marked 'Uncategorized' so the user can fix it instead of getting a
    confident-looking wrong category.

    Returns (category, confidence, text_source)."""

    if items:

        model_text = clean_for_model(" ".join(items))

        if model_text:

            category, confidence = predict_text(model_text)

            if category is not None:

                print(f"\nTEXT USED FOR AI PREDICTION (items): {model_text}")

                return category, confidence, "items"

    if shop_name and shop_name != "Unknown":

        category, confidence = predict_text(clean_for_model(shop_name))

        if category is not None and confidence >= REVIEW_THRESHOLD:

            print(f"\nTEXT USED FOR AI PREDICTION (shop name): {shop_name}")

            return category, confidence, "shop_name"

    print("\nNo reliable items -> Uncategorized (needs review)")

    return "Uncategorized", None, "none"


def predict_each_item(items):
    """Item-level categories (stored in expense_items.csv)."""

    results = []

    for item in items:

        category, confidence = predict_text(clean_for_model(item))

        results.append({
            "item": item,
            "category": category or "",
            "confidence": confidence if confidence is not None else ""
        })

    return results


# ============================================================
# 9. COMPLETE BILL INFORMATION EXTRACTION
# ============================================================

def extract_bill_info(cells):

    rows = group_into_rows(cells)

    # Line-ordered text (better than raw OCR order for dates/amounts)
    lines = [row_text(row) for row in rows]
    text = "\n".join(lines)

    shop_name = extract_shop_name(rows)
    date = extract_date(text)
    amount = extract_amount(text)

    items, method = extract_items(rows)

    category, confidence, text_source = predict_expense_category(
        items, shop_name
    )

    needs_review = (
        confidence is None
        or confidence < REVIEW_THRESHOLD
        or text_source != "items"
    )

    return {
        "shop_name": shop_name,
        "date": date,
        "amount": amount,
        "items": items,
        "item_predictions": predict_each_item(items),
        "category": category,
        "confidence": confidence,
        "needs_review": needs_review,
        "extraction_method": method,
        "raw_text": text,
        "_rows": rows
    }


# ============================================================
# 9b. DEBUG DUMP
# ============================================================

def save_debug(bill_id, data, rows):

    if not SAVE_DEBUG:
        return

    print("\nOCR ROWS (what the extractor sees):")

    for number, row in enumerate(rows):
        print(f"  {number:02d}: {row_text(row)}")

    try:

        os.makedirs(DEBUG_DIR, exist_ok=True)

        payload = {
            "bill_id": bill_id,
            "shop_name": data["shop_name"],
            "date": data["date"],
            "amount": data["amount"],
            "items": data["items"],
            "extraction_method": data["extraction_method"],
            "category": data["category"],
            "confidence": data["confidence"],
            "rows": [
                [
                    [c["text"], round(c["x1"]), round(c["y1"]),
                     round(c["x2"]), round(c["y2"])]
                    for c in row
                ]
                for row in rows
            ]
        }

        with open(
            os.path.join(DEBUG_DIR, f"{bill_id}.json"),
            "w", encoding="utf-8"
        ) as file:
            json.dump(payload, file, ensure_ascii=False, indent=1)

    except Exception as e:

        print("Debug dump failed:", e)


# ============================================================
# 10. SAVE EXPENSE  (CSV or MongoDB, see storage.py)
# ============================================================

def save_expense(data):

    try:

        store.save_bill(data)

        print("Expense saved to", store.name)

    except Exception as e:

        print("Saving failed:")
        print(e)


# ============================================================
# 12. HOME PAGE
# ============================================================

def render_upload(**extra):

    page = {
        "username": session.get("username"),
        "categories": CATEGORY_INFO,
        "saved": request.args.get("saved"),
        "error": None,
        "manual_tab": not ENABLE_OCR,          # no scan tab on small servers
        "form": {},
        "ocr_enabled": ENABLE_OCR,
    }

    page.update(extra)

    return render_template("upload.html", **page)


@app.route("/")
@login_required
def index():

    return render_upload()


# ============================================================
# 12b. MANUAL ENTRY (no bill photo)
# ============================================================

@app.route("/manual", methods=["POST"])
@login_required
def manual_entry():

    form = request.form

    user_id = session["user_id"]
    category = form.get("category", "").strip()
    shop = form.get("shop_name", "").strip()[:80]

    def fail(message):
        return render_upload(error=message, manual_tab=True, form=form), 400

    if category not in CATEGORIES:
        return fail("Please choose a category.")

    try:
        amount = round(float(form.get("amount", "").replace(",", "")), 2)
    except ValueError:
        return fail("Please enter the amount as a number.")

    if not 0 < amount <= 10_000_000:
        return fail("Amount must be greater than 0.")

    today = now_local().date()
    date_text = form.get("date", "").strip()

    if date_text:

        try:
            day = datetime.strptime(date_text, "%Y-%m-%d").date()
        except ValueError:
            return fail("That date is not valid.")

        if day > today:
            return fail("The date cannot be in the future.")

    else:
        day = today

    items = [
        part.strip()[:60]
        for part in re.split(r"[,\n]", form.get("items", ""))
        if part.strip()
    ][:20]

    save_expense({
        "bill_id": uuid.uuid4().hex[:12],
        "user_id": user_id,
        "image_name": "",
        "shop_name": shop or "Manual entry",
        "date": day.isoformat(),
        "amount": amount,
        "items": items,
        "item_predictions": [
            {"item": item, "category": category, "confidence": 100.0}
            for item in items
        ],
        "category": category,
        "confidence": 100.0,
        "needs_review": False,
        "extraction_method": "manual",
        "raw_text": "",
        "created_at": now_local().strftime("%Y-%m-%d %H:%M:%S"),
    })

    return redirect(url_for("index", saved=1))


# ============================================================
# 13. UPLOAD + PROCESS BILLS
# ============================================================

@app.route("/upload", methods=["POST"])
@login_required
def upload():

    if not ENABLE_OCR:
        return render_upload(
            error="Bill scanning is not available on this server yet. Please add the expense manually.",
            manual_tab=True,
        ), 503

    if "bill_images" not in request.files:
        return ("No file uploaded", 400)

    files = request.files.getlist("bill_images")

    extracted_data = []

    # Optional user ID (defaults until login is added)
    user_id = session["user_id"]

    os.makedirs("uploads", exist_ok=True)

    for file in files:

        if file.filename == "":
            continue

        safe_name = secure_filename(file.filename)

        extension = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""

        if extension not in ALLOWED_EXTENSIONS:
            print(f"Skipped (unsupported file type): {file.filename}")
            continue

        bill_id = uuid.uuid4().hex[:12]

        stored_name = f"{bill_id}_{safe_name}"

        file_path = os.path.join("uploads", stored_name)

        file.save(file_path)

        print(f"\nProcessing: {stored_name}")

        cells = read_cells(file_path)

        structured_data = extract_bill_info(cells)

        rows = structured_data.pop("_rows")

        structured_data["bill_id"] = bill_id
        structured_data["image_name"] = stored_name
        structured_data["user_id"] = user_id
        structured_data["created_at"] = now_local().strftime("%Y-%m-%d %H:%M:%S")

        print("\n================================")
        print("EXTRACTED BILL INFORMATION")
        print("================================")
        print("User ID:", structured_data["user_id"])
        print("Shop Name:", structured_data["shop_name"])
        print("Date:", structured_data["date"])
        print("Amount:", structured_data["amount"])
        print("Items:", structured_data["items"])
        print("Extraction method:", structured_data["extraction_method"])
        print("Category:", structured_data["category"])
        print("Confidence:", structured_data["confidence"])
        print("Needs review:", structured_data["needs_review"])

        save_debug(bill_id, structured_data, rows)

        save_expense(structured_data)

        extracted_data.append(structured_data)

    if not extracted_data:
        return ("No valid files processed", 400)

    return render_template("result.html", bills=extracted_data)


# ============================================================
# 14. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    # use_reloader=False: otherwise debug mode starts the script twice
    # and loads EasyOCR twice.
    app.run(debug=True, use_reloader=False)
