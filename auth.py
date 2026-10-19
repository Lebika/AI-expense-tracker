"""Sign-up, login, logout and the personal profile page.

Storage (plain CSV for now, moved to MongoDB later):
  users.csv     user_id, username, password_hash, created_at
  profiles.csv  user_id, updated_at, profile_json
Passwords are stored only as salted hashes, never as text.
"""

import csv
import io
import json
import os
import re
import secrets
import threading
import time
import uuid
from functools import wraps

from flask import (
    Blueprint, Response, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

from categories import CATEGORY_NAMES
from storage import UserExists, store
from timeutil import now_local

MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 300

HOUSEHOLD = ["Just me", "Couple", "Family with kids", "Joint family"]
WORK_TYPE = ["Student", "Salaried", "Business / self-employed", "Homemaker", "Retired", "Other"]
STYLES = [
    ("Strict saver", "I cut anything that is not essential"),
    ("Balanced", "I save regularly but allow some treats"),
    ("Enjoy now", "I prefer spending on experiences and save what is left"),
]

auth_bp = Blueprint("auth", __name__)

_failed = {}                      # username -> [count, first_failure_time]
_DUMMY_HASH = generate_password_hash("not-a-real-password")


# ------------------------------------------------------------
# Data access (CSV or MongoDB, chosen in storage.py)
# ------------------------------------------------------------

def find_user(username):

    return store.get_user(username)


def load_profile(user_id):

    return store.get_profile(user_id)


def save_profile(user_id, profile):

    store.save_profile(user_id, profile)


def has_legacy_rows():

    return store.legacy_exists()


def claim_legacy_rows(new_user_id):

    store.claim_legacy(new_user_id)


# ------------------------------------------------------------
# Login protection
# ------------------------------------------------------------

def login_required(view):

    @wraps(view)
    def wrapped(*args, **kwargs):

        if not session.get("user_id"):

            target = None

            if request.method == "GET":
                target = request.full_path.rstrip("?")

            return redirect(url_for("auth.login", next=target))

        return view(*args, **kwargs)

    return wrapped


def safe_next(target):
    """Only allow redirects to pages inside this site."""

    if target and target.startswith("/") and not target.startswith("//"):
        return target

    return url_for("index")


def start_session(user):

    session.clear()
    session["user_id"] = user["user_id"]
    session["username"] = user["username"]
    session["csrf_token"] = secrets.token_hex(16)
    session.permanent = False


def is_locked(username):

    record = _failed.get(username.lower())

    if not record:
        return False

    if time.time() - record[1] >= LOCKOUT_SECONDS:
        del _failed[username.lower()]
        return False

    return record[0] >= MAX_FAILED_LOGINS


def note_failure(username):

    record = _failed.setdefault(username.lower(), [0, time.time()])
    record[0] += 1


_hits = {}                         # (name, ip) -> list of timestamps


def rate_limited(name, limit, window_seconds):
    """True if this IP already did `name` `limit` times in the window."""

    key = (name, request.remote_addr or "?")
    now = time.time()

    recent = [t for t in _hits.get(key, []) if now - t < window_seconds]

    if len(recent) >= limit:
        _hits[key] = recent
        return True

    recent.append(now)
    _hits[key] = recent

    return False


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@auth_bp.route("/register", methods=["GET", "POST"])
def register():

    first_user = store.user_count() == 0
    offer_import = first_user and has_legacy_rows()

    if request.method == "GET":

        return render_template(
            "register.html", error=None, username="", offer_import=offer_import
        )

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm", "")

    def fail(message):
        return render_template(
            "register.html", error=message, username=username,
            offer_import=offer_import
        ), 400

    if rate_limited("register", 8, 3600):
        return fail("Too many sign-ups from this network. Please try again later.")

    if not request.form.get("agree_terms"):
        return fail("Please tick the box to accept the Privacy Policy and Terms.")

    if not re.fullmatch(r"[A-Za-z0-9_.@-]{3,40}", username):
        return fail("Username must be 3-40 characters: letters, numbers and _ . @ -")

    if len(password) < 8 or len(password) > 128:
        return fail("Password must be at least 8 characters.")

    if password != confirm:
        return fail("The two passwords do not match.")

    user = {
        "user_id": uuid.uuid4().hex,
        "username": username,
        "password_hash": generate_password_hash(password),
        "created_at": now_local().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        store.add_user(user)
    except UserExists:
        return fail("That username is already taken.")

    if offer_import and request.form.get("import_old"):
        claim_legacy_rows(user["user_id"])

    start_session(user)

    return redirect(url_for("auth.profile", welcome=1))


@auth_bp.route("/login", methods=["GET", "POST"])
def login():

    next_page = request.values.get("next", "")

    if request.method == "GET":
        return render_template("login.html", error=None, next=next_page, username="")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    def fail(message):
        return render_template(
            "login.html", error=message, next=next_page, username=username
        ), 401

    if is_locked(username) or rate_limited("login", 40, 600):
        return fail("Too many failed attempts. Please wait 5 minutes and try again.")

    user = find_user(username)

    # always run one hash check so timing does not reveal valid usernames
    valid = check_password_hash(user["password_hash"] if user else _DUMMY_HASH, password)

    if not user or not valid:
        note_failure(username)
        return fail("Wrong username or password.")

    _failed.pop(username.lower(), None)

    start_session(user)

    return redirect(safe_next(next_page))


@auth_bp.route("/logout", methods=["POST"])
def logout():

    session.clear()

    return redirect(url_for("auth.login"))


def parse_money(text, label, errors):

    text = (text or "").replace(",", "").replace("₹", "").strip()

    if not text:
        return ""

    try:
        value = float(text)
    except ValueError:
        errors.append(f"{label} must be a number.")
        return ""

    if not 0 <= value <= 100_000_000:
        errors.append(f"{label} is out of range.")
        return ""

    return round(value, 2)


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():

    user_id = session["user_id"]
    saved = False
    errors = []
    data = load_profile(user_id)

    if request.method == "POST":

        form = request.form

        data = {
            "display_name": form.get("display_name", "").strip()[:60],
            "work_type": form.get("work_type", ""),
            "household": form.get("household", ""),
            "monthly_income": parse_money(form.get("monthly_income"), "Monthly income", errors),
            "monthly_budget": parse_money(form.get("monthly_budget"), "Monthly budget", errors),
            "savings_goal": parse_money(form.get("savings_goal"), "Monthly savings goal", errors),
            "spending_style": form.get("spending_style", ""),
            "cut_categories": [c for c in form.getlist("cut_categories") if c in CATEGORY_NAMES],
            "protect_categories": [c for c in form.getlist("protect_categories") if c in CATEGORY_NAMES],
            "goal_text": form.get("goal_text", "").strip()[:500],
            "habits_text": form.get("habits_text", "").strip()[:800],
        }

        if data["work_type"] not in WORK_TYPE:
            data["work_type"] = ""

        if data["household"] not in HOUSEHOLD:
            data["household"] = ""

        if data["spending_style"] not in [s for s, _ in STYLES]:
            data["spending_style"] = ""

        both = set(data["cut_categories"]) & set(data["protect_categories"])

        if both:
            errors.append(
                "A category cannot be in both lists: " + ", ".join(sorted(both))
            )

        if not errors:
            save_profile(user_id, data)
            saved = True

    return render_template(
        "profile.html",
        p=data,
        saved=saved,
        errors=errors,
        welcome=request.args.get("welcome"),
        delete_error=request.args.get("delete_error"),
        categories=CATEGORY_NAMES,
        household=HOUSEHOLD,
        work_type=WORK_TYPE,
        styles=STYLES,
    ), (400 if errors else 200)


# ------------------------------------------------------------
# Your data: download and delete
# ------------------------------------------------------------

@auth_bp.route("/export.csv")
@login_required
def export_csv():
    """All of the signed-in user's expenses as a spreadsheet file."""

    df = store.bills_df(session["user_id"])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", "shop", "category", "amount", "items", "added_by"])

    if not df.empty:

        for _, row in df.sort_values("date", ascending=False).iterrows():

            items = row.get("items")

            if isinstance(items, str):

                try:
                    items = json.loads(items)
                except ValueError:
                    items = [items]

            writer.writerow([
                row.get("date", ""),
                row.get("shop_name", ""),
                row.get("category", ""),
                row.get("amount", ""),
                ", ".join(items) if isinstance(items, list) else "",
                "typed by me" if row.get("extraction_method") == "manual" else "bill scan",
            ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=my-expenses.csv"},
    )


@auth_bp.route("/account/delete", methods=["POST"])
@login_required
def delete_account():
    """Deletes the account and ALL its data after a password check."""

    user = find_user(session["username"])

    if (
        not request.form.get("confirm_delete")
        or not user
        or not check_password_hash(user["password_hash"], request.form.get("password", ""))
    ):
        return redirect(url_for("auth.profile", delete_error=1))

    store.delete_user(session["user_id"])
    session.clear()

    return render_template("deleted.html")
