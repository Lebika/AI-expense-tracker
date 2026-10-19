"""Where the app keeps its data.

STORAGE=csv    (default)  plain CSV files next to app.py
STORAGE=mongo             MongoDB Atlas (needs MONGODB_URI in .env)

The rest of the app only talks to `store`, so switching is one line in .env.
"""

import csv
import json
import os
import shutil
import threading
import uuid
from datetime import datetime

import pandas as pd

from settings import setting
from timeutil import now_local

LEGACY_USER = "default_user"          # data saved before sign-in existed

EXPENSES_FILE = "expenses.csv"
ITEMS_FILE = "expense_items.csv"
USERS_FILE = "users.csv"
PROFILES_FILE = "profiles.csv"
RECS_FILE = "recommendations.csv"

BILL_COLUMNS = [
    "bill_id", "user_id", "image_name", "shop_name", "date", "month",
    "amount", "items", "item_count", "category", "confidence",
    "needs_review", "extraction_method", "raw_text", "created_at",
]

ITEM_COLUMNS = [
    "bill_id", "user_id", "date", "month", "shop_name", "item",
    "item_category", "item_confidence", "created_at",
]

USER_COLUMNS = ["user_id", "username", "password_hash", "created_at"]
PROFILE_COLUMNS = ["user_id", "updated_at", "profile_json"]
REC_COLUMNS = ["user_id", "created_at", "data_hash", "provider", "text"]


class UserExists(Exception):
    pass


def month_of(date_text):

    import re

    if date_text and re.match(r"^\d{4}-\d{2}-\d{2}$", str(date_text)):
        return str(date_text)[:7]

    return ""


def bill_document(data):
    """The bill row/document, built from the dict app.py produces."""

    return {
        "bill_id": data["bill_id"],
        "user_id": data["user_id"],
        "image_name": data["image_name"],
        "shop_name": data["shop_name"],
        "date": data["date"],
        "month": month_of(data["date"]),
        "amount": data["amount"],
        "item_count": len(data["items"]),
        "category": data["category"],
        "confidence": data["confidence"],
        "needs_review": bool(data["needs_review"]),
        "extraction_method": data["extraction_method"],
        "raw_text": data["raw_text"],
        "created_at": data["created_at"],
    }


def item_documents(data):

    return [
        {
            "bill_id": data["bill_id"],
            "user_id": data["user_id"],
            "date": data["date"],
            "month": month_of(data["date"]),
            "shop_name": data["shop_name"],
            "item": entry["item"],
            "item_category": entry["category"],
            "item_confidence": entry["confidence"],
            "created_at": data["created_at"],
        }
        for entry in data["item_predictions"]
    ]


# ============================================================
# CSV backend
# ============================================================

def ensure_csv(path, columns):
    """Create the CSV if missing. If an older file has different columns,
    back it up and migrate it so new rows always line up with the header."""

    if not os.path.exists(path):

        with open(path, mode="w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=columns).writeheader()

        return

    with open(path, mode="r", newline="", encoding="utf-8") as file:
        dict_reader = csv.DictReader(file)
        old_columns = dict_reader.fieldnames or []
        rows = list(dict_reader)

    if old_columns == columns:
        return

    backup = "{}_backup_{}.csv".format(
        os.path.splitext(path)[0], datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    shutil.copy(path, backup)

    print(f"CSV columns changed -> migrating {path} (backup: {backup})")

    for row in rows:

        if "bill_id" in columns and not row.get("bill_id"):
            row["bill_id"] = uuid.uuid4().hex[:12]

        if "month" in columns and not row.get("month"):
            row["month"] = month_of(row.get("date"))

    with open(path, mode="w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file, fieldnames=columns, extrasaction="ignore", restval=""
        )

        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path):

    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_rows(path, columns, rows):

    with open(path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_row(path, columns, row):

    with open(path, "a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=columns).writerow(row)


class CsvStore:

    name = "CSV files"

    def __init__(self):

        self.lock = threading.Lock()

        for path, columns in (
            (EXPENSES_FILE, BILL_COLUMNS), (ITEMS_FILE, ITEM_COLUMNS),
            (USERS_FILE, USER_COLUMNS), (PROFILES_FILE, PROFILE_COLUMNS),
            (RECS_FILE, REC_COLUMNS),
        ):
            ensure_csv(path, columns)

    # ---- bills
    def save_bill(self, data):

        bill = bill_document(data)
        bill["items"] = json.dumps(data["items"], ensure_ascii=False)

        with self.lock:

            _append_row(EXPENSES_FILE, BILL_COLUMNS, bill)

            for row in item_documents(data):
                _append_row(ITEMS_FILE, ITEM_COLUMNS, row)

    def bills_df(self, user_id):

        df = pd.read_csv(EXPENSES_FILE)

        if df.empty:
            return df

        return df[df["user_id"].astype(str) == str(user_id)].copy()

    def items_df(self, bill_ids):

        df = pd.read_csv(ITEMS_FILE)

        if df.empty:
            return df

        return df[df["bill_id"].isin(bill_ids)]

    # ---- data saved before sign-in existed
    def legacy_exists(self):

        return any(r.get("user_id") == LEGACY_USER for r in _read_rows(EXPENSES_FILE))

    def claim_legacy(self, new_user_id):

        with self.lock:

            for path, columns in ((EXPENSES_FILE, BILL_COLUMNS), (ITEMS_FILE, ITEM_COLUMNS)):

                rows = _read_rows(path)

                for row in rows:

                    if row.get("user_id") == LEGACY_USER:
                        row["user_id"] = new_user_id

                _write_rows(path, columns, rows)

    # ---- users
    def user_count(self):

        return len(_read_rows(USERS_FILE))

    def get_user(self, username):

        wanted = username.strip().lower()

        for row in _read_rows(USERS_FILE):

            if row["username"].lower() == wanted:
                return row

        return None

    def add_user(self, user):

        with self.lock:

            if self.get_user(user["username"]):
                raise UserExists()

            _append_row(USERS_FILE, USER_COLUMNS, user)

    # ---- profiles
    def get_profile(self, user_id):

        for row in _read_rows(PROFILES_FILE):

            if row["user_id"] == user_id:

                try:
                    return json.loads(row["profile_json"])
                except ValueError:
                    return {}

        return {}

    def save_profile(self, user_id, profile):

        with self.lock:

            rows = [r for r in _read_rows(PROFILES_FILE) if r["user_id"] != user_id]

            rows.append({
                "user_id": user_id,
                "updated_at": now_local().strftime("%Y-%m-%d %H:%M:%S"),
                "profile_json": json.dumps(profile, ensure_ascii=False),
            })

            _write_rows(PROFILES_FILE, PROFILE_COLUMNS, rows)

    # ---- delete everything about one user
    def delete_user(self, user_id):

        with self.lock:

            for path, columns in (
                (EXPENSES_FILE, BILL_COLUMNS), (ITEMS_FILE, ITEM_COLUMNS),
                (USERS_FILE, USER_COLUMNS), (PROFILES_FILE, PROFILE_COLUMNS),
                (RECS_FILE, REC_COLUMNS),
            ):
                rows = [r for r in _read_rows(path) if r.get("user_id") != user_id]
                _write_rows(path, columns, rows)

    # ---- saved AI tips
    def latest_rec(self, user_id):

        mine = [r for r in _read_rows(RECS_FILE) if r["user_id"] == user_id]

        return mine[-1] if mine else None

    def calls_today(self, user_id):

        today = now_local().strftime("%Y-%m-%d")

        return sum(
            1 for r in _read_rows(RECS_FILE)
            if r["user_id"] == user_id and r["created_at"].startswith(today)
        )

    def add_rec(self, row):

        with self.lock:
            _append_row(RECS_FILE, REC_COLUMNS, row)


# ============================================================
# MongoDB Atlas backend
# ============================================================

class MongoStore:

    name = "MongoDB Atlas"

    def __init__(self, uri, db_name):

        if not uri:
            raise RuntimeError(
                "STORAGE=mongo but MONGODB_URI is empty. Put your Atlas "
                "connection string in the .env file."
            )

        from pymongo import MongoClient

        options = {"serverSelectionTimeoutMS": 8000}

        if uri.startswith("mongodb+srv://"):
            import certifi
            options["tlsCAFile"] = certifi.where()

        self.client = MongoClient(uri, **options)

        try:
            self.client.admin.command("ping")
        except Exception as error:
            raise RuntimeError(
                "Could not connect to MongoDB Atlas. Check: (1) the password in "
                "MONGODB_URI, (2) Network Access allows your IP address, "
                f"(3) your internet. Details: {error}"
            )

        self.db = self.client[db_name]

        db = self.db
        db.bills.create_index("bill_id", unique=True)
        db.bills.create_index([("user_id", 1), ("date", -1)])
        db.bill_items.create_index([("user_id", 1), ("bill_id", 1)])
        db.users.create_index("username_lower", unique=True)
        db.users.create_index("user_id", unique=True)
        db.profiles.create_index("user_id", unique=True)
        db.recommendations.create_index([("user_id", 1), ("created_at", -1)])

    # ---- bills
    def save_bill(self, data):

        bill = bill_document(data)
        bill["items"] = list(data["items"])

        self.db.bills.insert_one(bill)

        docs = item_documents(data)

        if docs:
            self.db.bill_items.insert_many(docs)

    def bills_df(self, user_id):

        docs = list(self.db.bills.find({"user_id": user_id}, {"_id": 0}))

        return pd.DataFrame(docs) if docs else pd.DataFrame()

    def items_df(self, bill_ids):

        docs = list(
            self.db.bill_items.find({"bill_id": {"$in": list(bill_ids)}}, {"_id": 0})
        )

        return pd.DataFrame(docs) if docs else pd.DataFrame()

    # ---- data saved before sign-in existed
    def legacy_exists(self):

        return self.db.bills.count_documents({"user_id": LEGACY_USER}) > 0

    def claim_legacy(self, new_user_id):

        for collection in (self.db.bills, self.db.bill_items):
            collection.update_many(
                {"user_id": LEGACY_USER}, {"$set": {"user_id": new_user_id}}
            )

    # ---- users
    def user_count(self):

        return self.db.users.count_documents({})

    def get_user(self, username):

        return self.db.users.find_one(
            {"username_lower": username.strip().lower()}, {"_id": 0}
        )

    def add_user(self, user):

        from pymongo.errors import DuplicateKeyError

        document = dict(user)
        document["username_lower"] = user["username"].lower()

        try:
            self.db.users.insert_one(document)
        except DuplicateKeyError:
            raise UserExists()

    # ---- profiles
    def get_profile(self, user_id):

        doc = self.db.profiles.find_one({"user_id": user_id}, {"_id": 0})

        return (doc or {}).get("profile", {})

    def save_profile(self, user_id, profile):

        self.db.profiles.replace_one(
            {"user_id": user_id},
            {
                "user_id": user_id,
                "updated_at": now_local().strftime("%Y-%m-%d %H:%M:%S"),
                "profile": profile,
            },
            upsert=True,
        )

    # ---- delete everything about one user
    def delete_user(self, user_id):

        for collection in (
            self.db.bills, self.db.bill_items, self.db.profiles,
            self.db.recommendations, self.db.users,
        ):
            collection.delete_many({"user_id": user_id})

    # ---- saved AI tips
    def latest_rec(self, user_id):

        docs = list(
            self.db.recommendations.find({"user_id": user_id}, {"_id": 0})
            .sort("created_at", -1).limit(1)
        )

        return docs[0] if docs else None

    def calls_today(self, user_id):

        today = now_local().strftime("%Y-%m-%d")

        return self.db.recommendations.count_documents(
            {"user_id": user_id, "created_at": {"$regex": "^" + today}}
        )

    def add_rec(self, row):

        self.db.recommendations.insert_one(dict(row))


# ============================================================
# Choose the backend
# ============================================================

def create_store():

    mode = setting("STORAGE", "csv").lower()

    if mode == "mongo":
        return MongoStore(setting("MONGODB_URI"), setting("MONGODB_DB", "expense_tracker"))

    return CsvStore()


store = create_store()

print(f"Data is stored in: {store.name}")
