"""Copies your existing CSV data into MongoDB Atlas.

Safe to run more than once: rows are matched by their ids, so nothing is
duplicated.   Run:  python migrate_csv_to_mongo.py
Your CSV files are only read, never changed or deleted.
"""

import csv
import hashlib
import json
import os
import sys

from settings import setting
from storage import (
    EXPENSES_FILE, ITEMS_FILE, PROFILES_FILE, RECS_FILE, USERS_FILE,
    MongoStore, month_of,
)


def read(path):

    if not os.path.exists(path):
        return []

    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def number(text):

    try:
        return float(text)
    except (TypeError, ValueError):
        return None


try:
    mongo = MongoStore(setting("MONGODB_URI"), setting("MONGODB_DB", "expense_tracker"))
except RuntimeError as error:
    sys.exit(str(error))

db = mongo.db
counts = {"bills": 0, "bill_items": 0, "users": 0, "profiles": 0, "recommendations": 0}

# ---- bills
for row in read(EXPENSES_FILE):

    bill_id = row.get("bill_id") or hashlib.sha1(
        f"{row.get('user_id')}{row.get('created_at')}{row.get('amount')}".encode()
    ).hexdigest()[:12]

    try:
        items = json.loads(row.get("items") or "[]")
    except ValueError:
        items = []

    document = {
        "bill_id": bill_id,
        "user_id": row.get("user_id", ""),
        "image_name": row.get("image_name", ""),
        "shop_name": row.get("shop_name", ""),
        "date": row.get("date", ""),
        "month": row.get("month") or month_of(row.get("date")),
        "amount": number(row.get("amount")) or 0.0,
        "items": items,
        "item_count": len(items),
        "category": row.get("category", ""),
        "confidence": number(row.get("confidence")),
        "needs_review": str(row.get("needs_review", "")).lower() == "true",
        "extraction_method": row.get("extraction_method", ""),
        "raw_text": row.get("raw_text", ""),
        "created_at": row.get("created_at", ""),
    }

    db.bills.update_one({"bill_id": bill_id}, {"$set": document}, upsert=True)
    counts["bills"] += 1

# ---- items
for row in read(ITEMS_FILE):

    if not row.get("bill_id"):
        continue

    document = {
        "bill_id": row["bill_id"],
        "user_id": row.get("user_id", ""),
        "date": row.get("date", ""),
        "month": row.get("month") or month_of(row.get("date")),
        "shop_name": row.get("shop_name", ""),
        "item": row.get("item", ""),
        "item_category": row.get("item_category", ""),
        "item_confidence": number(row.get("item_confidence")),
        "created_at": row.get("created_at", ""),
    }

    key = {"bill_id": document["bill_id"], "item": document["item"], "created_at": document["created_at"]}

    db.bill_items.update_one(key, {"$set": document}, upsert=True)
    counts["bill_items"] += 1

# ---- users
for row in read(USERS_FILE):

    document = dict(row)
    document["username_lower"] = row["username"].lower()

    db.users.update_one({"user_id": row["user_id"]}, {"$set": document}, upsert=True)
    counts["users"] += 1

# ---- profiles
for row in read(PROFILES_FILE):

    try:
        profile = json.loads(row["profile_json"])
    except ValueError:
        continue

    db.profiles.replace_one(
        {"user_id": row["user_id"]},
        {"user_id": row["user_id"], "updated_at": row["updated_at"], "profile": profile},
        upsert=True,
    )
    counts["profiles"] += 1

# ---- saved tips
for row in read(RECS_FILE):

    db.recommendations.update_one(
        {"user_id": row["user_id"], "created_at": row["created_at"]},
        {"$set": row}, upsert=True,
    )
    counts["recommendations"] += 1

print("✅ Copied to MongoDB Atlas:")

for name, count in counts.items():
    print(f"  {name}: {count}")

print("\nNow set STORAGE=mongo in .env and restart the app.")
