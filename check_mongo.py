"""Tests the connection to MongoDB Atlas.   Run:  python check_mongo.py"""

import sys

from settings import setting

uri = setting("MONGODB_URI")

if not uri:
    sys.exit("MONGODB_URI is empty. Put your Atlas connection string in the .env file.")

try:
    from pymongo import MongoClient
except ImportError:
    sys.exit('pymongo is not installed. Run:  pip install "pymongo[srv]" certifi')

options = {"serverSelectionTimeoutMS": 8000}

if uri.startswith("mongodb+srv://"):
    import certifi
    options["tlsCAFile"] = certifi.where()

try:
    client = MongoClient(uri, **options)
    client.admin.command("ping")
except Exception as error:
    sys.exit(
        "FAILED to connect.\n"
        "Check: (1) the password in MONGODB_URI (no < > around it), "
        "(2) Atlas > Network Access allows your IP, (3) your internet.\n"
        f"Details: {error}"
    )

db = client[setting("MONGODB_DB", "expense_tracker")]

print("✅ Connected to MongoDB Atlas.")
print("Database:", db.name)

names = db.list_collection_names()

if not names:
    print("No collections yet (they are created automatically on first save).")

for name in sorted(names):
    print(f"  {name}: {db[name].count_documents({})} documents")
