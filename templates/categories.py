"""Single source of truth for expense categories.

Used by app.py (manual dropdown + scanned bills), dashboard.py and
train_model.py, so every screen and the model use the same names.
To add a category: add it to CATEGORY_INFO, add ~25 example items for it
to extra_items.csv, then run  python train_model.py  again.
"""

import re

# (name, hint shown in the dropdown) - ordered from most to least common
CATEGORY_INFO = [
    ("Food & Dining", "restaurants, tea, tiffin, snacks"),
    ("Groceries", "provisions, vegetables, milk, household items"),
    ("Transport & Fuel", "petrol, auto, bus, metro, vehicle service"),
    ("Shopping", "clothes, electronics, home goods"),
    ("Health & Medical", "medicines, doctor, tests, hospital, gym"),
    ("Bills & Utilities", "electricity, water, gas, mobile, internet"),
    ("Rent & Housing", "rent, maintenance, repairs, maid"),
    ("Education", "fees, courses, books, coaching"),
    ("Entertainment", "movies, games, outings"),
    ("Travel", "flights, trains, hotels, trips"),
    ("Personal Care", "salon, spa, cosmetics"),
    ("Subscriptions", "Netflix, Spotify, app plans"),
    ("Insurance", "life, health, vehicle premiums"),
    ("Gifts & Donations", "gifts, donations, function moi"),
    ("Savings & Investments", "SIP, FD, gold, chit"),
    ("Other", "anything else"),
]

CATEGORY_NAMES = [name for name, _ in CATEGORY_INFO]

# Old names (earlier CSV rows / training data) -> current names
LEGACY_MAP = {
    "Health & Fitness": "Health & Medical",
    "Transportation": "Transport & Fuel",
    "Miscellaneous": "Other",
    "Kids & Family": "Other",
    "Pets": "Other",
}


def normalize_category(name):

    name = str(name).strip()

    return LEGACY_MAP.get(name, name)


# Training data only: rows that were labelled "Other" but clearly belong
# to one of the newer categories are moved there.
RELABEL_RULES = [
    (re.compile(r"\binsurance\b"), "Insurance"),
    (re.compile(r"\b(?:tuition|school|college|exam|coaching|course|university|academy)\b"), "Education"),
    (re.compile(r"\b(?:salon|spa|barber|beauty|parlour|parlor|facial|haircut|massage)\b"), "Personal Care"),
    (re.compile(r"\b(?:donation|charity|gift|temple|church)\b"), "Gifts & Donations"),
    (re.compile(r"\b(?:rent|maintenance|plumber|electrician|carpenter)\b"), "Rent & Housing"),
]


def relabel_other(items_text):

    for pattern, category in RELABEL_RULES:

        if pattern.search(str(items_text).lower()):
            return category

    return "Other"
