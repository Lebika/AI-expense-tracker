"""Shared text cleaning. Used by train_model.py AND app.py so the model
sees text in exactly the same form during training and prediction."""

import re

# bill / payment words that say nothing about what was bought
MODEL_NOISE = set("""
total bill paid to gst invoice txn upi pos visit thank you again ref incl rs
payment order tax cash memo receipt inv no kg g l ltr ml and of the for at on
online store autopay recurring pnr sgst cgst amt amount date time customer
copy web
""".split())

# City / state words. In the training data a place name such as "madurai"
# only ever appears with one label (Travel), so a bill from a Madurai
# restaurant was being pushed towards Travel.
PLACE_WORDS = set("""
madurai chennai coimbatore trichy tiruchirappalli salem tirunelveli erode
vellore tirupur thanjavur tamil nadu tamilnadu kerala karnataka bangalore
bengaluru hyderabad mumbai delhi kolkata pune kochi cochin india
""".split())


def clean_for_model(text):

    tokens = []

    for token in re.findall(r"[a-z&]+", str(text).lower()):

        if token in MODEL_NOISE or token in PLACE_WORDS or len(token) < 2:
            continue

        tokens.append(token)

    return " ".join(tokens)
