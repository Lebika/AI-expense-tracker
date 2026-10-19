"""Shows which Gemini models YOUR API key can use.

Run:  python list_models.py
Then put one of the names in .env, for example:  GEMINI_MODEL=gemini-2.5-flash
"""

import json
import os
import sys
import urllib.error
import urllib.request


def read_env(path=".env"):

    values = {}

    if os.path.exists(path):

        with open(path, encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip().strip('"').strip("'")

    return values


key = os.environ.get("GEMINI_API_KEY") or read_env().get("GEMINI_API_KEY")

if not key:
    sys.exit("No GEMINI_API_KEY found. Put it in the .env file first.")

request = urllib.request.Request(
    "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
    headers={"x-goog-api-key": key},
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        models = json.loads(response.read().decode("utf-8")).get("models", [])
except urllib.error.HTTPError as error:
    sys.exit(f"Google returned error {error.code}. Check that the key is correct.")
except urllib.error.URLError:
    sys.exit("Could not reach Google. Check your internet connection.")

names = [
    m["name"].replace("models/", "") for m in models
    if "generateContent" in m.get("supportedGenerationMethods", [])
]

skip = ("image", "tts", "audio", "live", "embedding", "robotics", "vision", "imagen", "veo", "lyria")
text_models = [n for n in names if not any(word in n for word in skip)]

print("Models your key can use for text:\n")

for name in sorted(text_models):
    print("  ", name)

print("\nAdd ONE of these lines to .env (a 'flash' or 'flash-lite' model is best):")
print("  GEMINI_MODEL=<name from the list>")
