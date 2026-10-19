"""Reads settings from environment variables or the .env file."""

import os


def load_env_file(path=".env"):

    values = {}

    if os.path.exists(path):

        with open(path, encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def setting(name, default=""):

    return os.environ.get(name) or load_env_file().get(name) or default
