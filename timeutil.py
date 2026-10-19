"""Local time helper. Cloud servers run in UTC, which would give a wrong
'today' for users in India. Change APP_TZ (environment variable) if needed."""

import os
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(os.environ.get("APP_TZ", "Asia/Kolkata"))
except Exception:          # e.g. Windows without the 'tzdata' package
    TZ = None


def now_local():

    return datetime.now(TZ) if TZ else datetime.now()
