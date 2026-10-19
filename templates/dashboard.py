"""Spending dashboard built from expenses.csv and expense_items.csv.

Views: week / month / last 30 days / year / all time / custom dates,
grouped by day, week, month or year.
"""

import calendar
import os
from datetime import date, timedelta

import pandas as pd
from flask import Blueprint, render_template, request

from timeutil import now_local
from categories import normalize_category

CSV_FILE = "expenses.csv"
ITEMS_CSV_FILE = "expense_items.csv"

RANGES = [
    ("week", "This week"),
    ("month", "This month"),
    ("30d", "Last 30 days"),
    ("year", "This year"),
    ("all", "All time"),
    ("custom", "Custom dates"),
]

GROUPS = [
    ("auto", "Auto"),
    ("day", "Day"),
    ("week", "Week"),
    ("month", "Month"),
    ("year", "Year"),
]

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

dashboard_bp = Blueprint("dashboard", __name__)


# ------------------------------------------------------------
# Loading
# ------------------------------------------------------------

def load_bills(user_id):

    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()

    df = pd.read_csv(CSV_FILE)

    if df.empty:
        return df

    if "user_id" in df.columns:
        df = df[df["user_id"].astype(str) == str(user_id)].copy()

    if df.empty:
        return df

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # bill date, or the upload date when the bill date could not be read
    bill_date = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    upload_date = pd.to_datetime(df["created_at"], errors="coerce")

    df["bill_date"] = bill_date.fillna(upload_date)
    df = df.dropna(subset=["bill_date"]).copy()

    df["day"] = df["bill_date"].dt.normalize()
    # old CSV rows use old names (Health & Fitness ...): show them under the new ones
    df["category"] = df["category"].fillna("Uncategorized").map(normalize_category)
    df["shop_name"] = df["shop_name"].fillna("Unknown")
    df["review"] = df["needs_review"].astype(str).str.lower() == "true"
    df["manual"] = df["extraction_method"].astype(str) == "manual"

    return df


def load_items(bill_ids):

    if not os.path.exists(ITEMS_CSV_FILE):
        return pd.DataFrame()

    items = pd.read_csv(ITEMS_CSV_FILE)

    if items.empty:
        return items

    return items[items["bill_id"].isin(bill_ids)]


# ------------------------------------------------------------
# Periods
# ------------------------------------------------------------

def parse_day(text):

    try:
        return date.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def period_bounds(key, today, first_day, last_day, start_text, end_text):
    """Returns (start, end, prev_start, prev_end)."""

    if key == "week":
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)

    elif key == "30d":
        start, end = today - timedelta(days=29), today

    elif key == "year":
        start, end = date(today.year, 1, 1), date(today.year, 12, 31)

    elif key == "all":
        start, end = first_day or today, max(today, last_day or today)

    elif key == "custom":
        start, end = parse_day(start_text), parse_day(end_text)

        if not start or not end:
            key = "month"

    if key == "month":
        start = today.replace(day=1)
        end = today.replace(day=calendar.monthrange(today.year, today.month)[1])

    if start > end:
        start, end = end, start

    if key == "month":
        prev_end = start - timedelta(days=1)
        prev_start = prev_end.replace(day=1)

    elif key == "year":
        prev_start = date(start.year - 1, 1, 1)
        prev_end = date(start.year - 1, 12, 31)

    elif key == "all":
        prev_start = prev_end = None

    else:
        length = (end - start).days + 1
        prev_end = start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=length - 1)

    return key, start, end, prev_start, prev_end


def resolve_group(group, start, end):

    if group in ("day", "week", "month", "year"):
        return group

    days = (end - start).days + 1

    if days <= 31:
        return "day"

    if days <= 120:
        return "week"

    if days <= 800:
        return "month"

    return "year"


def bucket_table(cur, start, end, group):
    """Amount and entry count per day / week / month / year, with empty
    periods shown as 0."""

    s, e = pd.Timestamp(start), pd.Timestamp(end)
    day = cur["day"]

    if group == "day":
        key = day
        full = pd.date_range(s, e, freq="D")
        label = lambda t: t.strftime("%a %d %b")

    elif group == "week":
        key = day - pd.to_timedelta(day.dt.weekday, unit="D")
        full = pd.date_range(s - pd.Timedelta(days=s.weekday()), e, freq="7D")
        label = lambda t: "Week of " + t.strftime("%d %b")

    elif group == "month":
        key = day.dt.to_period("M").dt.to_timestamp()
        full = pd.date_range(s.replace(day=1), e, freq="MS")
        label = lambda t: t.strftime("%b %Y")

    else:
        key = day.dt.to_period("Y").dt.to_timestamp()
        full = pd.date_range(pd.Timestamp(s.year, 1, 1), e, freq="YS")
        label = lambda t: str(t.year)

    grouped = (
        cur.assign(bucket=key)
        .groupby("bucket")["amount"].agg(["sum", "count"])
        .reindex(full, fill_value=0)
    )

    return [
        {"label": label(t), "amount": round(float(r["sum"]), 2), "count": int(r["count"])}
        for t, r in grouped.iterrows()
    ]


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

def money(x):

    return f"₹{x:,.2f}"


def build_summary(user_id="default_user", range_key="month", group="auto",
                  start_text=None, end_text=None):

    today = now_local().date()
    bills = load_bills(user_id)

    first_day = bills["day"].min().date() if not bills.empty else None
    last_day = bills["day"].max().date() if not bills.empty else None

    range_key, start, end, prev_start, prev_end = period_bounds(
        range_key, today, first_day, last_day, start_text, end_text
    )

    group = resolve_group(group, start, end)

    result = {
        "range": range_key,
        "group": group,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "period_label": f"{start.strftime('%d %b %Y')} – {end.strftime('%d %b %Y')}",
        "has_data": False,
    }

    if bills.empty:
        return result

    in_range = lambda df, a, b: df[(df["day"] >= pd.Timestamp(a)) & (df["day"] <= pd.Timestamp(b))]

    cur = in_range(bills, start, end)

    if cur.empty:
        return result

    total = float(cur["amount"].sum())

    prev_total = None

    if prev_start is not None:
        prev_total = float(in_range(bills, prev_start, prev_end)["amount"].sum())

    change_pct = None

    if prev_total:
        change_pct = round((total - prev_total) / prev_total * 100, 1)

    days_elapsed = max(1, (min(end, today) - start).days + 1)

    # ---- categories
    cats = (
        cur.groupby("category")["amount"].agg(["sum", "count", "mean"])
        .sort_values("sum", ascending=False)
    )

    categories = [
        {
            "name": name,
            "amount": round(float(r["sum"]), 2),
            "times": int(r["count"]),
            "average": round(float(r["mean"]), 2),
            "share": round(float(r["sum"]) / total * 100, 1) if total else 0,
        }
        for name, r in cats.iterrows()
    ]

    # ---- trend
    trend = bucket_table(cur, start, end, group)

    # ---- weekday pattern
    weekday = (
        cur.groupby(cur["day"].dt.weekday)["amount"].sum()
        .reindex(range(7), fill_value=0)
    )

    # ---- shops
    shops = (
        cur.groupby("shop_name")["amount"].agg(["sum", "count"])
        .sort_values("sum", ascending=False).head(5)
    )

    # ---- items
    items = load_items(set(cur["bill_id"]))

    top_items = []

    if not items.empty:

        for name, rows in items.assign(
            item=items["item"].astype(str).str.lower()
        ).groupby("item"):

            cat = rows["item_category"].dropna().map(normalize_category)

            top_items.append({
                "name": name,
                "times": int(len(rows)),
                "category": cat.mode().iat[0] if not cat.empty else "",
            })

        top_items = sorted(top_items, key=lambda x: -x["times"])[:8]

    # ---- biggest expense
    big = cur.loc[cur["amount"].idxmax()]

    # ---- plain-language insights
    insights = []

    if change_pct is not None:

        direction = "more" if change_pct > 0 else "less"

        insights.append(
            f"You spent {abs(change_pct)}% {direction} than the previous period "
            f"({money(prev_total)})."
        )

    if categories:

        insights.append(
            f"{categories[0]['name']} is your biggest category: "
            f"{categories[0]['share']}% of spending ({money(categories[0]['amount'])})."
        )

    if len(cur) >= 5 and weekday.max() > 0:

        insights.append(f"You spend the most on {WEEKDAYS[int(weekday.idxmax())]}s.")

    insights.append(f"On average you spend about {money(total / days_elapsed)} per day.")

    review_count = int(cur["review"].sum())

    if review_count:
        insights.append(
            f"{review_count} entr{'y' if review_count == 1 else 'ies'} "
            "may have a wrong category — please check."
        )

    recent = cur.sort_values("bill_date", ascending=False).head(10)

    result.update({
        "has_data": True,
        "total": round(total, 2),
        "previous_total": round(prev_total, 2) if prev_total is not None else None,
        "change_pct": change_pct,
        "entries": int(len(cur)),
        "per_day": round(total / days_elapsed, 2),
        "biggest": {
            "amount": round(float(big["amount"]), 2),
            "shop": big["shop_name"],
            "date": big["day"].strftime("%d %b %Y"),
        },
        "categories": categories,
        "trend": trend,
        "weekday_labels": WEEKDAYS,
        "weekday_values": [round(float(v), 2) for v in weekday.values],
        "top_shops": [
            {"name": n, "amount": round(float(r["sum"]), 2), "times": int(r["count"])}
            for n, r in shops.iterrows()
        ],
        "top_items": top_items,
        "insights": insights,
        "review_count": review_count,
        "recent": [
            {
                "date": r["day"].strftime("%d %b %Y"),
                "shop": r["shop_name"],
                "amount": round(float(r["amount"]), 2),
                "category": r["category"],
                "review": bool(r["review"]),
                "manual": bool(r["manual"]),
            }
            for _, r in recent.iterrows()
        ],
    })

    return result


@dashboard_bp.route("/dashboard")
def dashboard():

    user_id = request.args.get("user_id", "default_user")

    summary = build_summary(
        user_id,
        request.args.get("range", "month"),
        request.args.get("group", "auto"),
        request.args.get("start"),
        request.args.get("end"),
    )

    return render_template(
        "dashboard.html", s=summary, user_id=user_id,
        ranges=RANGES, groups=GROUPS, group_choice=request.args.get("group", "auto"),
    )
