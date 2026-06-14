"""
Local learning memory for the classifier.

Before calling the Groq API, the classifier checks here for a near-duplicate
of a previously classified email. If one is found, its category, priority,
assigned_role and reasoning are reused directly, so only a short reply-writing
prompt has to be sent to the LLM instead of the full classification prompt.

Classified emails are saved here afterwards so memory keeps growing over time.
Entries are split into data/memory/<category>/<priority>.json buckets, so
lookups only ever scan a small slice of the memory, no matter how much it
has learned.
"""

import difflib
import json
import os
import re

MEMORY_DIR = os.getenv("MEMORY_DIR", "data/memory")
SIMILARITY_THRESHOLD = 0.82
MAX_ENTRIES_PER_BUCKET = 200

PRIORITIES = ["URGENT", "MEDIUM", "LOW"]

# Cheap rule-based pre-filter: keywords used to guess which category buckets
# are worth checking first, without involving the LLM at all.
CATEGORY_KEYWORDS = {
    "Order & Delivery Issues": {
        "order", "delivery", "shipment", "shipping", "package", "parcel",
        "tracking", "arrived", "delayed", "courier",
    },
    "Technical / IT Problems": {
        "error", "bug", "login", "password", "crash", "not working",
        "technical", "server", "app", "broken", "reset",
    },
    "Billing & Payment": {
        "invoice", "payment", "charge", "billing", "subscription",
        "price", "credit card", "paid", "overcharged",
    },
    "Returns & Refund Requests": {
        "return", "refund", "exchange", "money back", "cancel",
        "wrong item", "send back",
    },
    "General Questions / FAQs": {
        "question", "information", "inquiry", "hours", "how do i", "wondering",
    },
    "Complaints & Escalations": {
        "complaint", "unacceptable", "disappointed", "escalate",
        "manager", "angry", "terrible", "frustrated",
    },
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _bucket_path(category: str, priority: str) -> str:
    folder = os.path.join(MEMORY_DIR, _slugify(category))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{priority.lower()}.json")


def _load_bucket(category: str, priority: str) -> list:
    path = _bucket_path(category, priority)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def _save_bucket(category: str, priority: str, entries: list) -> None:
    path = _bucket_path(category, priority)
    with open(path, "w") as f:
        json.dump(entries[-MAX_ENTRIES_PER_BUCKET:], f, indent=2)


def candidate_categories(email_text: str, all_categories: list) -> list:
    """Rank categories by keyword overlap with the email, cheapest categories first.

    Categories with no keyword match (e.g. custom categories) are still
    returned, just last, so nothing is ever skipped entirely.
    """
    text = normalize_text(email_text)
    scored = []
    for category in all_categories:
        keywords = CATEGORY_KEYWORDS.get(category, set())
        score = sum(1 for kw in keywords if kw in text)
        scored.append((score, category))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [category for _, category in scored]


def find_cached_match(email_text: str, all_categories: list) -> dict:
    """Return a cached classification for a near-duplicate email, or None."""
    normalized = normalize_text(email_text)

    for category in candidate_categories(email_text, all_categories):
        for priority in PRIORITIES:
            for entry in _load_bucket(category, priority):
                ratio = difflib.SequenceMatcher(None, normalized, entry["text"]).ratio()
                if ratio >= SIMILARITY_THRESHOLD:
                    return entry

    return None


def save_to_memory(email_text: str, result: dict) -> None:
    """Store a classified email so future near-duplicates can skip classification."""
    category = result["category"]
    priority = result["priority"]

    entry = {
        "text": normalize_text(email_text),
        "category": category,
        "priority": priority,
        "assigned_role": result["assigned_role"],
        "reasoning": result["reasoning"],
    }

    entries = _load_bucket(category, priority)
    entries.append(entry)
    _save_bucket(category, priority, entries)
