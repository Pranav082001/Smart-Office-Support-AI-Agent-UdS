"""
Fixed evaluation set for agent_evaluation.py -- 6 synthetic emails, one per
DEFAULT_CATEGORIES bucket, each with a gold category/priority to check
classify_email against.

Emails 1-4 match an existing wiki article so the agent can draft a grounded
reply. Email 5 covers something no article has (enterprise demos), so it
should get a ticket instead of a reply. Email 6 has a partial wiki hit
(wiki-009, escalation policy) that's internal guidance rather than
customer-facing copy -- checks whether the agent knows not to parrot it back.
"""
from typing import TypedDict


class EvalEmail(TypedDict):
    id: str
    from_address: str
    subject: str
    body: str
    gold_category: str
    gold_priority: str


EVAL_EMAILS: list[EvalEmail] = [
    {
        "id": "eval-001",
        "from_address": "sam.delivery@example.com",
        "subject": "Where is my order? It's been over a week",
        "body": (
            "I ordered a laptop stand 8 days ago and it still hasn't arrived. The "
            "tracking page hasn't updated in 4 days. Can you tell me what's going on?"
        ),
        "gold_category": "Order & Delivery Issues",
        "gold_priority": "MEDIUM",
    },
    {
        "id": "eval-002",
        "from_address": "priya.tech@example.com",
        "subject": "App crashes immediately every time I open it",
        "body": (
            "Since yesterday, your Android app crashes the second I try to log in. "
            "I've reinstalled it twice and cleared the cache but nothing works. I "
            "can't access my account at all and I need this for work."
        ),
        "gold_category": "Technical / IT Problems",
        "gold_priority": "URGENT",
    },
    {
        "id": "eval-003",
        "from_address": "marcus.billing@example.com",
        "subject": "Charged twice for my subscription",
        "body": (
            "I just noticed two identical charges of $29.99 on my card from your "
            "company today. I only have one subscription. Can you refund the extra charge?"
        ),
        "gold_category": "Billing & Payment",
        "gold_priority": "MEDIUM",
    },
    {
        "id": "eval-004",
        "from_address": "olivia.returns@example.com",
        "subject": "Can I return an item I bought last week?",
        "body": (
            "I bought a pair of headphones 10 days ago but they're not really what I "
            "expected. Is it possible to return them for a refund? They're unused, "
            "still in the box."
        ),
        "gold_category": "Returns & Refund Requests",
        "gold_priority": "LOW",
    },
    {
        "id": "eval-005",
        "from_address": "dave.enterprise@example.com",
        "subject": "Enterprise pricing and demo",
        "body": (
            "We're a 200-person company evaluating your product for our whole team. "
            "Could someone from sales set up a demo call and walk us through "
            "enterprise pricing options?"
        ),
        "gold_category": "General Questions / FAQs",
        "gold_priority": "LOW",
    },
    {
        "id": "eval-006",
        "from_address": "angry.customer@example.com",
        "subject": "This is my third email and nobody has helped me",
        "body": (
            "I have emailed twice already about my broken order and no one has "
            "responded. This is completely unacceptable and I am seriously "
            "considering cancelling my account and never using your service again."
        ),
        "gold_category": "Complaints & Escalations",
        "gold_priority": "URGENT",
    },
]
