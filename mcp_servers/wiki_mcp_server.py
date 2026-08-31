"""
Minimal MCP server exposing a company-wiki search tool -- basically a mock RAG.
Retrieval is plain keyword overlap (no embeddings, no vector DB) so it runs with
zero setup. Swap `score` for a real embedding lookup to make this production
RAG without touching the tool's interface.
"""
import re
from typing import List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("wiki")

# tags mirror DEFAULT_CATEGORIES in support_agent.py so a wiki hit maps
# straight onto the category classify_email/create_ticket already use
ARTICLES = [
    {
        "id": "wiki-001",
        "title": "Delivery Delays & Tracking",
        "tags": ["Order & Delivery Issues"],
        "content": (
            "Standard delivery is 5-7 business days. If a package is more than 3 days "
            "late, apologize, offer a $10 credit or free reshipment, and log a ticket."
        ),
    },
    {
        "id": "wiki-002",
        "title": "Order Status & Tracking Numbers",
        "tags": ["Order & Delivery Issues"],
        "content": (
            "Tracking numbers are emailed within 24 hours of an order shipping. If a "
            "customer says they never received one, resend it from the order page first "
            "-- most cases are the email landing in spam, not a lost shipment."
        ),
    },
    {
        "id": "wiki-003",
        "title": "App Login & Crash Troubleshooting",
        "tags": ["Technical / IT Problems"],
        "content": (
            "Known issue: app version 3.2 crashes on login for some Android devices. "
            "Workaround: clear app cache, or reinstall. A fix is planned for v3.3. "
            "If the customer is still blocked, log a ticket as URGENT."
        ),
    },
    {
        "id": "wiki-004",
        "title": "Password Reset & Account Lockouts",
        "tags": ["Technical / IT Problems"],
        "content": (
            "Password reset emails can take up to 10 minutes to arrive. After 3 failed "
            "login attempts the account locks for 15 minutes as an anti-brute-force "
            "measure -- this is expected behavior, not a bug."
        ),
    },
    {
        "id": "wiki-005",
        "title": "Billing, Duplicate Charges & Refunds",
        "tags": ["Billing & Payment"],
        "content": (
            "Duplicate subscription charges usually come from a double-submitted payment "
            "form. Refunds for confirmed duplicates are automatic within 5-7 business days "
            "once a ticket is logged -- no need to escalate unless unresolved after 7 days."
        ),
    },
    {
        "id": "wiki-006",
        "title": "Discount Codes & Promo Pricing",
        "tags": ["Billing & Payment"],
        "content": (
            "Discount codes only apply if entered at checkout -- they cannot be applied "
            "retroactively by support. If a valid code visibly failed to apply (invoice "
            "shows full price despite a valid code), log a ticket and refund the difference."
        ),
    },
    {
        "id": "wiki-007",
        "title": "Returns & Exchanges Policy",
        "tags": ["Returns & Refund Requests"],
        "content": (
            "Items can be returned within 30 days of delivery for a full refund or exchange "
            "if unused. Customers can generate a prepaid return label from their order page "
            "-- no ticket needed for a standard request, just point them to that flow."
        ),
    },
    {
        "id": "wiki-008",
        "title": "Refund Processing Times",
        "tags": ["Returns & Refund Requests"],
        "content": (
            "Approved refunds appear on the original payment method within 5-10 business "
            "days depending on the customer's bank. If it's been longer, log a ticket -- "
            "don't just tell the customer to keep waiting."
        ),
    },
    {
        "id": "wiki-009",
        "title": "Escalation & Complaint Handling",
        "tags": ["Complaints & Escalations"],
        "content": (
            "Mark a ticket URGENT if the customer is threatening to cancel, has contacted "
            "support more than twice about the same issue, or the tone is unusually angry. "
            "Always acknowledge frustration before explaining next steps."
        ),
    },
    {
        "id": "wiki-010",
        "title": "Business Hours & Contacting Support",
        "tags": ["General Questions / FAQs"],
        "content": (
            "Support is available Monday-Friday, 9am-6pm in the customer's local time "
            "zone. Outside those hours, replies go out the next business day -- this can "
            "usually just be told directly to the customer without a ticket."
        ),
    },
    {
        "id": "wiki-011",
        "title": "International Shipping & Regions",
        "tags": ["General Questions / FAQs"],
        "content": (
            "We currently ship to the US, Canada, EU, and UK. Customers asking about "
            "other regions should be told international expansion is on the roadmap "
            "but has no confirmed date -- no ticket needed, just a direct reply."
        ),
    },
    {
        "id": "wiki-012",
        "title": "Managing Your Subscription",
        "tags": ["General Questions / FAQs"],
        "content": (
            "Customers can upgrade, downgrade, or cancel a subscription anytime from "
            "Account > Subscription. Cancelling stops the next renewal but doesn't "
            "refund the current billing period -- that's a Billing & Payment matter, "
            "not something to answer here."
        ),
    },
]


def score(query: str, text: str) -> int:
    q_words = set(re.findall(r"\w+", query.lower()))
    t_words = set(re.findall(r"\w+", text.lower()))
    return len(q_words & t_words)


@mcp.tool()
def search_wiki(query: str, top_k: int = 2) -> List[dict]:
    """Search the company wiki for articles relevant to a query. Use this when you're
    unsure how to summarize an issue, what company policy is, or whether something can
    be resolved with a direct reply instead of a ticket."""
    scored = [
        (score(query, a["title"] + " " + a["content"] + " " + " ".join(a["tags"])), a)
        for a in ARTICLES
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    results = [
        {"id": a["id"], "title": a["title"], "content": a["content"], "relevance": s}
        for s, a in scored[:top_k] if s > 0
    ]
    return results or [{"id": None, "title": None, "content": "No relevant article found.", "relevance": 0}]


if __name__ == "__main__":
    mcp.run(transport="stdio")
