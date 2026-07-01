"""
Part 1: LLM Classification & Reply Engine

Uses Groq API (free) with LLaMA 3 8B. Takes a raw email + company profile
and returns category, priority, assigned_role, reply_draft, and reasoning.
"""

import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from .memory import find_cached_match, save_to_memory

load_dotenv()

DEFAULT_CATEGORIES = [
    "Order & Delivery Issues",
    "Technical / IT Problems",
    "Billing & Payment",
    "Returns & Refund Requests",
    "General Questions / FAQs",
    "Complaints & Escalations",
]

PRIORITIES = ["URGENT", "MEDIUM", "LOW"]


def load_company_profile(path: str = None) -> dict:
    path = path or os.getenv("COMPANY_PROFILE_PATH", "data/company_profile.json")
    with open(path, "r") as f:
        return json.load(f)


def get_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


PRIORITY_RANK = {"URGENT": 0, "MEDIUM": 1, "LOW": 2}


def build_system_prompt(profile: dict) -> str:
    categories = DEFAULT_CATEGORIES + profile.get("custom_categories", [])
    categories_str = "\n".join(f"  - {c}" for c in categories)
    roles_str = "\n".join(
        f"  - {cat}: {role}"
        for cat, role in profile.get("team_roles", {}).items()
    )
    tone = profile.get("preferred_tone", "formal")

    return f"""You are a smart customer support assistant for {profile['company_name']}.

Company description:
{profile['description']}

Carefully read the incoming support email. A customer may raise one issue or several distinct issues in the same message.

For EACH distinct issue you identify:
1. Classify it into exactly ONE of these categories:
{categories_str}

2. Assign a priority level:
   - URGENT: needs response within 2 hours
   - MEDIUM: needs response within 24 hours
   - LOW: needs response within 72 hours

3. Assign it to the correct team role based on category:
{roles_str}

4. Write a {tone} reply draft for ONLY that one specific issue:
   - Address the customer by name if possible
   - Acknowledge and respond to THIS issue only — do NOT mention or reference any other problems from the email
   - Reflect the company's products/services
   - End with the company name: {profile['company_name']}

IMPORTANT: You must respond ONLY with valid JSON in this exact format.
Return one ticket per distinct issue, ordered from highest to lowest priority.
If the email contains only one issue, return a tickets array with exactly one item.
Each reply_draft must cover its own issue only — never combine issues into one reply.

{{
  "tickets": [
    {{
      "category": "<one of the categories above>",
      "priority": "<URGENT|MEDIUM|LOW>",
      "assigned_role": "<email of responsible team>",
      "reply_draft": "<reply addressing only this issue>",
      "reasoning": "<1-2 sentences explaining this classification>"
    }}
  ]
}}
Do not include any text outside the JSON."""


def build_reply_prompt(profile: dict, category: str, priority: str) -> str:
    tone = profile.get("preferred_tone", "formal")

    return f"""You are a {tone} customer support assistant for {profile['company_name']}.

Company description:
{profile['description']}

The customer's email may mention multiple problems. You are responsible ONLY for the "{category}" issue (priority: {priority}).

Write a {tone} reply that:
- Addresses the customer by name if possible
- Responds to the "{category}" issue only — do NOT acknowledge or mention any other problems from the email
- Reflects the company's products/services
- Ends with the company name: {profile['company_name']}

Respond with ONLY the reply text. No JSON, no extra commentary."""


def generate_reply(email_text: str, company_profile: dict, category: str, priority: str) -> str:
    llm = get_llm()
    system_prompt = build_reply_prompt(company_profile, category, priority)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Customer email:\n\n{email_text}"),
    ]

    response = llm.invoke(messages)
    return response.content.strip()


def classify_email(email_text: str, company_profile: dict) -> dict:
    """
    Main interface for Part 1.
    Takes raw email text and company profile dict.

    Returns {"tickets": [...]} where each ticket covers one distinct issue
    found in the email, with its own category, priority, assigned_role,
    reply_draft, and reasoning. Tickets are ordered most urgent first.

    If a near-duplicate has been seen before, cached tickets are reused for
    classification and only short reply prompts are sent to the LLM.
    """
    all_categories = DEFAULT_CATEGORIES + company_profile.get("custom_categories", [])
    cached = find_cached_match(email_text, all_categories)

    if cached:
        fresh_tickets = []
        for ticket in cached["tickets"]:
            reply = generate_reply(email_text, company_profile, ticket["category"], ticket["priority"])
            fresh_tickets.append({
                "category": ticket["category"],
                "priority": ticket["priority"],
                "assigned_role": ticket["assigned_role"],
                "reply_draft": reply,
                "reasoning": ticket["reasoning"] + " (matched a previously seen email)",
            })
        return {"tickets": fresh_tickets}

    llm = get_llm()
    system_prompt = build_system_prompt(company_profile)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Please classify and respond to this support email:\n\n{email_text}"),
    ]

    response = llm.invoke(messages)
    result = json.loads(_strip_code_fence(response.content))

    result["tickets"].sort(key=lambda t: PRIORITY_RANK.get(t["priority"], 3))
    save_to_memory(email_text, result)
    return result


def classify_email_few_shot(email_text: str, company_profile: dict, examples: list) -> dict:
    """
    Few-shot version: pass a list of example dicts with 'email' and 'output' keys.
    Used for harder or ambiguous cases.
    """
    llm = get_llm()
    system_prompt = build_system_prompt(company_profile)

    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"\nExample {i}:\nEmail: {ex['email']}\nOutput: {json.dumps(ex['output'])}\n"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Here are some examples of correct classifications:\n{examples_text}\n"
            f"Now classify this email:\n\n{email_text}"
        )),
    ]

    response = llm.invoke(messages)
    result = json.loads(_strip_code_fence(response.content))
    result["tickets"].sort(key=lambda t: PRIORITY_RANK.get(t["priority"], 3))
    return result


if __name__ == "__main__":
    profile = load_company_profile("data/company_profile.example.json")

    test_email = """
    From: anna@example.com
    Subject: Multiple issues with our account

    Hello,
    I placed an order 5 days ago (#4521) and it still hasn't arrived — my event is tomorrow.
    Also, we were charged twice on our invoice this month.
    And our team cannot log in to the platform since this morning.
    Please help as soon as possible!

    Best,
    Anna
    """

    print("Testing classify_email with Groq...")
    result = classify_email(test_email, profile)
    print(f"\n{len(result['tickets'])} ticket(s) generated:\n")
    for i, ticket in enumerate(result["tickets"], 1):
        print(f"--- Ticket {i} ---")
        print(json.dumps(ticket, indent=2))
        print()
