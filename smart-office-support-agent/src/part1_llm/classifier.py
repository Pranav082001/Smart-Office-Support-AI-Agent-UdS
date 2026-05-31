"""
Part 1: LLM Classification & Reply Engine
------------------------------------------
Uses Groq API (free) with LLaMA 3 8B.
Takes a raw email + company profile and returns:
- category
- priority
- assigned_role
- reply_draft
- reasoning
"""

import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ── Categories & Priorities ──────────────────────────────────────

DEFAULT_CATEGORIES = [
    "Order & Delivery Issues",
    "Technical / IT Problems",
    "Billing & Payment",
    "Returns & Refund Requests",
    "General Questions / FAQs",
    "Complaints & Escalations",
]

PRIORITIES = ["URGENT", "MEDIUM", "LOW"]


# ── Load company profile ─────────────────────────────────────────

def load_company_profile(path: str = None) -> dict:
    path = path or os.getenv("COMPANY_PROFILE_PATH", "data/company_profile.json")
    with open(path, "r") as f:
        return json.load(f)


# ── Get LLM instance ─────────────────────────────────────────────

def get_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )


# ── Build system prompt ──────────────────────────────────────────

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

Your job is to:
1. Read the incoming support email carefully.
2. Classify it into exactly ONE of these categories:
{categories_str}

3. Assign a priority level:
   - URGENT: needs response within 2 hours
   - MEDIUM: needs response within 24 hours
   - LOW: needs response within 72 hours

4. Assign it to the correct team role based on category:
{roles_str}

5. Write a {tone} reply draft that:
   - Addresses the customer by name if possible
   - Acknowledges their issue clearly
   - Reflects the company's products/services
   - Ends with the company name: {profile['company_name']}

IMPORTANT: You must respond ONLY with valid JSON in this exact format:
{{
  "category": "<one of the categories above>",
  "priority": "<URGENT|MEDIUM|LOW>",
  "assigned_role": "<email of responsible team>",
  "reply_draft": "<full reply text>",
  "reasoning": "<1-2 sentences explaining your classification>"
}}
Do not include any text outside the JSON."""


# ── Main classify function ───────────────────────────────────────

def classify_email(email_text: str, company_profile: dict) -> dict:
    """
    Main interface for Part 1.
    Takes raw email text and company profile dict.
    Returns structured dict with category, priority, assigned_role,
    reply_draft, and reasoning.
    """
    llm = get_llm()
    system_prompt = build_system_prompt(company_profile)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Please classify and respond to this support email:\n\n{email_text}"),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if model wraps in ```json
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ── Few-shot version ─────────────────────────────────────────────

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
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ── Quick test ───────────────────────────────────────────────────

if __name__ == "__main__":
    profile = load_company_profile("data/company_profile.example.json")

    test_email = """
    From: customer@example.com
    Subject: My order #4521 has not arrived

    Hello,
    I placed an order 5 days ago and it still hasn't arrived.
    My event is tomorrow and I urgently need this item.
    Please help as soon as possible!

    Best,
    Anna
    """

    print("Testing classify_email with Groq...")
    result = classify_email(test_email, profile)
    print(json.dumps(result, indent=2))
