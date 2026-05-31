"""
Part 1: Company Onboarding Form (CLI)
--------------------------------------
Collects company profile at setup time and saves it as JSON.
Run once before starting the agent.
"""

import json
import os


INDUSTRIES = [
    "E-commerce",
    "IT/Software",
    "Healthcare",
    "Logistics",
    "Finance",
    "Education",
    "Other",
]

TONES = ["formal", "neutral", "friendly"]

DEFAULT_CATEGORIES = [
    "Order & Delivery Issues",
    "Technical / IT Problems",
    "Billing & Payment",
    "Returns & Refund Requests",
    "General Questions / FAQs",
    "Complaints & Escalations",
]


def prompt(question: str, default: str = None, required: bool = True) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"{question}{suffix}: ").strip()
        if answer:
            return answer
        if default:
            return default
        if not required:
            return ""
        print("  This field is required. Please enter a value.")


def choose(question: str, options: list, default_index: int = 0) -> str:
    print(f"\n{question}")
    for i, opt in enumerate(options):
        marker = " (default)" if i == default_index else ""
        print(f"  {i+1}. {opt}{marker}")
    while True:
        raw = input("Enter number: ").strip()
        if not raw:
            return options[default_index]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(options)}.")


def collect_roles(categories: list) -> dict:
    print("\n── Team Role Assignment (Optional) ──────────────────────")
    print("Map each email category to a responsible team email address.")
    print("Press Enter to skip any category.\n")
    roles = {}
    for cat in categories:
        role = input(f"  {cat} → responsible email: ").strip()
        if role:
            roles[cat] = role
    return roles


def run_onboarding(output_path: str = "data/company_profile.json"):
    print("\n" + "="*55)
    print("  Smart Office Support Agent — Company Onboarding")
    print("="*55)
    print("Please fill in your company details.")
    print("Fields marked [Required] must be completed.\n")

    # Required fields
    company_name = prompt("Company Name [Required]")
    industry     = choose("Industry / Company Type [Required]", INDUSTRIES)
    description  = prompt(
        "Company Description [Required]\n"
        "(2-5 sentences about what your company does and its customers)\n> "
    )
    support_email = prompt("Support Inbox Email Address [Required]")

    # Optional fields
    print("\n── Optional Fields ──────────────────────────────────────")
    tone = choose("Preferred Reply Tone", TONES, default_index=0)

    print("\nCustom Email Categories (Optional)")
    print("You can add up to 4 extra categories beyond the defaults.")
    custom_categories = []
    for i in range(4):
        cat = input(f"  Custom category {i+1} (or Enter to skip): ").strip()
        if not cat:
            break
        custom_categories.append(cat)

    all_categories = DEFAULT_CATEGORIES + custom_categories
    roles = collect_roles(all_categories)

    # Build profile
    profile = {
        "company_name":      company_name,
        "industry":          industry,
        "description":       description,
        "preferred_tone":    tone,
        "support_email":     support_email,
        "custom_categories": custom_categories,
        "team_roles":        roles,
    }

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\n Company profile saved to: {output_path}")
    print("\nSummary:")
    print(f"  Company:   {company_name}")
    print(f"  Industry:  {industry}")
    print(f"  Tone:      {tone}")
    print(f"  Roles set: {len(roles)}")
    print(f"  Categories: {len(all_categories)}")
    print("\nSetup complete. You can now start the agent.")

    return profile


if __name__ == "__main__":
    run_onboarding()
