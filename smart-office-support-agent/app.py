"""
Onboarding web app.

Lets a company fill in their profile (categories, tone, team roles) through
a browser form instead of a CLI prompt, and saves it to data/company_profile.json
so it can be picked up by classify_email().
"""

import json
import os

from flask import Flask, render_template, request

app = Flask(__name__)

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

PROFILE_PATH = os.getenv("COMPANY_PROFILE_PATH", "data/company_profile.json")


@app.route("/")
def index():
    return render_template(
        "onboarding.html",
        industries=INDUSTRIES,
        tones=TONES,
        default_categories=DEFAULT_CATEGORIES,
    )


@app.route("/submit", methods=["POST"])
def submit():
    form = request.form

    custom_categories = []
    team_roles = {}

    for cat, role in zip(DEFAULT_CATEGORIES, form.getlist("default_role")):
        if role.strip():
            team_roles[cat] = role.strip()

    for i in range(4):
        cat = form.get(f"custom_category_{i}", "").strip()
        role = form.get(f"custom_role_{i}", "").strip()
        if cat:
            custom_categories.append(cat)
            if role:
                team_roles[cat] = role

    profile = {
        "company_name": form.get("company_name", "").strip(),
        "industry": form.get("industry", "").strip(),
        "description": form.get("description", "").strip(),
        "preferred_tone": form.get("tone", "formal"),
        "support_email": form.get("support_email", "").strip(),
        "custom_categories": custom_categories,
        "team_roles": team_roles,
    }

    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)

    return render_template("success.html", profile=profile, profile_path=PROFILE_PATH)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
