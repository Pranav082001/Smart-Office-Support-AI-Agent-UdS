"""
Part 3: Classifier Evaluation

Runs classify_email() against the labeled test set in data/test_emails/
and reports category/priority accuracy.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from part1_llm.classifier import classify_email, load_company_profile

TEST_EMAILS_PATH = "data/test_emails/test_emails.json"
RESULTS_PATH = "results/evaluation_results.csv"


def run_evaluation(profile_path: str = "data/company_profile.example.json") -> pd.DataFrame:
    profile = load_company_profile(profile_path)

    with open(TEST_EMAILS_PATH) as f:
        test_emails = json.load(f)

    rows = []
    for item in test_emails:
        email_text = f"From: {item['from']}\nSubject: {item['subject']}\n\n{item['body']}"
        prediction = classify_email(email_text, profile)

        # Use the primary (most urgent) ticket for single-issue evaluation metrics.
        primary = prediction["tickets"][0]
        rows.append({
            "email_id": item["email_id"],
            "expected_category": item["expected_category"],
            "predicted_category": primary["category"],
            "expected_priority": item["expected_priority"],
            "predicted_priority": primary["priority"],
            "num_tickets": len(prediction["tickets"]),
        })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)

    print(f"Category accuracy: {accuracy_score(df.expected_category, df.predicted_category):.1%}")
    print(f"Priority accuracy: {accuracy_score(df.expected_priority, df.predicted_priority):.1%}")
    print()
    print("Category report:")
    print(classification_report(df.expected_category, df.predicted_category, zero_division=0))
    print(f"Full results saved to {RESULTS_PATH}")

    return df


if __name__ == "__main__":
    run_evaluation()
