"""
Evaluation harness for the support agent. Runs the fixed 6-email set from
eval_dataset.py through the same agent loop as a real inbox sweep, except
fetch_unread_emails is skipped (email injected directly) and send_email is
mocked to capture the draft instead of sending it. Everything else --
classify_email, search_wiki, the Notion tools, the wiki-grounding gate -- runs
for real.

Reports classification accuracy against the gold labels and saves drafted
replies for manual read-through.

Usage: python3 agent_evaluation.py
"""
import asyncio
import json
import os
import sys

import pandas as pd
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from eval_dataset import EVAL_EMAILS
from agent.support_agent import (
    DEFAULT_CATEGORIES,
    LLM_MODEL,
    LLM_PROVIDER,
    MCP_SERVERS,
    PRIORITIES,
    WikiGroundingState,
    build_chat_model,
    classify_email,
    run_agent_loop,
    with_normalized_output,
    with_wiki_required_for_reply,
    with_wiki_tracking,
)

EVAL_SYSTEM_PROMPT = f"""You are a smart office support agent. You'll be given one
customer email that has already been fetched. Handle it using your own judgment
about what it needs.

Tools available to you:
- classify_email: given a subject/body, returns a category (one of {DEFAULT_CATEGORIES})
  and a priority (one of {PRIORITIES}). Call this once, before create_ticket, so the
  ticket is classified consistently instead of by ad hoc judgment.
- search_wiki: keyword search over internal company policy/FAQ articles. Check this
  before replying or creating a ticket -- if there's a relevant article, use it to
  answer directly or to decide whether a ticket is even needed, instead of guessing.
- send_email: reply directly to the customer, using ONLY what search_wiki actually
  returned -- never answer from your own guess or general knowledge. This tool is
  hard-blocked (it will refuse to send) unless search_wiki already found a real,
  relevant article earlier for this email. If search_wiki found nothing relevant,
  do not attempt send_email at all -- log a ticket instead so a human can answer it.
- list_tickets: see the most recently created tickets (default: 10 most recent; pass
  a higher `limit` to look further back). Use this for status/context, not to prevent
  duplicates -- create_ticket already refuses a duplicate on its own.
- create_ticket: log a new support ticket -- args are subject, category/priority (use
  exactly what classify_email gave you), and assigned_role (which team should handle
  it). Refuses and returns the existing ticket_id instead if a similar, still-open
  ticket in the same category already exists.
- update_ticket_status: change a ticket's status (Open / In Progress / Closed).
- delete_ticket: remove a ticket that was logged by mistake or is a duplicate.

Judgment calls (guidance, not a checklist to follow in a fixed order):
- Replying and logging a ticket are separate decisions -- a ticket does not
  replace answering the customer. If search_wiki found a relevant, customer-facing
  article, use send_email to give the customer that grounded answer regardless of
  priority, in addition to whatever ticket the priority below calls for.
- URGENT: the customer can't use the product/service at all, or a billing error is
  actively costing them money.
- MEDIUM: a real problem, but the customer has a workaround or it isn't time-critical.
- LOW: a question or minor inconvenience you can resolve with a direct reply -- a
  ticket usually isn't necessary.
- Not every email needs every tool -- decide what's actually necessary case by case.
"""


def make_mock_send_email():
    """Stand-in for the real Gmail-backed send_email -- captures the draft
    instead of actually sending it."""
    captured = []

    async def mock_send_email(to_address: str, subject: str, body: str) -> str:
        captured.append({"to_address": to_address, "subject": subject, "body": body})
        return f"(mock) reply drafted for {to_address} -- not actually sent, this is an evaluation run."

    tool = StructuredTool.from_function(
        name="send_email",
        description="Send a reply email to the customer.",
        coroutine=mock_send_email,
    )
    return tool, captured


async def build_eval_toolkit() -> dict:
    """Connect to notion + wiki only -- no gmail (email is injected directly,
    send_email is mocked below) and no calendar (out of scope for this eval)."""
    client = MultiServerMCPClient({"notion": MCP_SERVERS["notion"], "wiki": MCP_SERVERS["wiki"]})
    raw_tools = await client.get_tools()
    return {t.name: with_normalized_output(t) for t in raw_tools}


def build_tools_for_email(base_tools: dict) -> tuple[dict, list]:
    """Fresh wiring per email -- a new WikiGroundingState and mock send_email
    capture list, so nothing leaks from one email into the next."""
    wiki_state = WikiGroundingState()
    mock_send_email, captured = make_mock_send_email()

    tools = [
        with_wiki_tracking(t, wiki_state) if name == "search_wiki" else t
        for name, t in base_tools.items()
    ]
    tools.append(with_wiki_required_for_reply(mock_send_email, wiki_state))
    tools.append(classify_email)

    return {t.name: t for t in tools}, captured


def extract_classification(messages: list) -> tuple[str | None, str | None]:
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "classify_email":
            try:
                data = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                return None, None
            return data.get("category"), data.get("priority")
    return None, None


def format_email(email: dict) -> str:
    """Same ID/From/Subject/Body shape as fetch_unread_emails, so the agent sees
    an actual email rather than a summary."""
    return f"ID: {email['id']}\nFrom: {email['from_address']}\nSubject: {email['subject']}\nBody: {email['body']}"


def user_message_for(email: dict) -> str:
    return (
        "A customer email has already been fetched for you -- fetch_unread_emails is "
        "not available in this run, do not call it. Handle this one email:\n\n"
        + format_email(email)
    )


# test.example.json's category labels differ in case/wording from DEFAULT_CATEGORIES
# (e.g. "Billing & payment queries" vs "Billing & Payment"), so normalize at load time
CATEGORY_ALIASES = {label.lower(): label for label in DEFAULT_CATEGORIES}
CATEGORY_ALIASES["billing & payment queries"] = "Billing & Payment"


def load_emails_from_json(path: str) -> list:
    """Load a test set shaped like data/test.example.json into the same shape
    EVAL_EMAILS uses."""
    with open(path) as f:
        raw = json.load(f)

    emails = []
    for entry in raw:
        gold_category = CATEGORY_ALIASES.get(entry["expected_category"].strip().lower())
        if gold_category is None:
            raise ValueError(f"Unrecognized category {entry['expected_category']!r} in {path}")
        emails.append(
            {
                "id": str(entry["email_id"]),
                "from_address": entry["from"],
                "subject": entry["subject"],
                "body": entry["body"],
                "gold_category": gold_category,
                "gold_priority": entry["expected_priority"].strip().upper(),
            }
        )
    return emails


async def load_toolkit():
    print(f"Loading agent (provider={LLM_PROVIDER}, model={LLM_MODEL})...")
    base_tools = await build_eval_toolkit()
    return base_tools, build_chat_model(0.3)


async def main(emails: list = EVAL_EMAILS, output_path: str | None = None) -> pd.DataFrame:
    base_tools, llm_base = await load_toolkit()

    rows = []
    for i, email in enumerate(emails, start=1):
        print(f"\n=== [{i}/{len(emails)}] {email['id']}: {email['subject']!r} ===")
        tools_by_name, captured = build_tools_for_email(base_tools)
        llm = llm_base.bind_tools(list(tools_by_name.values()))

        try:
            messages = await run_agent_loop(
                llm, tools_by_name, user_message_for(email), system_prompt=EVAL_SYSTEM_PROMPT
            )
            pred_category, pred_priority = extract_classification(messages)
            reply = captured[0]["body"] if captured else None
        except Exception as exc:
            # bad schema on Groq, unparseable <tool_call> locally -- either way, count as a miss
            print(f"  !! error handling this email, recording as a miss: {exc}")
            pred_category, pred_priority, reply = None, None, None

        rows.append({
            "input_email": format_email(email),
            "gold_category": email["gold_category"],
            "predicted_category": pred_category,
            "category_correct": pred_category == email["gold_category"],
            "gold_priority": email["gold_priority"],
            "predicted_priority": pred_priority,
            "priority_correct": pred_priority == email["gold_priority"],
            "agent_reply": reply,
        })

    df = pd.DataFrame(rows)

    category_accuracy = df["category_correct"].mean()
    priority_accuracy = df["priority_correct"].mean()

    print("\n" + "=" * 90)
    print(f"Category accuracy: {category_accuracy:.0%} ({df['category_correct'].sum()}/{len(df)})")
    print(f"Priority accuracy: {priority_accuracy:.0%} ({df['priority_correct'].sum()}/{len(df)})")
    print("=" * 90)

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    output_path = output_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

    return df


if __name__ == "__main__":
    # no args uses EVAL_EMAILS; one arg is a path to a JSON file shaped like
    # data/test.example.json, results saved to <name>.results.csv alongside it
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        dataset_emails = load_emails_from_json(dataset_path)
        base = os.path.splitext(os.path.basename(dataset_path))[0]
        results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{base}.results.csv")
        asyncio.run(main(dataset_emails, results_path))
    else:
        asyncio.run(main())
