"""
Evaluation harness for the support agent: runs the fixed 6-email set in
eval_dataset.py through the SAME agent loop as a real inbox sweep (support_agent.run_agent_loop),
with two differences from a live run:

  - fetch_unread_emails isn't available -- each email is injected directly into the
    first user message instead, so no real inbox is touched.
  - send_email is replaced with a local mock that captures the drafted reply instead
    of actually sending anything, and needs no human approval (nothing real happens),
    so the whole 6-email batch runs unattended. create_followup_reminder and the
    Gmail/Calendar MCP servers aren't wired in at all -- out of scope for this eval.
    classify_email, search_wiki, and the notion ticket tools are all the real thing,
    including the wiki-grounding gate on send_email (see support_agent.py).

Measures:
  - classification accuracy: classify_email's category/priority vs. eval_dataset's
    gold labels
  - reply quality (for manual read-through via the printed table): what the agent
    drafted, or None if it correctly found no grounded answer and drafted nothing

Usage: python3 agent_evaluation.py
"""
import asyncio
import json
import os

import pandas as pd
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from eval_dataset import EVAL_EMAILS
from agent.support_agent import (
    DEFAULT_CATEGORIES,
    LLM_MODEL,
    MCP_SERVERS,
    PRIORITIES,
    WikiGroundingState,
    classify_email,
    run_agent_loop,
    with_normalized_output,
    with_wiki_required_for_reply,
    with_wiki_tracking,
)

# A trimmed copy of support_agent.SYSTEM_PROMPT, describing only the tools actually
# bound during evaluation (no fetch_unread_emails -- the email is injected directly;
# no create_followup_reminder -- Calendar isn't wired in for this eval). This has to
# be a *separate* prompt, not the production one: Groq validates each tool call
# against the request's declared tools and hard-rejects (400) any call to a tool the
# prompt describes but bind_tools() didn't actually include.
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
- list_tickets: see tickets already logged, so you don't create a duplicate for the
  same customer/issue.
- create_ticket: log a new support ticket (title, summary, category, priority,
  customer_email) -- use the category/priority classify_email gave you.
- update_ticket_status: change a ticket's status (Open / In Progress / Closed).
- delete_ticket: remove a ticket that was logged by mistake or is a duplicate.

Judgment calls (guidance, not a checklist to follow in a fixed order):
- URGENT: the customer can't use the product/service at all, or a billing error is
  actively costing them money.
- MEDIUM: a real problem, but the customer has a workaround or it isn't time-critical.
- LOW: a question or minor inconvenience you can resolve with a direct reply -- a
  ticket usually isn't necessary.
- Not every email needs every tool -- decide what's actually necessary case by case.
"""


def _make_mock_send_email():
    """Stand-in for the real (Gmail-backed) send_email tool: captures the drafted
    reply instead of sending anything for real."""
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
    """Connect to notion (ticket mock) + wiki (search) only -- no gmail (the email is
    injected directly and send_email is mocked below) and no calendar (out of scope
    for this eval)."""
    client = MultiServerMCPClient({"notion": MCP_SERVERS["notion"], "wiki": MCP_SERVERS["wiki"]})
    raw_tools = await client.get_tools()
    return {t.name: with_normalized_output(t) for t in raw_tools}


def build_tools_for_email(base_tools: dict) -> tuple[dict, list]:
    """Fresh per-email wiring: a new WikiGroundingState and a new mock send_email
    capture list, so one email's wiki hit / drafted reply can't leak into the next."""
    wiki_state = WikiGroundingState()
    mock_send_email, captured = _make_mock_send_email()

    tools = [
        with_wiki_tracking(t, wiki_state) if name == "search_wiki" else t
        for name, t in base_tools.items()
    ]
    tools.append(with_wiki_required_for_reply(mock_send_email, wiki_state))
    tools.append(classify_email)

    return {t.name: t for t in tools}, captured


def _extract_classification(messages: list) -> tuple[str | None, str | None]:
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "classify_email":
            try:
                data = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                return None, None
            return data.get("category"), data.get("priority")
    return None, None


def _format_email(email: dict) -> str:
    """Same ID/From/Subject/Body shape email_server.py's fetch_unread_emails returns,
    so the agent (and the eval output table) sees the actual email, not a summary."""
    return f"ID: {email['id']}\nFrom: {email['from_address']}\nSubject: {email['subject']}\nBody: {email['body']}"


def _user_message_for(email: dict) -> str:
    return (
        "A customer email has already been fetched for you -- fetch_unread_emails is "
        "not available in this run, do not call it. Handle this one email:\n\n"
        + _format_email(email)
    )


async def main() -> pd.DataFrame:
    print(f"Loading agent (model={LLM_MODEL})...")
    base_tools = await build_eval_toolkit()
    llm_base = ChatGroq(model=LLM_MODEL, temperature=0.3)

    rows = []
    for i, email in enumerate(EVAL_EMAILS, start=1):
        print(f"\n=== [{i}/{len(EVAL_EMAILS)}] {email['id']}: {email['subject']!r} ===")
        tools_by_name, captured = build_tools_for_email(base_tools)
        llm = llm_base.bind_tools(list(tools_by_name.values()))

        try:
            messages = await run_agent_loop(
                llm, tools_by_name, _user_message_for(email), system_prompt=EVAL_SYSTEM_PROMPT
            )
            pred_category, pred_priority = _extract_classification(messages)
            reply = captured[0]["body"] if captured else None
        except Exception as exc:
            # Groq's strict tool-call schema validation occasionally hard-rejects a
            # malformed call (e.g. the model emitting top_k as "2" instead of 2) with
            # a 400 before returning any message at all -- there's nothing to recover
            # mid-turn for that email, but one bad turn shouldn't sink the other 5.
            print(f"  !! error handling this email, recording as a miss: {exc}")
            pred_category, pred_priority, reply = None, None, None

        rows.append({
            "input_email": _format_email(email),
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

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

    return df


if __name__ == "__main__":
    asyncio.run(main())
