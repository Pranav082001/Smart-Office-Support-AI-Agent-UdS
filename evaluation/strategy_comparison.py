"""
Compares four prompting/orchestration strategies on the same email set, reusing
the tools and wiki-grounding gate from agent_evaluation.py:

  - sequential: same tool-calling loop as react, but the system prompt spells out
    a fixed step order instead of leaving the sequence to the model's judgment
    (see old_sequential_support_agent.py).
  - react: the normal loop (support_agent.run_agent_loop), open judgment throughout.
  - self_correction: react to a finished answer, then one more turn asking the
    model to check its own trajectory against the guidelines and fix mistakes.
  - plan_execute: one upfront LLM call for a numbered plan (no tools executed),
    injected into the system prompt before a normal react loop follows it.

Captures latency, LLM/tool call counts, and token usage per (strategy, email).
classify_email's internal LLM call doesn't report usage_metadata like the main
loop's calls do, so `classify_calls_uninstrumented` just counts how many
happened rather than claiming exact tokens for them.

Reply quality isn't scored here -- drafted replies go to a separate CSV for
manual read-through or an LLM-as-judge pass later.

Usage: python3 strategy_comparison.py
"""
import asyncio
import os
import sys
import time

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_evaluation import (
    EVAL_SYSTEM_PROMPT,
    build_tools_for_email,
    extract_classification,
    load_emails_from_json,
    load_toolkit,
    user_message_for,
)
from eval_dataset import EVAL_EMAILS
from agent.support_agent import run_agent_loop

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def balanced_sample(emails: list, per_category: int) -> list:
    """First `per_category` emails per gold_category. test.example.json is
    grouped by category, so just taking the first N would skew the sample."""
    picked, counts = [], {}
    for email in emails:
        cat = email["gold_category"]
        if counts.get(cat, 0) < per_category:
            picked.append(email)
            counts[cat] = counts.get(cat, 0) + 1
    return picked


# two separate orderings on purpose: STRATEGIES is how the final summary table is
# sorted, CSV_ROW_ORDER is what order each email's runs actually happen in
STRATEGIES = ["sequential", "react", "self_correction", "plan_execute"]
CSV_ROW_ORDER = ["sequential", "react", "plan_execute", "self_correction"]

# same recipe as old_sequential_support_agent.py's SEQUENTIAL_SYSTEM_PROMPT, minus
# the fetch_unread_emails and create_followup_reminder steps -- the eval harness
# injects the email directly and doesn't bind a calendar tool at all
SEQUENTIAL_SYSTEM_PROMPT = """You are a support agent. You'll be given one customer
email that has already been fetched. Follow this exact sequence, in this order,
with no deviation:
1. classify_email on it to get a category and priority.
2. search_wiki for a relevant policy/FAQ article.
3. If search_wiki found a relevant article: send_email with that article's content
   as the reply -- do not create a ticket in this case.
4. If search_wiki found nothing relevant: create_ticket instead, using the
   category/priority from step 1.
5. Stop once the sequence above is complete and give a short final summary.

Unlike a normal support agent, do not use your own judgment to skip, reorder, or
combine these steps.
"""

SELF_CORRECTION_PROMPT = """Before finishing, review your own handling of this email against these checks:
1. Did you call classify_email exactly once, and use its category/priority consistently in any ticket you logged?
2. Did you call search_wiki before attempting send_email? send_email should only be used if search_wiki found a real, relevant article -- not a guess.
3. Did you call list_tickets before create_ticket, to avoid logging a duplicate for the same issue?
4. Does the priority match the guidance -- URGENT (can't use the product/service at all, or an active billing error), MEDIUM (a real problem with a workaround), LOW (a question or minor inconvenience)?

If you find a mistake, correct it now by calling the appropriate tool(s). If everything above was already handled correctly, reply with a short confirmation and do not call any more tools."""

PLANNING_SYSTEM_PROMPT = EVAL_SYSTEM_PROMPT + """

For this turn only: do NOT call any tools yet. Instead, write a short numbered plan -- which tools you intend to call, in roughly what order, and why -- based on the email below and the tools/guidance above. Plain text only."""

PLAN_SUFFIX_TEMPLATE = """

Before acting, you already produced this plan for this email:
{plan}

Follow it, but adapt if a tool result makes a planned step unnecessary or reveals it was wrong -- the plan is guidance, not a rigid script."""


def collect_metrics(messages: list) -> dict:
    llm_calls = tool_calls = classify_calls = 0
    input_tokens = output_tokens = total_tokens = 0
    for m in messages:
        if isinstance(m, AIMessage):
            llm_calls += 1
            tool_calls += len(m.tool_calls or [])
            usage = m.usage_metadata or {}
            input_tokens += usage.get("input_tokens", 0) or 0
            output_tokens += usage.get("output_tokens", 0) or 0
            total_tokens += usage.get("total_tokens", 0) or 0
        elif isinstance(m, ToolMessage) and m.name == "classify_email":
            classify_calls += 1
    return {
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "classify_calls_uninstrumented": classify_calls,
    }


def sum_metrics(*metric_dicts: dict) -> dict:
    keys = metric_dicts[0].keys()
    return {k: sum(d[k] for d in metric_dicts) for k in keys}


async def run_sequential(llm, tools_by_name: dict, email: dict) -> tuple[list, dict]:
    messages = await run_agent_loop(
        llm, tools_by_name, user_message_for(email), system_prompt=SEQUENTIAL_SYSTEM_PROMPT
    )
    return messages, collect_metrics(messages)


async def run_react(llm, tools_by_name: dict, email: dict) -> tuple[list, dict]:
    messages = await run_agent_loop(
        llm, tools_by_name, user_message_for(email), system_prompt=EVAL_SYSTEM_PROMPT
    )
    return messages, collect_metrics(messages)


async def run_self_correction(llm, tools_by_name: dict, email: dict) -> tuple[list, dict]:
    messages = await run_agent_loop(
        llm, tools_by_name, user_message_for(email), system_prompt=EVAL_SYSTEM_PROMPT, max_turns=12
    )
    messages.append(HumanMessage(content=SELF_CORRECTION_PROMPT))
    messages = await run_agent_loop(llm, tools_by_name, messages=messages, max_turns=5)
    return messages, collect_metrics(messages)


async def run_plan_execute(llm, tools_by_name: dict, email: dict) -> tuple[list, dict]:
    # done by hand rather than run_agent_loop -- this call should never execute a tool
    plan_response = await llm.ainvoke(
        [
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=user_message_for(email)),
        ]
    )
    if plan_response.tool_calls:
        plan_text = "(model attempted a tool call during planning instead of describing one; proceeding without an explicit plan)"
    else:
        plan_text = plan_response.content or "(model returned an empty plan)"

    exec_system_prompt = EVAL_SYSTEM_PROMPT + PLAN_SUFFIX_TEMPLATE.format(plan=plan_text)
    exec_messages = await run_agent_loop(
        llm, tools_by_name, user_message_for(email), system_prompt=exec_system_prompt
    )

    metrics = sum_metrics(collect_metrics([plan_response]), collect_metrics(exec_messages))
    all_messages = [plan_response] + exec_messages
    return all_messages, metrics


RUNNERS = {
    "sequential": run_sequential,
    "react": run_react,
    "self_correction": run_self_correction,
    "plan_execute": run_plan_execute,
}


async def main(
    emails: list = EVAL_EMAILS, metrics_filename: str | None = None, replies_filename: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_tools, llm_base = await load_toolkit()

    metric_rows = []
    reply_rows = []

    for i, email in enumerate(emails, start=1):
        for strategy in CSV_ROW_ORDER:
            print(f"\n=== [{strategy}] [{i}/{len(emails)}] {email['id']}: {email['subject']!r} ===")
            tools_by_name, captured = build_tools_for_email(base_tools)
            llm = llm_base.bind_tools(list(tools_by_name.values()))

            t0 = time.perf_counter()
            try:
                messages, metrics = await RUNNERS[strategy](llm, tools_by_name, email)
                pred_category, pred_priority = extract_classification(messages)
                error = None
            except Exception as exc:
                print(f"  !! error handling this email, recording as a miss: {exc}")
                messages, metrics = [], collect_metrics([])
                pred_category, pred_priority = None, None
                error = str(exc)
            latency_seconds = time.perf_counter() - t0

            reply = captured[0] if captured else None

            metric_rows.append(
                {
                    "strategy": strategy,
                    "email_id": email["id"],
                    "subject": email["subject"],
                    "gold_category": email["gold_category"],
                    "predicted_category": pred_category,
                    "gold_priority": email["gold_priority"],
                    "predicted_priority": pred_priority,
                    "latency_seconds": round(latency_seconds, 2),
                    **metrics,
                    "error": error,
                }
            )
            reply_rows.append(
                {
                    "strategy": strategy,
                    "email_id": email["id"],
                    "subject": email["subject"],
                    "to_address": reply["to_address"] if reply else None,
                    "reply_subject": reply["subject"] if reply else None,
                    "reply_body": reply["body"] if reply else None,
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    replies_df = pd.DataFrame(reply_rows)

    print("\n" + "=" * 100)
    summary = metrics_df.groupby("strategy").agg(
        avg_latency_s=("latency_seconds", "mean"),
        avg_llm_calls=("llm_calls", "mean"),
        avg_tool_calls=("tool_calls", "mean"),
        avg_input_tokens=("input_tokens", "mean"),
        avg_output_tokens=("output_tokens", "mean"),
        avg_total_tokens=("total_tokens", "mean"),
    ).reindex(STRATEGIES)
    pd.set_option("display.width", 200)
    print(summary.round(2))
    print("=" * 100)

    eval_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(eval_dir, metrics_filename or "strategy_comparison_metrics.csv")
    replies_path = os.path.join(eval_dir, replies_filename or "strategy_comparison_replies.csv")
    metrics_df.to_csv(metrics_path, index=False)
    replies_df.to_csv(replies_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved drafted replies to {replies_path}")

    return metrics_df, replies_df


if __name__ == "__main__":
    # no args uses EVAL_EMAILS; an integer arg N samples N emails, balanced
    # across categories, from data/test.example.json instead
    if len(sys.argv) > 1:
        per_category = int(sys.argv[1]) // 6
        if per_category < 1:
            raise ValueError("N must be at least 6 (>=1 per category) to sample from data/test.example.json")
        all_emails = load_emails_from_json(os.path.join(DATA_DIR, "test.example.json"))
        dataset_emails = balanced_sample(all_emails, per_category)
        n = len(dataset_emails)
        asyncio.run(
            main(dataset_emails, f"strategy_comparison_metrics_{n}.csv", f"strategy_comparison_replies_{n}.csv")
        )
    else:
        asyncio.run(main())
