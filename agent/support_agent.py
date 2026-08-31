"""
Smart office support agent -- a plain LangChain tool-calling loop 
wired to four MCP servers: Gmail, Calendar, Notion, and a company wiki. The agent
decides per email which tools to call and in what order, until it gives a final
answer instead of another tool call.
"""
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(AGENT_DIR)
MCP_DIR = os.path.join(PROJECT_ROOT, "mcp_servers")

sys.path.insert(0, AGENT_DIR)

# default is groq (cloud API); set LLM_PROVIDER=local to use the quantized Qwen2.5-7B via llama.cpp instead
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
if LLM_PROVIDER == "groq":
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        try:
            from api import api_key as GROQ_API_KEY
        except ImportError:
            GROQ_API_KEY = None
    if not GROQ_API_KEY:
        raise RuntimeError("Set the GROQ_API_KEY environment variable before running this script.")
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

LLM_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b") if LLM_PROVIDER == "groq" else "Qwen/Qwen2.5-7B-Instruct-GGUF"


def build_chat_model(temperature: float) -> BaseChatModel:
    if LLM_PROVIDER == "local":
        from local_llm import LocalQwenChat

        return LocalQwenChat(temperature=temperature)

    from langchain_groq import ChatGroq

    return ChatGroq(model=LLM_MODEL, temperature=temperature)


MCP_SERVERS = {
    "gmail": {
        "command": "python3",
        "args": [os.path.join(MCP_DIR, "email_server.py")],
        "transport": "stdio",
    },
    "calendar": {
        "command": "python3",
        "args": [os.path.join(MCP_DIR, "calendar_server.py")],
        "transport": "stdio",
    },
    "notion": {
        "command": "python3",
        "args": [os.path.join(MCP_DIR, "notion_mcp_server.py")],
        "transport": "stdio",
    },
    "wiki": {
        "command": "python3",
        "args": [os.path.join(MCP_DIR, "wiki_mcp_server.py")],
        "transport": "stdio",
    },
}

DEFAULT_CATEGORIES = [
    "Order & Delivery Issues",
    "Technical / IT Problems",
    "Billing & Payment",
    "Returns & Refund Requests",
    "General Questions / FAQs",
    "Complaints & Escalations",
]
PRIORITIES = ["URGENT", "MEDIUM", "LOW"]

SYSTEM_PROMPT = f"""You are a smart office support agent with access to a real inbox,
a real Notion ticket system, and a real calendar. Work through the unread inbox
and handle every email appropriately, using your own judgment about what each one needs.
Current date/time: {datetime.now().isoformat(timespec="minutes")}.

Tools available to you:
- fetch_unread_emails: get the current unread emails (id, from, subject, body). Each
  fetched email is marked as read right away, so it won't be returned again on a
  later run.
- classify_email: given a subject/body, returns a category (one of {DEFAULT_CATEGORIES})
  and a priority (one of {PRIORITIES}). Call this once per email, before create_ticket,
  so tickets are classified consistently instead of by ad hoc judgment.
- search_wiki: keyword search over internal company policy/FAQ articles (delivery,
  billing, returns, technical known-issues, escalation rules, account/subscription
  questions, etc). Check this before replying or creating a ticket -- if there's a
  relevant article, use it to answer directly or to decide whether a ticket is even
  needed, instead of guessing at policy.
- send_email: reply directly to the customer, using ONLY what search_wiki actually
  returned -- never answer from your own guess or general knowledge. This tool is
  hard-blocked (it will refuse to send) unless search_wiki already found a real,
  relevant article earlier for this email. If search_wiki found nothing relevant,
  do not attempt send_email at all -- log a ticket instead so a human can answer it.
- list_tickets: see the most recently created tickets (ticket_id, subject, category,
  priority, assigned_role, status) -- default is the 10 most recent; pass a higher
  `limit` if you need to look further back. Use this to check status or context on
  existing tickets, not to prevent duplicates -- create_ticket already refuses a
  duplicate on its own (see below), so you don't need to call this first just for that.
- create_ticket: log a new support ticket in Notion -- args are subject (a short
  title for the issue), category/priority (use exactly what classify_email gave you),
  and assigned_role (which team should handle it, e.g. "IT", "Billing", "Shipping",
  "General Support" -- infer this from the category). Refuses and returns the
  existing ticket_id instead if a similar, still-open ticket in the same category
  already exists. Otherwise returns the new ticket_id: reuse that exact value for
  update_ticket_status or delete_ticket on this ticket later.
- update_ticket_status: change a ticket's status (Open / In Progress / Closed) by
  its ticket_id.
- delete_ticket: remove a ticket (by ticket_id) that was logged by mistake or is a
  duplicate.
- create_followup_reminder: put a real event on Google Calendar to re-check a ticket
  later. Use start_time_iso/end_time_iso in ISO 8601 with a timezone offset.

Judgment calls (guidance, not a checklist to follow in a fixed order):
- Replying and logging a ticket are separate decisions -- a ticket does not
  replace answering the customer. If search_wiki found a relevant, customer-facing
  article, use send_email to give the customer that grounded answer regardless of
  priority, in addition to whatever ticket/reminder the priority below calls for.
- URGENT: the customer can't use the product/service at all, or a billing error is
  actively costing them money. Log the ticket AND schedule a follow-up reminder a few
  hours out.
- MEDIUM: a real problem, but the customer has a workaround or it isn't time-critical.
  Log the ticket; a follow-up reminder is optional.
- LOW: a question or minor inconvenience you can resolve with a direct reply -- a
  ticket usually isn't necessary.
- Not every email needs every tool -- decide what's actually necessary case by case.
"""

USER_INSTRUCTION = (
    "Check the inbox for just the single most recent unread email (call "
    "fetch_unread_emails with limit=1) and handle only that one."
)

# create_ticket/update_ticket_status are low-risk and reversible so they're left
# ungated; these three need a human "y/n" before they actually run
SENSITIVE_TOOLS = {"send_email", "create_followup_reminder", "delete_ticket"}


def with_human_approval(tool: StructuredTool) -> StructuredTool:
    """Require human approval before a sensitive tool call actually runs. A
    rejection goes back to the agent as a normal tool result, not an exception,
    so it can adapt instead of blindly retrying."""
    if tool.name not in SENSITIVE_TOOLS:
        return tool

    async def guarded(**kwargs):
        print(f"\n[approval needed] agent wants to call `{tool.name}` with:", flush=True)
        for key, value in kwargs.items():
            print(f"    {key} = {value!r}", flush=True)
        print("  Approve this action? [y/N]: ", end="", flush=True)
        answer = await asyncio.to_thread(input)

        if answer.strip().lower() not in ("y", "yes"):
            print("  -> rejected by human reviewer.\n", flush=True)
            return (
                f"Human reviewer REJECTED this `{tool.name}` call and it was NOT executed. "
                "Do not retry it with the same or similar arguments. Note this in your "
                "final summary so the user knows it needs manual handling."
            )

        result = await tool.ainvoke(kwargs)
        print("  -> approved and executed.\n", flush=True)
        return result

    return StructuredTool.from_function(
        name=tool.name,
        description=f"[REQUIRES HUMAN APPROVAL] {tool.description}",
        args_schema=tool.args_schema,
        coroutine=guarded,
    )


class WikiGroundingState:
    """Shared between the search_wiki and send_email wrappers so a hit found by
    one is visible to the other."""

    def __init__(self):
        self.answer_found = False


def with_wiki_tracking(tool: StructuredTool, state: WikiGroundingState) -> StructuredTool:
    if tool.name != "search_wiki":
        return tool

    async def tracked(**kwargs):
        result = await tool.ainvoke(kwargs)
        if "No relevant article found" not in str(result):
            state.answer_found = True
        return result

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=tracked,
    )


def with_wiki_required_for_reply(tool: StructuredTool, state: WikiGroundingState) -> StructuredTool:
    """Blocks send_email unless search_wiki already found a real answer this run.
    A code-level gate rather than a prompt instruction, so it can't be talked past."""
    if tool.name != "send_email":
        return tool

    async def gated(**kwargs):
        if not state.answer_found:
            print(f"\n[blocked] agent tried to call `send_email` without a search_wiki hit this run:", flush=True)
            for key, value in kwargs.items():
                print(f"    {key} = {value!r}", flush=True)
            print("  -> blocked: no grounded answer found.\n", flush=True)
            return (
                "BLOCKED: send_email requires a prior search_wiki call that found a real "
                "answer (not 'No relevant article found') earlier in this run. No grounded "
                "answer was found, so this email was NOT sent. Do not retry send_email for "
                "this email -- log a ticket instead so a human can handle it."
            )
        return await tool.ainvoke(kwargs)

    return StructuredTool.from_function(
        name=tool.name,
        description=f"{tool.description} Requires a prior search_wiki hit this run, or the call is blocked.",
        args_schema=tool.args_schema,
        coroutine=gated,
    )


def with_normalized_output(tool: StructuredTool) -> StructuredTool:
    """Groq rejects a ToolMessage with empty content (which list_tickets etc. can
    return on a fresh run), so coerce everything to a non-empty string."""

    async def normalized(**kwargs):
        result = await tool.ainvoke(kwargs)
        if isinstance(result, list) and result and all(
            isinstance(b, dict) and isinstance(b.get("text"), str) for b in result
        ):
            result = "\n".join(b["text"] for b in result)
        if isinstance(result, str):
            return result if result.strip() else "(empty result)"
        if not result:
            return "(empty result)"
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=normalized,
    )


EMAIL_BODY_CHAR_CAP = 700


def truncate_email_dump(text: str) -> str:
    parts = text.split("\n---\n")
    out = []
    for part in parts:
        if len(part) > EMAIL_BODY_CHAR_CAP:
            part = part[:EMAIL_BODY_CHAR_CAP] + "\n... [body truncated to fit the LLM's rate limit]"
        out.append(part)
    return "\n---\n".join(out)


def with_capped_output(tool: StructuredTool) -> StructuredTool:
    """Caps fetch_unread_emails' output per email so one oversized email can't
    blow the token budget."""
    if tool.name != "fetch_unread_emails":
        return tool

    async def capped(**kwargs):
        result = await tool.ainvoke(kwargs)
        return truncate_email_dump(result) if isinstance(result, str) else result

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=capped,
    )


class EmailClassification(BaseModel):
    category: Literal[
        "Order & Delivery Issues",
        "Technical / IT Problems",
        "Billing & Payment",
        "Returns & Refund Requests",
        "General Questions / FAQs",
        "Complaints & Escalations",
    ] = Field(description="Best-fit category for this email.")
    priority: Literal["URGENT", "MEDIUM", "LOW"] = Field(description="How urgently this email needs handling.")


classifier_llm = build_chat_model(0).with_structured_output(EmailClassification)


@tool
async def classify_email(subject: str, body: str) -> str:
    """Classify a support email into a fixed category and priority, so tickets are
    logged consistently. Call this once per email, before create_ticket."""
    result = await classifier_llm.ainvoke(
        f"Classify this customer support email.\nSubject: {subject}\nBody: {body}"
    )
    return json.dumps({"category": result.category, "priority": result.priority})


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


MAX_TURNS = 15


async def build_agent():
    log(f"[setup] connecting to MCP servers (gmail, calendar, notion, wiki), model={LLM_MODEL}...")
    client = MultiServerMCPClient(MCP_SERVERS)
    raw_tools = await client.get_tools()
    wiki_state = WikiGroundingState()
    tools = [
        with_wiki_required_for_reply(
            with_human_approval(with_wiki_tracking(with_capped_output(with_normalized_output(t)), wiki_state)),
            wiki_state,
        )
        for t in raw_tools
    ]
    tools.append(classify_email)
    log(f"[setup] connected tools: {sorted(t.name for t in tools)}")
    tools_by_name = {t.name: t for t in tools}
    llm = build_chat_model(0.3).bind_tools(tools)
    log("[setup] agent ready.")
    return llm, tools_by_name


async def run_agent_loop(
    llm,
    tools_by_name: dict,
    user_message: str = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_turns: int = MAX_TURNS,
    messages: list = None,
) -> list:
    """Ask the model, run whatever tools it calls, feed results back, repeat until
    it gives a final answer. Returns the full message history.

    Pass `messages` to resume an existing trajectory instead of starting fresh
    (strategy_comparison.py uses this to append a self-correction turn)."""
    if messages is None:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

    for turn in range(1, max_turns + 1):
        response = await llm.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            if response.content:
                log(f"[turn {turn}] agent -> {response.content}")
            break

        for tc in response.tool_calls:
            log(f"[turn {turn}] agent -> calling `{tc['name']}` with {tc['args']}")

        for tc in response.tool_calls:
            tool = tools_by_name[tc["name"]]
            result = await tool.ainvoke(tc["args"])
            log(f"[turn {turn}] tool `{tc['name']}` -> {result}")
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))
    else:
        log(f"[agent] stopped after hitting the {max_turns}-turn safety cap.")

    return messages


async def run_inbox_sweep():
    llm, tools_by_name = await build_agent()
    log("[agent] starting inbox sweep...")
    await run_agent_loop(llm, tools_by_name, USER_INSTRUCTION)
    log("[agent] inbox sweep finished.")


if __name__ == "__main__":
    asyncio.run(run_inbox_sweep())
