"""
Smart office support agent -- a plain LangChain tool-calling loop (no LangGraph)
wired to four MCP servers over stdio (all in ../mcp_servers/, launched as subprocesses):

  - Gmail    (mcp_servers/email_server.py)      real Gmail API   -- fetch/read/send email
  - Calendar (mcp_servers/calendar_server.py)   real Google Calendar API -- schedule follow-ups
  - Notion   (mcp_servers/notion_mcp_server.py) MOCK/in-memory ticket store -- Notion isn't
             wired up yet (see notion_server.py), so this stands in for it with
             the same create/list/update/delete tool shapes a real integration
             would expose.
  - Wiki     (mcp_servers/wiki_mcp_server.py)   lexical (keyword-overlap) search over a small
             set of company policy/FAQ articles -- no vector DB, mock RAG.

The agent is given a goal and judgment guidance, not a fixed procedure. It decides
for itself, per email, which of the gmail/notion/calendar tools (if any) to call,
in what order, and how many at once: each turn, the LLM (with tools bound via
`bind_tools`) either returns tool calls -- which get executed and fed back as
ToolMessages -- or a final answer, which ends the loop.
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(AGENT_DIR)
MCP_DIR = os.path.join(PROJECT_ROOT, "mcp_servers")

# --- LLM setup (Groq) ---------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        from api import api_key as GROQ_API_KEY  # local fallback used elsewhere in this repo
    except ImportError:
        GROQ_API_KEY = None
if not GROQ_API_KEY:
    raise RuntimeError("Set the GROQ_API_KEY environment variable before running this script.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# openai/gpt-oss-120b is capped at 8000 tokens/minute on this account's Groq tier --
# a single real email plus the system prompt and 7 tool schemas blew past that in one
# request (413 rate_limit_exceeded). llama-3.1-8b-instant avoided that but its judgment
# on ambiguous emails (urgent vs medium, when to reply vs ticket) was noticeably weaker.
# llama-3.3-70b-versatile is the default now as a middle ground -- still supports tool
# calling, better reasoning than the 8b model, and (per Groq's docs at the time of this
# change) has a higher TPM ceiling than gpt-oss-120b, though not unlimited. If it starts
# 413'ing again, drop back to llama-3.1-8b-instant via GROQ_MODEL.
LLM_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- MCP servers ---------------------------------------------------------------
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
        "args": [os.path.join(MCP_DIR, "notion_mcp_server.py")],  # mock -- see module docstring
        "transport": "stdio",
    },
    "wiki": {
        "command": "python3",
        "args": [os.path.join(MCP_DIR, "wiki_mcp_server.py")],  # lexical keyword-overlap search, no vector DB
        "transport": "stdio",
    },
}

# --- Email classification -------------------------------------------------------
# Fixed taxonomy for classify_email (below) and create_ticket's category/priority
# args (see notion_mcp_server.py) -- kept as the single source of truth here so the
# classifier's output always lines up with what create_ticket will accept.
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
a (currently mocked) ticket system, and a real calendar. Work through the unread inbox
and handle every email appropriately, using your own judgment about what each one needs.
Current date/time: {datetime.now().isoformat(timespec="minutes")}.

Tools available to you:
- fetch_unread_emails: get the current unread emails (id, from, subject, body). Note:
  this server doesn't expose a mark-as-read tool, so the same emails may reappear on
  a later run -- always check list_tickets first so you don't log a duplicate ticket
  for one you've already handled.
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
- list_tickets: see tickets already logged, so you don't create a duplicate for the
  same customer/issue.
- create_ticket: log a new support ticket (title, summary, category, priority,
  customer_email) -- use the category/priority classify_email gave you. NOTE: this
  writes to an in-memory mock, not real Notion.
- update_ticket_status: change a ticket's status (Open / In Progress / Closed).
- delete_ticket: remove a ticket that was logged by mistake or is a duplicate.
- create_followup_reminder: put a real event on Google Calendar to re-check a ticket
  later. Use start_time_iso/end_time_iso in ISO 8601 with a timezone offset.

Judgment calls (guidance, not a checklist to follow in a fixed order):
- URGENT: the customer can't use the product/service at all, or a billing error is
  actively costing them money. Log the ticket AND schedule a follow-up reminder a few
  hours out.
- MEDIUM: a real problem, but the customer has a workaround or it isn't time-critical.
  Log the ticket; a follow-up reminder is optional.
- LOW: a question or minor inconvenience you can resolve with a direct reply -- a
  ticket usually isn't necessary.
- Check existing tickets before creating a new one for the same issue.
- Not every email needs every tool -- decide what's actually necessary case by case.
"""

USER_INSTRUCTION = (
    "Check the inbox for just the single most recent unread email (call "
    "fetch_unread_emails with limit=1) and handle only that one."
)

# --- Human-in-the-loop guard ----------------------------------------------------
# Tools with real, external, or destructive effects require a human "y/n" before
# they actually run. Read-only / low-risk tools (fetch_unread_emails, list_tickets,
# create_ticket, update_ticket_status -- all just writes to the in-memory mock) are
# left alone so the agent can still work through the easy cases on its own.
SENSITIVE_TOOLS = {"send_email", "create_followup_reminder", "delete_ticket"}


def with_human_approval(tool: StructuredTool) -> StructuredTool:
    """Wrap a sensitive tool so a human must approve the exact call before it executes.
    A rejection is returned to the agent as a tool result (not an exception), so it can
    adapt -- e.g. tell the user it needs manual follow-up -- instead of retrying blindly."""
    if tool.name not in SENSITIVE_TOOLS:
        return tool

    async def guarded(**kwargs):
        # flush=True: without it, this prompt can sit in stdout's buffer and never
        # become visible when stdout isn't a real terminal (piped/redirected/some
        # IDE run panels) -- the process isn't hung, but it *looks* hung because
        # you can't see it's waiting on you.
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


# --- Wiki-grounding guard for send_email -----------------------------------------
# Blocks the agent from replying off a guess: send_email can only execute if
# search_wiki already turned up a real answer (not the "no relevant article"
# placeholder) earlier in this same run. This is a code-level gate, not just a
# prompt instruction -- the LLM can't talk its way past it. WikiGroundingState is
# shared (by reference) between the search_wiki and send_email wrappers below, so
# a hit recorded by one is visible to the other.
class WikiGroundingState:
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
    """Applied outermost (after with_human_approval) so a blocked call never even
    reaches the approval prompt -- there's nothing for a human to approve if the
    agent hasn't grounded its answer in anything."""
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


# --- Output normalization (Groq compatibility) -----------------------------------
# Groq's API rejects a ToolMessage whose content is an empty string or empty array.
# Mock tools naturally return that before anything's logged (e.g. list_tickets == []
# on a fresh run), so every tool's result is coerced to a guaranteed non-empty string
# before it becomes part of the LLM's context.
def with_normalized_output(tool: StructuredTool) -> StructuredTool:
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


# --- Output cap (keeps real inbox content within the LLM's token budget) --------
# Real emails (e.g. Google's own notification mail) can carry bodies many times
# larger than a short mock email ever would. email_server.py itself is left
# unmodified -- this trims its output client-side, per email, before it reaches
# the LLM's context.
EMAIL_BODY_CHAR_CAP = 700


def _truncate_email_dump(text: str) -> str:
    parts = text.split("\n---\n")
    out = []
    for part in parts:
        if len(part) > EMAIL_BODY_CHAR_CAP:
            part = part[:EMAIL_BODY_CHAR_CAP] + "\n... [body truncated to fit the LLM's rate limit]"
        out.append(part)
    return "\n---\n".join(out)


def with_capped_output(tool: StructuredTool) -> StructuredTool:
    """Cap fetch_unread_emails' output per-email so one oversized real email can't
    blow past the LLM provider's tokens-per-minute limit in a single request."""
    if tool.name != "fetch_unread_emails":
        return tool

    async def capped(**kwargs):
        result = await tool.ainvoke(kwargs)
        return _truncate_email_dump(result) if isinstance(result, str) else result

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=capped,
    )


# --- Email classifier tool -------------------------------------------------------
# A local LangChain tool, not an MCP server: it wraps no external system and holds
# no state worth isolating in its own process, so a plain `@tool` function is enough.
# Uses `with_structured_output` (rather than leaving category/priority to the main
# agent's freeform judgment) so the result is always one of the exact allowed values.
class EmailClassification(BaseModel):
    # Mirrors DEFAULT_CATEGORIES/PRIORITIES above -- written out directly (rather than
    # derived from those lists) so static type checkers can verify the Literal, same
    # as notion_mcp_server.py's create_ticket does for the equivalent fields.
    category: Literal[
        "Order & Delivery Issues",
        "Technical / IT Problems",
        "Billing & Payment",
        "Returns & Refund Requests",
        "General Questions / FAQs",
        "Complaints & Escalations",
    ] = Field(description="Best-fit category for this email.")
    priority: Literal["URGENT", "MEDIUM", "LOW"] = Field(description="How urgently this email needs handling.")


_classifier_llm = ChatGroq(model=LLM_MODEL, temperature=0).with_structured_output(EmailClassification)


@tool
async def classify_email(subject: str, body: str) -> str:
    """Classify a support email into a fixed category and priority, so tickets are
    logged consistently. Call this once per email, before create_ticket."""
    result = await _classifier_llm.ainvoke(
        f"Classify this customer support email.\nSubject: {subject}\nBody: {body}"
    )
    return json.dumps({"category": result.category, "priority": result.priority})


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# Safety cap on LLM<->tool round trips for one inbox sweep (mirrors the old
# LangGraph recursion_limit's purpose: stop a stuck loop, not a real ceiling
# we expect to hit for "handle one email").
MAX_TURNS = 15


async def build_agent():
    _log(f"[setup] connecting to MCP servers (gmail, calendar, notion, wiki), model={LLM_MODEL}...")
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
    _log(f"[setup] connected tools: {sorted(t.name for t in tools)}")
    tools_by_name = {t.name: t for t in tools}
    llm = ChatGroq(model=LLM_MODEL, temperature=0.3).bind_tools(tools)
    _log("[setup] agent ready.")
    return llm, tools_by_name


async def run_agent_loop(
    llm, tools_by_name: dict, user_message: str, system_prompt: str = SYSTEM_PROMPT, max_turns: int = MAX_TURNS
) -> list:
    """The core ask-model / run-tool-calls / feed-results-back loop, extracted so
    agent_evaluation.py can drive it with a synthetic email instead of a live inbox
    fetch. Returns the full message history (System/Human/AI/Tool messages) so a
    caller can inspect exactly which tools were called, with what args and results.

    system_prompt is a parameter (not just the module SYSTEM_PROMPT) because it must
    describe exactly the tools actually bound to `llm` -- Groq validates tool calls
    against the request's declared tools and hard-rejects (400) any call to a tool
    the prompt mentions but bind_tools() didn't include, e.g. if a caller (like the
    evaluation harness) runs with a reduced tool set."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

    for turn in range(1, max_turns + 1):
        response = await llm.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            if response.content:
                _log(f"[turn {turn}] agent -> {response.content}")
            break

        for tc in response.tool_calls:
            _log(f"[turn {turn}] agent -> calling `{tc['name']}` with {tc['args']}")

        for tc in response.tool_calls:
            tool = tools_by_name[tc["name"]]
            result = await tool.ainvoke(tc["args"])
            _log(f"[turn {turn}] tool `{tc['name']}` -> {result}")
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))
    else:
        _log(f"[agent] stopped after hitting the {max_turns}-turn safety cap.")

    return messages


async def run_inbox_sweep():
    llm, tools_by_name = await build_agent()
    _log("[agent] starting inbox sweep...")
    await run_agent_loop(llm, tools_by_name, USER_INSTRUCTION)
    _log("[agent] inbox sweep finished.")


if __name__ == "__main__":
    asyncio.run(run_inbox_sweep())
