"""
MCP server exposing Notion tools for ticket management, backed by the real
Notion API via notion-client. Ticket IDs are just the underlying Notion page IDs.

Notion's newer API splits a database into "data sources" -- pages.create still
takes a database_id, but querying/updating needs the data_source_id, which we
resolve once and cache in DATA_SOURCE_ID_CACHE (adding a Status property on first
use if the database doesn't have one yet).

list_tickets only returns the most recent few, since a database that
accumulates tickets over many runs would otherwise blow past a small local
model's context window. create_ticket does its own duplicate check
server-side instead of relying on the agent to call list_tickets first.
"""
import os
import sys
from typing import List, Literal

from fastmcp import FastMCP
from notion_client import Client

from lexical import dice_overlap, words

mcp = FastMCP("NotionTicketServer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(os.path.dirname(BASE_DIR), "agent")

NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
DATABASE_ID = os.environ.get("NOTION_DATABASE_ID")
if not NOTION_TOKEN or not DATABASE_ID:
    try:
        sys.path.insert(0, AGENT_DIR)
        from api import notion_token, notion_database_id  # fallback used elsewhere in this repo too
        NOTION_TOKEN = NOTION_TOKEN or notion_token
        DATABASE_ID = DATABASE_ID or notion_database_id
    except ImportError:
        pass

notion = Client(auth=NOTION_TOKEN) if NOTION_TOKEN else None

STATUS_OPTIONS = ["Open", "In Progress", "Closed"]
LIST_TICKETS_DEFAULT_LIMIT = 10
DUPLICATE_SUBJECT_SIMILARITY = 0.6  # dice coefficient threshold, see is_duplicate_subject
DATA_SOURCE_ID_CACHE = None


def missing_config() -> str | None:
    if not NOTION_TOKEN or not DATABASE_ID:
        return "Error: Missing NOTION_TOKEN or NOTION_DATABASE_ID environment variables, cannot connect to Notion."
    return None


def get_data_source_id() -> str:
    """Resolve and cache this database's data source id, adding a Status select
    property if it's missing one."""
    global DATA_SOURCE_ID_CACHE
    if DATA_SOURCE_ID_CACHE is not None:
        return DATA_SOURCE_ID_CACHE
    db = notion.databases.retrieve(database_id=DATABASE_ID)
    ds_id = db["data_sources"][0]["id"]
    ds = notion.data_sources.retrieve(data_source_id=ds_id)
    if "Status" not in ds.get("properties", {}):
        notion.data_sources.update(
            data_source_id=ds_id,
            properties={"Status": {"select": {"options": [{"name": s} for s in STATUS_OPTIONS]}}},
        )
    DATA_SOURCE_ID_CACHE = ds_id
    return DATA_SOURCE_ID_CACHE


def rich_text(prop: dict) -> str:
    return "".join(t["plain_text"] for t in prop.get("rich_text", []))


def title(prop: dict) -> str:
    return "".join(t["plain_text"] for t in prop.get("title", []))


def select_name(prop: dict) -> str | None:
    select = prop.get("select")
    return select["name"] if select else None


def is_duplicate_subject(a: str, b: str) -> bool:
    """Fuzzy match on word overlap, so e.g. "App crashes on login" and "App
    keeps crashing when logging in" still count as the same subject."""
    return dice_overlap(words(a), words(b)) >= DUPLICATE_SUBJECT_SIMILARITY


def find_duplicate_ticket(subject: str, category: str) -> dict | None:
    """Queries Notion directly instead of reusing list_tickets, so older
    tickets outside its recent-only window still get caught."""
    results = notion.data_sources.query(
        data_source_id=get_data_source_id(),
        filter={"property": "Category", "select": {"equals": category}},
    )["results"]
    for page in results:
        props = page["properties"]
        status = select_name(props["Status"])
        if status == "Closed":
            continue
        existing_subject = title(props["Id"])
        if is_duplicate_subject(subject, existing_subject):
            return {"ticket_id": page["id"], "subject": existing_subject, "status": status}
    return None


@mcp.tool()
def create_ticket(
    subject: str,
    category: Literal[
        "Order & Delivery Issues",
        "Technical / IT Problems",
        "Billing & Payment",
        "Returns & Refund Requests",
        "General Questions / FAQs",
        "Complaints & Escalations",
    ],
    priority: Literal["URGENT", "MEDIUM", "LOW"],
    assigned_role: str,
) -> str:
    """Record a new customer request, complaint, or ticket in the company's Notion
    system. New tickets start with status Open. Returns the ticket_id (Notion page
    ID) to use later with update_ticket_status or delete_ticket. Refuses (and returns
    the existing ticket_id instead) if a similar, still-open ticket in the same
    category already exists -- you don't need to check list_tickets yourself first."""
    error = missing_config()
    if error:
        return error

    try:
        get_data_source_id()  # make sure the Status property exists before we write it

        duplicate = find_duplicate_ticket(subject, category)
        if duplicate:
            return (
                f"Skipped: a similar open ticket already exists in {category} "
                f"(ticket_id={duplicate['ticket_id']}, subject={duplicate['subject']!r}, "
                f"status={duplicate['status']}). Not creating a duplicate."
            )

        response = notion.pages.create(
            parent={"database_id": DATABASE_ID},
            properties={
                "Id": {"title": [{"text": {"content": subject}}]},
                "Category": {"select": {"name": category}},
                "Priority": {"select": {"name": priority}},
                "Assigned Role": {"rich_text": [{"text": {"content": assigned_role}}]},
                "Status": {"select": {"name": "Open"}},
            },
        )
        return f"Success: Ticket created in Notion, ticket_id={response['id']}"
    except Exception as e:
        return f"Failed: Error occurred while creating ticket - {str(e)}"


@mcp.tool()
def list_tickets(limit: int = LIST_TICKETS_DEFAULT_LIMIT) -> List[dict]:
    """List the most recently created tickets (newest first), up to `limit` (default
    10). Note: create_ticket already refuses duplicates on its own, so you don't need
    to call this first just to avoid one -- use it to check on existing tickets'
    status, or raise `limit` if you specifically need to look further back."""
    error = missing_config()
    if error:
        return [{"error": error}]

    try:
        results = notion.data_sources.query(
            data_source_id=get_data_source_id(),
            sorts=[{"timestamp": "created_time", "direction": "descending"}],
            page_size=limit,
        )["results"]
    except Exception as e:
        return [{"error": f"Failed to list tickets - {str(e)}"}]

    tickets = []
    for page in results:
        props = page["properties"]
        tickets.append(
            {
                "ticket_id": page["id"],
                "subject": title(props["Id"]),
                "category": select_name(props["Category"]),
                "priority": select_name(props["Priority"]),
                "assigned_role": rich_text(props["Assigned Role"]),
                "status": select_name(props["Status"]),
            }
        )
    return tickets


@mcp.tool()
def update_ticket_status(ticket_id: str, status: Literal["Open", "In Progress", "Closed"]) -> str:
    """Update an existing ticket's status."""
    error = missing_config()
    if error:
        return error

    try:
        notion.pages.update(page_id=ticket_id, properties={"Status": {"select": {"name": status}}})
        return f"{ticket_id} set to {status}"
    except Exception as e:
        return f"Failed: Error occurred while updating {ticket_id} - {str(e)}"


@mcp.tool()
def delete_ticket(ticket_id: str) -> str:
    """Delete a ticket that was logged by mistake or is a duplicate. Notion has no
    permanent delete, so this archives the page instead (same as deleting it from
    the Notion UI) -- it drops out of list_tickets either way."""
    error = missing_config()
    if error:
        return error

    try:
        notion.pages.update(page_id=ticket_id, archived=True)
        return f"{ticket_id} deleted"
    except Exception as e:
        return f"Failed: Error occurred while deleting {ticket_id} - {str(e)}"


if __name__ == "__main__":
    mcp.run()
