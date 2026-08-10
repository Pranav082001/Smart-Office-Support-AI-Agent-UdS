"""
Minimal MCP server exposing Notion-like tools for ticket management.
category/priority are typed as Literal so FastMCP publishes them as an `enum` in the
tool's JSON schema -- the agent sees the valid options directly in the tool definition.
For real Notion, swap for the `notion-client` SDK.
"""
import itertools
from typing import List, Literal
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("notion")

_ticket_id_counter = itertools.count(1)
TICKETS_DB: List[dict] = []


@mcp.tool()
def create_ticket(
    title: str,
    summary: str,
    category: Literal[
        "Order & Delivery Issues",
        "Technical / IT Problems",
        "Billing & Payment",
        "Returns & Refund Requests",
        "General Questions / FAQs",
        "Complaints & Escalations",
    ],
    priority: Literal["URGENT", "MEDIUM", "LOW"],
    customer_email: str,
) -> dict:
    """Create a support ticket in the Notion tasks database."""
    ticket = {
        "ticket_id": f"TCK-{next(_ticket_id_counter):03d}",
        "title": title,
        "summary": summary,
        "category": category,
        "priority": priority,
        "customer_email": customer_email,
        "status": "Open",
    }
    TICKETS_DB.append(ticket)
    return ticket


@mcp.tool()
def list_tickets() -> List[dict]:
    """List all tickets currently in the Notion database. Check this before creating
    a new ticket, to avoid logging a duplicate for the same customer/issue."""
    return TICKETS_DB


@mcp.tool()
def update_ticket_status(ticket_id: str, status: Literal["Open", "In Progress", "Closed"]) -> str:
    """Update an existing ticket's status."""
    for t in TICKETS_DB:
        if t["ticket_id"] == ticket_id:
            t["status"] = status
            return f"{ticket_id} set to {status}"
    return f"No ticket found with id {ticket_id}"


@mcp.tool()
def delete_ticket(ticket_id: str) -> str:
    """Delete a ticket from the Notion tasks database (e.g. it was logged by mistake or is a duplicate)."""
    for i, t in enumerate(TICKETS_DB):
        if t["ticket_id"] == ticket_id:
            TICKETS_DB.pop(i)
            return f"{ticket_id} deleted"
    return f"No ticket found with id {ticket_id}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
