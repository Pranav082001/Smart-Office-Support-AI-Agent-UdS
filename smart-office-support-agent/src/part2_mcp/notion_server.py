import os
from fastmcp import FastMCP
from notion_client import Client

# initializing the MCP server
mcp = FastMCP("NotionTicketServer")

# get Notion API credentials from environment variables
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
DATABASE_ID = os.environ.get("NOTION_DATABASE_ID")

# initialize Notion client if token is available`
if NOTION_TOKEN:
    notion = Client(auth=NOTION_TOKEN)

@mcp.tool()
def create_ticket(subject: str, category: str, priority: str, assigned_role: str) -> str:
    """
    record a new customer request, complaint, or ticket in the company's Notion system
    """
    if not NOTION_TOKEN or not DATABASE_ID:
        return "Error：Missing NOTION_TOKEN or NOTION_DATABASE_ID environment variables, cannot connect to Notion."

    try:
        # create data for the new page to be added to the Notion database
        new_page = {
            "parent": {"database_id": DATABASE_ID},
            "properties": {
                "Name": {
                    "title": [{"text": {"content": subject}}]
                },
                "Category": {
                    "select": {"name": category}
                },
                "Priority": {
                    "select": {"name": priority}
                },
                "Assigned Role": {
                    "rich_text": [{"text": {"content": assigned_role}}]
                }
            }
        }
        
        # call the API to create a new page in the database
        response = notion.pages.create(**new_page)
        return f"Success: Ticket created in Notion, ticket page ID is {response['id']}"
        
    except Exception as e:
        return f"Failed: Error occurred while creating ticket - {str(e)}"

if __name__ == "__main__":
    mcp.run()