import os
import pickle
from fastmcp import FastMCP
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from pathlib import Path


# Path configuration (relative to this script's location)

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CREDENTIALS_PATH = str(PROJECT_ROOT / "credentials/credentials.json")
TOKEN_PATH = str(PROJECT_ROOT / "credentials/token.json")

# FastMCP server
# ------------------------------------------------------------------
mcp = FastMCP("CalendarServer")
SCOPES = ['https://www.googleapis.com/auth/calendar.events']


# Authentication helper
def get_calendar_service():
    """
    Authenticate and return the Google Calendar API service instance.
    Uses the same token.json (pickle format) as Gmail.
    """
    creds = None

    # 1. Load existing token (pickle format)
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, 'rb') as f:
            creds = pickle.load(f)          

    # 2. If credentials are invalid or missing, start OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the new token in pickle format (matching Gmail)
        with open(TOKEN_PATH, 'wb') as f:
            pickle.dump(creds, f)           

    return build('calendar', 'v3', credentials=creds)


# MCP Tool: Create a follow‑up reminder
@mcp.tool()
def create_followup_reminder(summary: str, description: str, start_time_iso: str, end_time_iso: str) -> str:
    """
    Create a follow-up reminder or event on Google Calendar.
    The LLM uses this tool to schedule a time to re-check a customer ticket.

    Args:
        summary: Title of the event (e.g., "Follow up on Ticket #123").
        description: Details about what needs to be checked.
        start_time_iso: Start time in ISO 8601 format (e.g., "2026-06-07T10:00:00+02:00").
        end_time_iso: End time in ISO 8601 format (e.g., "2026-06-07T10:30:00+02:00").
    """
    try:
        service = get_calendar_service()

        event_body = {
            'summary': summary,
            'description': description,
            'start': {'dateTime': start_time_iso},
            'end': {'dateTime': end_time_iso},
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 10},
                    {'method': 'popup', 'minutes': 10},
                ],
            },
        }

        event_result = service.events().insert(calendarId='primary', body=event_body).execute()
        return f"Success: Follow-up reminder created. Event Link: {event_result.get('htmlLink')}"

    except Exception as e:
        return f"Failed to create calendar event via API: {str(e)}"

# Run the server
if __name__ == "__main__":
    mcp.run()