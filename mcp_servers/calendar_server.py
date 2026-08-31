import os
from fastmcp import FastMCP

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

mcp = FastMCP("CalendarServer")

SCOPES = ['https://www.googleapis.com/auth/calendar']

# same OAuth client as gmail_credentials.json (one Desktop app client, both APIs
# enabled), but the token is scoped to calendar so it needs its own token file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.environ.get("CALENDAR_CREDENTIALS_PATH", os.path.join(BASE_DIR, "calendar_credentials.json"))
TOKEN_PATH = os.environ.get("CALENDAR_TOKEN_PATH", os.path.join(BASE_DIR, "calendar_token.json"))

def get_calendar_service():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())

    return build('calendar', 'v3', credentials=creds)

@mcp.tool()
def create_followup_reminder(summary: str, description: str, start_time_iso: str, end_time_iso: str) -> str:
    """Create a follow-up reminder on Google Calendar. Times are ISO 8601 with a
    timezone offset, e.g. "2026-06-07T10:00:00+02:00"."""
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

if __name__ == "__main__":
    mcp.run()