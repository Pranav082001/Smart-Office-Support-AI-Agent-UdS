import os
from fastmcp import FastMCP

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Initialize the FastMCP server for the Calendar tool
mcp = FastMCP("CalendarServer")

# Define the scope of access (Access to read and edit calendar events)
SCOPES = ['https://www.googleapis.com/auth/calendar.events']

# Read the paths from environment variables, fallback to the default paths
CREDENTIALS_PATH = os.environ.get("CALENDAR_CREDENTIALS_PATH", "credentials/calendar_credentials.json")
TOKEN_PATH = os.environ.get("CALENDAR_TOKEN_PATH", "credentials/calendar_token.json")

def get_calendar_service():
    """
    Authenticate and return the Google Calendar API service instance.
    Handles the OAuth2 flow automatically, just like the Gmail service.
    """
    creds = None
    
    # 1. Check if we already have a saved token (calendar_token.json)
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    
    # 2. If there are no valid credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Refresh the token if it has expired
            creds.refresh(Request())
        else:
            # First time setup: open a local browser window to ask for user consent
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the new or refreshed token for the next run
        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())
            
    # Build and return the Calendar API client
    return build('calendar', 'v3', credentials=creds)

@mcp.tool()
def create_followup_reminder(summary: str, description: str, start_time_iso: str, end_time_iso: str) -> str:
    """
    Create a follow-up reminder or event on Google Calendar.
    The LLM uses this tool to schedule a time to re-check a customer ticket.
    
    Args:
        summary: The title of the event (e.g., "Follow up on Ticket #123").
        description: Details about what needs to be checked.
        start_time_iso: The start time in ISO 8601 format (e.g., "2026-06-07T10:00:00+02:00").
        end_time_iso: The end time in ISO 8601 format (e.g., "2026-06-07T10:30:00+02:00").
    """
    try:
        service = get_calendar_service()
        
        # Construct the event dictionary required by Google Calendar API
        event_body = {
            'summary': summary,
            'description': description,
            'start': {
                'dateTime': start_time_iso,
                # Using the time zone offset provided in the ISO string
            },
            'end': {
                'dateTime': end_time_iso,
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 10},
                    {'method': 'popup', 'minutes': 10},
                ],
            },
        }
        
        # Insert the event into the user's primary calendar
        event_result = service.events().insert(calendarId='primary', body=event_body).execute()
        
        return f"Success: Follow-up reminder created. Event Link: {event_result.get('htmlLink')}"

    except Exception as e:
        return f"Failed to create calendar event via API: {str(e)}"

if __name__ == "__main__":
    # Start the MCP server
    mcp.run()