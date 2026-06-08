import os
import base64
from email.message import EmailMessage
import email
from fastmcp import FastMCP

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Initialize the FastMCP server for the Email tool
mcp = FastMCP("EmailServer_OAuth")

# Define the scope of access (Full access to read, send, and modify emails)
SCOPES = ['https://mail.google.com/']

# Read the paths from environment variables, fallback to the default paths your teammate defined
CREDENTIALS_PATH = os.environ.get("GMAIL_CREDENTIALS_PATH")
TOKEN_PATH = os.environ.get("GMAIL_TOKEN_PATH")

def get_gmail_service():
    """
    Authenticate and return the Gmail API service instance.
    Handles the OAuth2 flow automatically.
    """
    creds = None
    
    # 1. Check if we already have a saved token (gmail_token.json)
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    
    # 2. If there are no valid credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Refresh the token if it has expired
            creds.refresh(Request())
        else:
            # First time setup: open a local browser window to ask for user consent
            # This relies on the gmail_credentials.json file downloaded from Google Cloud
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the new or refreshed token for the next run
        # This creates the gmail_token.json file
        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())
            
    # Build and return the Gmail API client
    return build('gmail', 'v1', credentials=creds)

@mcp.tool()
def fetch_unread_emails(limit: int = 5) -> str:
    """
    Fetch the most recent unread emails from the inbox.
    The LLM uses this tool to check for new customer requests.
    """
    try:
        service = get_gmail_service()
        
        # Search for unread emails in the primary inbox
        results = service.users().messages().list(userId='me', q='is:unread', maxResults=limit).execute()
        messages = results.get('messages', [])

        if not messages:
            return "No new unread emails found."

        fetched_emails = []
        for msg in messages:
            # Fetch the raw, full content of the email
            txt = service.users().messages().get(userId='me', id=msg['id'], format='raw').execute()
            
            # Decode the base64url encoded string
            msg_raw = base64.urlsafe_b64decode(txt['raw'].encode('ASCII'))
            mime_msg = email.message_from_bytes(msg_raw)
            
            # Extract headers
            subject = mime_msg['Subject']
            sender = mime_msg['From']
            body = ""

            # Extract the plain text body from the multipart email structure
            if mime_msg.is_multipart():
                for part in mime_msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
            else:
                body = mime_msg.get_payload(decode=True).decode('utf-8', errors='ignore')

            # Format the output string for the LLM
            email_str = f"ID: {msg['id']}\nFrom: {sender}\nSubject: {subject}\nBody: {body.strip()}\n"
            fetched_emails.append(email_str)

        # Return all fetched emails separated by a divider
        return "\n---\n".join(fetched_emails)

    except Exception as e:
        return f"Failed to fetch emails via Gmail API: {str(e)}"

@mcp.tool()
def send_email(to_address: str, subject: str, body: str) -> str:
    """
    Send an email reply to a customer.
    The LLM uses this tool AFTER human approval to dispatch the generated response.
    """
    try:
        service = get_gmail_service()
        
        # Construct the standard EmailMessage object
        message = EmailMessage()
        message.set_content(body)
        message['To'] = to_address
        message['From'] = 'me'  # Gmail API automatically uses the authenticated user's address
        message['Subject'] = subject

        # Encode the message into base64url format as required by the Gmail API
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {'raw': encoded_message}
        
        # Send the email via API
        send_message = service.users().messages().send(userId='me', body=create_message).execute()
        
        return f"Success: Email sent to {to_address} with subject '{subject}'. Message ID: {send_message['id']}"

    except Exception as e:
        return f"Failed to send email via Gmail API: {str(e)}"

if __name__ == "__main__":
    # Start the MCP server
    mcp.run()