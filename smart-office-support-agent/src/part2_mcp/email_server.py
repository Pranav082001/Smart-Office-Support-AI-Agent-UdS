import os
import base64
import email
import pickle
import quopri
import re
from email.message import EmailMessage
from email.header import decode_header
from fastmcp import FastMCP

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


from pathlib import Path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Initialize the FastMCP server for the Email tool
mcp = FastMCP("EmailServer_OAuth")

# Define the scope of access (Full access to read, send, and modify emails)
SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',  # read and modify emails (mark as read, delete, etc.)
    'https://www.googleapis.com/auth/gmail.send'     # send emails
]

# Read the paths from environment variables, fallback to the default paths your teammate defined
CREDENTIALS_PATH = str(PROJECT_ROOT / "credentials/credentials.json")
TOKEN_PATH = str(PROJECT_ROOT / "credentials/token.json")

def get_gmail_service():
    """
    Authenticate and return the Gmail API service instance.
    Handles the OAuth2 flow automatically.
    """
    creds = None
    
    # 1. Check if we already have a saved token (gmail_token.json)
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, 'rb') as f:
            creds = pickle.load(f) 
    
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
        with open(TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)
            
    # Build and return the Gmail API client
    return build('gmail', 'v1', credentials=creds)



def decode_subject(header):
    """
    Decode email subject or from header that may be encoded (e.g., =?UTF-8?B?...).
    """
    if header is None:
        return "(No Subject)"
    decoded_parts = []
    for chunk, encoding in decode_header(header):
        if isinstance(chunk, bytes):
            try:
                decoded_parts.append(chunk.decode(encoding or 'utf-8', errors='ignore'))
            except:
                decoded_parts.append(chunk.decode('utf-8', errors='ignore'))
        elif isinstance(chunk, str):
            decoded_parts.append(chunk)
    return ''.join(decoded_parts)

def decode_email_body_simple(mime_msg):
    """
    Extract email body without recursion.
    Prefers plain text; falls back to HTML stripped of tags.
    """
    body = None
    
    if mime_msg.is_multipart():
        for part in mime_msg.walk():
            # Skip attachments
            if part.get('Content-Disposition') and 'attachment' in part.get('Content-Disposition'):
                continue
            
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                payload = part.get_payload(decode=True)
                if payload:
                    try:
                        body = payload.decode('utf-8', errors='ignore')
                        return body  # Plain text found, return immediately
                    except:
                        pass
            elif content_type == 'text/html' and body is None:
                payload = part.get_payload(decode=True)
                if payload:
                    try:
                        html = payload.decode('utf-8', errors='ignore')
                        # Simple HTML tag removal
                        body = re.sub(r'<[^>]+>', ' ', html)
                        body = re.sub(r'\s+', ' ', body).strip()
                    except:
                        pass
    else:
        # Not multipart: directly extract
        content_type = mime_msg.get_content_type()
        payload = mime_msg.get_payload(decode=True)
        if payload:
            try:
                if content_type == 'text/plain':
                    body = payload.decode('utf-8', errors='ignore')
                elif content_type == 'text/html':
                    html = payload.decode('utf-8', errors='ignore')
                    body = re.sub(r'<[^>]+>', ' ', html)
                    body = re.sub(r'\s+', ' ', body).strip()
            except:
                pass
    
    return body or "(No readable content)"


@mcp.tool()
def fetch_unread_emails(limit: int = 5) -> str:
    """
    Fetch the most recent unread emails from the inbox.
    After fetching, automatically marks them as read to avoid re-processing.
    """
    try:
        service = get_gmail_service()
        
        # Search for unread emails
        results = service.users().messages().list(userId='me', q='is:unread', maxResults=limit).execute()
        messages = results.get('messages', [])

        if not messages:
            return "No new unread emails found."

        fetched_emails = []
        message_ids = []  # Keep track of IDs to mark as read later

        for msg in messages:
            message_ids.append(msg['id'])  # Collect ID for batch marking
            
            # Fetch the raw email content
            txt = service.users().messages().get(userId='me', id=msg['id'], format='raw').execute()
            msg_raw = base64.urlsafe_b64decode(txt['raw'].encode('ASCII'))
            mime_msg = email.message_from_bytes(msg_raw)
            
            # Decode subject
            raw_subject = mime_msg['Subject']
            subject = decode_subject(raw_subject)
            
            # Decode sender
            raw_from = mime_msg['From']
            sender = decode_subject(raw_from)
            
            # Extract body
            body = decode_email_body_simple(mime_msg)

            email_str = f"ID: {msg['id']}\nFrom: {sender}\nSubject: {subject}\nBody: {body[:500]}\n"
            fetched_emails.append(email_str)

        # mark all fetched emails as read (remove UNREAD label)
        for msg_id in message_ids:
            service.users().messages().modify(
                userId='me',
                id=msg_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()

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