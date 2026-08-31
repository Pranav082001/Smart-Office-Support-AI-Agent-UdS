import os
import base64
from email.message import EmailMessage
import email
from fastmcp import FastMCP

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

mcp = FastMCP("EmailServer_OAuth")

SCOPES = ['https://mail.google.com/']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.environ.get("GMAIL_CREDENTIALS_PATH", os.path.join(BASE_DIR, "gmail_credentials.json"))
TOKEN_PATH = os.environ.get("GMAIL_TOKEN_PATH", os.path.join(BASE_DIR, "gmail_token.json"))

def get_gmail_service():
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

    return build('gmail', 'v1', credentials=creds)

@mcp.tool()
def fetch_unread_emails(limit: int = 5) -> str:
    """Fetch the most recent unread emails from the inbox."""
    try:
        service = get_gmail_service()
        results = service.users().messages().list(userId='me', q='is:unread', maxResults=limit).execute()
        messages = results.get('messages', [])

        if not messages:
            return "No new unread emails found."

        fetched_emails = []
        for msg in messages:
            txt = service.users().messages().get(userId='me', id=msg['id'], format='raw').execute()
            msg_raw = base64.urlsafe_b64decode(txt['raw'].encode('ASCII'))
            mime_msg = email.message_from_bytes(msg_raw)

            subject = mime_msg['Subject']
            sender = mime_msg['From']
            body = ""

            if mime_msg.is_multipart():
                for part in mime_msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
            else:
                body = mime_msg.get_payload(decode=True).decode('utf-8', errors='ignore')

            email_str = f"ID: {msg['id']}\nFrom: {sender}\nSubject: {subject}\nBody: {body.strip()}\n"
            fetched_emails.append(email_str)

            service.users().messages().modify(userId='me', id=msg['id'], body={'removeLabelIds': ['UNREAD']}).execute()

        return "\n---\n".join(fetched_emails)

    except Exception as e:
        return f"Failed to fetch emails via Gmail API: {str(e)}"

@mcp.tool()
def send_email(to_address: str, subject: str, body: str) -> str:
    """Send an email reply to a customer."""
    try:
        service = get_gmail_service()

        message = EmailMessage()
        message.set_content(body)
        message['To'] = to_address
        message['From'] = 'me'
        message['Subject'] = subject

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {'raw': encoded_message}
        send_message = service.users().messages().send(userId='me', body=create_message).execute()

        return f"Success: Email sent to {to_address} with subject '{subject}'. Message ID: {send_message['id']}"

    except Exception as e:
        return f"Failed to send email via Gmail API: {str(e)}"

if __name__ == "__main__":
    mcp.run()