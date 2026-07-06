import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow


os.chdir(os.path.dirname(os.path.abspath(__file__)))

SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',   # read and modify emails (mark as read, delete, etc.)
    'https://www.googleapis.com/auth/gmail.send',     # send emails
    'https://www.googleapis.com/auth/calendar.events' # manage calendar events
]

CREDENTIALS_PATH = "credentials.json"
# generate tokens
TOKEN_PATH = "token.json"

def main():
    creds = None

    # if token already exists, load it
    if os.path.exists(TOKEN_PATH):
        print(f"{TOKEN_PATH} already exists. Loading credentials from it.")
        return

    # load credentials and start the OAuth flow
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    # authorize and get the credentials
    creds = flow.run_local_server(port=8080)

    # save tokens
    with open(TOKEN_PATH, 'wb') as token:
        pickle.dump(creds, token)
    print(f"✅ token.json generated: {TOKEN_PATH}")

if __name__ == '__main__':
    main()