# Smart Office Support Agent

An LLM-based agent that automates a customer support inbox: it reads an unread email,
classifies it, checks an internal company wiki for a grounded answer, drafts a reply or
logs a Notion ticket, schedules a calendar follow-up for urgent issues, and pauses for
human approval before anything customer-facing actually goes out. Built with LangChain
and four [MCP](https://modelcontextprotocol.io) servers (Gmail, Google Calendar, Notion,
Wiki).

## Setup

**1. Install dependencies** (Python 3.10+):

```bash
pip install -r requirements.txt
```

**2. Set your Groq API key and credentials** (see the two sections below).

**3. Run it:**

```bash
cd agent
python3 support_agent.py
```

### What `support_agent.py` does

- **No command-line arguments** — everything is controlled through environment
  variables (see [LLM provider](#llm-provider) and [Where API keys go](#where-api-keys-go)
  below), so `python3 support_agent.py` on its own is the whole command.
- **Default model: Groq**, `openai/gpt-oss-120b` (`LLM_PROVIDER` defaults to `groq`).
  Set `LLM_PROVIDER=local` first to use the offline Qwen2.5-7B model instead.
- **It only handles the single most recent unread email per run** — it calls
  `fetch_unread_emails(limit=1)`, not a full inbox sweep. That email is marked as
  read as soon as it's fetched, so re-running the script moves on to the next unread
  one each time (run it again, or put it in a loop/cron, to work through more).
- For that one email, the agent then decides for itself (based on the system
  prompt) whether to classify it, search the wiki, reply, log a Notion ticket,
  and/or schedule a calendar follow-up — it's judgment-driven, not a fixed sequence.
  For the fixed-sequence version instead, run `python3 old_sequential_support_agent.py`
  (same file/args behavior, just a stricter step-by-step prompt).
- **It will pause and prompt you in the terminal** (`Approve this action? [y/N]`)
  before actually sending an email, creating a calendar reminder, or deleting a
  ticket — type `y` to let it proceed or anything else to reject that one action.

```bash
# from the project root -- the evaluation scripts import agent.support_agent,
# so PYTHONPATH needs to include the project root
export PYTHONPATH=.
python3 evaluation/agent_evaluation.py                  # classification + reply accuracy on 6 emails
python3 evaluation/strategy_comparison.py               # compares all 4 orchestration strategies
python3 evaluation/strategy_comparison.py 60            # same, on the 60-email data/test.example.json set
```

## LLM provider

The agent runs on **Groq** by default (`LLM_PROVIDER=groq`, which you don't need to
set explicitly), using the `openai/gpt-oss-120b` model. To set it up:

1. Sign up at [console.groq.com](https://console.groq.com) and create an API key.
2. Set it as an environment variable:

   ```bash
   export GROQ_API_KEY=your_key_here
   ```

3. That's it — run `python3 support_agent.py` and it'll pick it up automatically.

You can pin a different Groq model with `export GROQ_MODEL=...` if you want.

(There's also an offline local-model path, `LLM_PROVIDER=local`, for running without
an API key — see `agent/local_llm.py` if you want to use it, it needs an extra
`llama-cpp-python` build step not covered here.)

## Where API keys go

Set these as environment variables before running anything (e.g. `export KEY=value`,
or put them in your shell profile):

| Variable | Required for | Notes |
|---|---|---|
| `GROQ_API_KEY` | `LLM_PROVIDER=groq` (default) | get one at [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL` | optional | defaults to `openai/gpt-oss-120b` |
| `NOTION_TOKEN` | ticket logging | create an internal integration at [notion.so/my-integrations](https://www.notion.so/my-integrations) |
| `NOTION_DATABASE_ID` | ticket logging | the database's ID from its URL; share the database with your integration |

Your Notion database needs these properties (create them manually — only the "Status"
select is added automatically on first run if missing):

| Property | Type |
|---|---|
| `Id` | Title (default) |
| `Category` | Select |
| `Priority` | Select |
| `Assigned Role` | Text |

There's also a fallback path already wired into the code: `agent/api.py` can define
`api_key`, `notion_token`, and `notion_database_id` as plain Python variables, and the
code will use them if the environment variables aren't set. This file is in
`.gitignore` so it's safe from accidental commits, but **environment variables are the
recommended approach** — don't put real secrets in a tracked file.

## Google credentials (Gmail + Calendar)

Both the Gmail and Calendar servers use Google OAuth2, not passwords. To set this up:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a
   project (or use an existing one).
2. Under **APIs & Services → Library**, enable the **Gmail API** and the
   **Google Calendar API**.
3. Under **APIs & Services → OAuth consent screen**, configure it (External is fine;
   "Testing" mode works as long as you add your own Google account as a test user).
4. Under **APIs & Services → Credentials**, click **Create Credentials → OAuth client
   ID**, choose **Desktop app**, and download the resulting JSON.
5. Save that JSON file as both `mcp_servers/gmail_credentials.json` and
   `mcp_servers/calendar_credentials.json` (it's the same OAuth client used for both
   scopes, so the same downloaded file works for each — the token generated later will
   differ since the scopes differ).

The first time `fetch_unread_emails`/`send_email` or `create_followup_reminder` runs, a
browser window will open asking you to log in and grant access; after that, a
`gmail_token.json` / `calendar_token.json` is saved automatically next to the
credentials file, and later runs reuse it without prompting again (until it expires or
is revoked).

You can override any of these paths with environment variables instead of the default
`mcp_servers/` location: `GMAIL_CREDENTIALS_PATH`, `GMAIL_TOKEN_PATH`,
`CALENDAR_CREDENTIALS_PATH`, `CALENDAR_TOKEN_PATH`.

**None of the `*_credentials.json` or `*_token.json` files should ever be committed** —
they're in `.gitignore`, but double-check before pushing if you're setting this up in a
repo that already has committed copies.
