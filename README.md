# Smart Office Support Agent

An LLM-based agent that automates a customer support inbox: it reads an unread email,
classifies it, checks an internal company wiki for a grounded answer, drafts a reply or
logs a Notion ticket, schedules a calendar follow-up for urgent issues, and pauses for
human approval before anything customer-facing actually goes out. Built with LangChain
and four [MCP](https://modelcontextprotocol.io) servers (Gmail, Google Calendar, Notion,
Wiki).

## Repo layout

```
agent/
  support_agent.py               dynamic (judgment-driven) agent -- the main entry point
  old_sequential_support_agent.py fixed-order agent, same tools, prompt-enforced sequence
  local_llm.py                   local Qwen2.5-7B-Instruct backend (llama-cpp-python)
  api.py                         optional local fallback for API keys (see below) -- gitignored
mcp_servers/
  email_server.py                Gmail MCP server (fetch_unread_emails, send_email)
  calendar_server.py             Google Calendar MCP server (create_followup_reminder)
  notion_mcp_server.py           Notion MCP server (create/list/update/delete ticket)
  wiki_mcp_server.py             company wiki MCP server (search_wiki, lexical retrieval)
  lexical.py                     stemming/word-overlap helpers used for ticket dedup
evaluation/
  agent_evaluation.py            runs a fixed email set through the agent, checks accuracy
  strategy_comparison.py         compares sequential / react / self_correction / plan_execute
  eval_dataset.py                6 built-in synthetic emails
data/test.example.json           60 synthetic emails (Gemini-generated), 6 categories
Final/main-1.tex                 project report (LaTeX)
```

## Setup

**1. Install dependencies** (Python 3.10+):

```bash
pip install -r requirements.txt
```

If you plan to use the local model (`LLM_PROVIDER=local`), `llama-cpp-python` needs a
Metal-enabled build on macOS:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

**2. Set your API keys and credentials** (see the two sections below).

**3. Run it:**

```bash
# from the agent/ directory
python3 support_agent.py                    # real inbox sweep (default: Groq)
python3 old_sequential_support_agent.py      # fixed-order variant

# from the project root -- the evaluation scripts import agent.support_agent,
# so PYTHONPATH needs to include the project root
export PYTHONPATH=.
python3 evaluation/agent_evaluation.py                  # classification + reply accuracy on 6 emails
python3 evaluation/strategy_comparison.py               # compares all 4 orchestration strategies
python3 evaluation/strategy_comparison.py 60            # same, on the 60-email data/test.example.json set
```

## LLM provider

The default provider is **Groq** (`LLM_PROVIDER=groq`), using `openai/gpt-oss-120b`.
Set the `LLM_PROVIDER` environment variable to `local` to run entirely offline instead,
using a quantized Qwen2.5-7B-Instruct model served through `llama-cpp-python` (weights
are downloaded automatically from Hugging Face on first run, cached after that).

```bash
export LLM_PROVIDER=groq   # default -- needs GROQ_API_KEY
export LLM_PROVIDER=local  # no API key needed, but slower and requires more disk/RAM
```

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
