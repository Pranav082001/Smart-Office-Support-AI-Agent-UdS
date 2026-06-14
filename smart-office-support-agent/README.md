# Smart Office Support Agent

A domain-agnostic LLM-based agent that automates customer support workflows using MCP (Model Context Protocol).

**Team:** Fahad Khalid · Pranav Kushare · Qiuyan LI  
**Course:** Software Project: LLM-Based Agents — Saarland University 2025  
**LLM:** LLaMA 3.1 8B via Groq API (free)

---

## Project Structure

```
smart-office-support-agent/
├── app.py                  # onboarding web app (Flask)
├── templates/              # onboarding form & success page
├── static/                 # CSS/JS for the onboarding form
├── src/
│   ├── part1_llm/          # LLM classification & prompting engine
│   │   └── classifier.py   # classify_email() — main Part 1 function
│   ├── part2_mcp/          # MCP servers & tool integration (coming next)
│   └── part3_agent/        # ReAct agent loop & evaluation (coming next)
├── data/
│   ├── company_profile.example.json
│   └── test_emails/
├── results/
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/<your-org>/smart-office-support-agent.git
cd smart-office-support-agent
```

### 2. Create virtual environment & install
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get your free Groq API key
Go to **https://console.groq.com** → sign up → API Keys → Create Key  
It's completely free, no credit card needed.

### 4. Set up environment variables
```bash
cp .env.example .env
# Open .env and paste your GROQ_API_KEY
```

### 5. Set up company profile
```bash
# Option A: copy and edit the example
cp data/company_profile.example.json data/company_profile.json

# Option B: run the onboarding web app
python app.py
# then open http://127.0.0.1:5000 and fill in the form
```

### 6. Test the classifier
```bash
python src/part1_llm/classifier.py
```

Expected output:
```json
{
  "category": "Order & Delivery Issues",
  "priority": "URGENT",
  "assigned_role": "support@techflow.de",
  "reply_draft": "Dear Anna, thank you for reaching out...",
  "reasoning": "Customer reports urgent delivery issue with event tomorrow."
}
```

---

## Available Groq Models (all free)

| Model | Speed | Best for |
|-------|-------|----------|
| `llama-3.1-8b-instant` | Fastest | Classification, quick replies |
| `llama-3.3-70b-versatile` | Slower | Complex emails, better replies |
| `mixtral-8x7b-32768` | Fast | Long emails (32k context) |

Change model in `.env`: `GROQ_MODEL=llama-3.3-70b-versatile`

---

## Local Learning Memory

Before calling the Groq API, `classify_email()` checks `data/memory/` for a
near-duplicate of a previously classified email (see `src/part1_llm/memory.py`).

- A cheap keyword-based pre-filter narrows the search to the most likely
  category before any text comparison happens.
- Entries are split into `data/memory/<category>/<priority>.json` buckets, so
  lookups stay fast even as the memory grows.
- On a match, the category/priority/assigned_role/reasoning are reused and
  only a short reply-writing prompt is sent to the LLM, cutting token usage.
- New emails (and anything with no match) are classified normally and saved
  to memory for next time.

This folder is gitignored — it's local, per-deployment learned state.

---

## Implementation Status

| Part | Description | Status |
|------|-------------|--------|
| Part 1 | Onboarding form + LLM classifier + reply generator | ✅ Done |
| Part 2 | MCP servers (email, Notion, calendar) | 🔜 Next |
| Part 3 | ReAct agent loop + evaluation pipeline | 🔜 Soon |
