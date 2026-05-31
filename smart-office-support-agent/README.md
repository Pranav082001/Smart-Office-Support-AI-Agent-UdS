# Smart Office Support Agent

A domain-agnostic LLM-based agent that automates customer support workflows using MCP (Model Context Protocol).

**Team:** Fahad Khalid · Pranav Kushare · Qiuyan LI  
**Course:** Software Project: LLM-Based Agents — Saarland University 2025  
**LLM:** LLaMA 3.1 8B via Groq API (free)

---

## Project Structure

```
smart-office-support-agent/
├── src/
│   ├── part1_llm/          # LLM classification & prompting engine
│   │   ├── classifier.py   # classify_email() — main Part 1 function
│   │   └── onboarding.py   # company onboarding form (CLI)
│   ├── part2_mcp/          # MCP servers & tool integration (coming next)
│   └── part3_agent/        # ReAct agent loop & evaluation (coming next)
├── data/
│   ├── company_profile.example.json
│   └── test_emails/
├── results/
├── scripts/
│   ├── agent.sub           # HTCondor submit file (for cluster use)
│   └── run_agent.sh        # shell script for cluster
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

# Option B: run the interactive onboarding form
python src/part1_llm/onboarding.py
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

## Implementation Status

| Part | Description | Status |
|------|-------------|--------|
| Part 1 | Onboarding form + LLM classifier + reply generator | ✅ Done |
| Part 2 | MCP servers (email, Notion, calendar) | 🔜 Next |
| Part 3 | ReAct agent loop + evaluation pipeline | 🔜 Soon |
