"""
Earlier version of the support agent that follows a fixed sequence instead of
using open judgment about what to do (see support_agent.py). It's still the
same LangChain tool-calling loop -- the sequence below is enforced through the
system prompt, not through hardcoded Python control flow:

  fetch_unread_emails -> classify_email -> search_wiki
      -> found a relevant article? send_email with it
      -> otherwise: create_ticket, plus create_followup_reminder if URGENT

Reuses build_agent()'s tool wiring (gmail/calendar/notion/wiki, plus the
wiki-grounding gate and human-approval gate) and run_agent_loop() as-is --
only the system prompt differs from support_agent.py.

Usage: python3 old_sequential_support_agent.py   (run from inside agent/)
"""
import asyncio
from datetime import datetime

from support_agent import USER_INSTRUCTION, build_agent, run_agent_loop

SEQUENTIAL_SYSTEM_PROMPT = f"""You are a support agent with access to a real inbox,
a real Notion ticket system, and a real calendar. Current date/time:
{datetime.now().isoformat(timespec="minutes")}.

Follow this exact sequence, in this order, with no deviation:
1. fetch_unread_emails (limit=1) to get the email.
2. classify_email on it to get a category and priority.
3. search_wiki for a relevant policy/FAQ article.
4. If search_wiki found a relevant article: send_email with that article's
   content as the reply -- do not create a ticket in this case.
5. If search_wiki found nothing relevant: create_ticket instead, using the
   category/priority from step 2.
6. Only if the priority from step 2 is URGENT: also call
   create_followup_reminder for a few hours from now.
7. Stop once the sequence above is complete and give a short final summary.

Unlike a normal support agent, do not use your own judgment to skip, reorder,
or combine these steps.
"""


async def run_sequential_agent() -> None:
    llm, tools_by_name = await build_agent()
    await run_agent_loop(llm, tools_by_name, USER_INSTRUCTION, system_prompt=SEQUENTIAL_SYSTEM_PROMPT)


if __name__ == "__main__":
    asyncio.run(run_sequential_agent())
