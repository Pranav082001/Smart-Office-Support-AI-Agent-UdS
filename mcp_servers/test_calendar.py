"""
Quick manual check for calendar_server.py's create_followup_reminder -- calls it
directly (no LLM, no MCP, no agent) against a near-future time slot and reports
whether the real Google Calendar API call actually went through.

First run pops a browser OAuth consent window if calendar_credentials.json
exists but there's no token yet; later runs just reuse the saved token.

Usage: python3 test_calendar.py   (run from inside mcp_servers/)
"""
from datetime import datetime, timedelta, timezone

from calendar_server import create_followup_reminder

start = datetime.now(timezone.utc) + timedelta(minutes=10)
end = start + timedelta(minutes=30)

result = create_followup_reminder(
    summary="Calendar integration test",
    description="Created by test_calendar.py to verify Google Calendar API access.",
    start_time_iso=start.isoformat(),
    end_time_iso=end.isoformat(),
)

print(result)
print("\n[PASS] Calendar integration is working." if result.startswith("Success")
      else "\n[FAIL] Calendar integration is NOT working -- see error above.")
