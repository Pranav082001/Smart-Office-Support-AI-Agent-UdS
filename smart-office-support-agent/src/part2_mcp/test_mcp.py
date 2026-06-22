import os
import time
from dotenv import load_dotenv

# 1. Load environment variables from .env file
load_dotenv()

# 2. Import tools from your server files
from notion_server import create_ticket
from email_server import fetch_unread_emails, send_email
from calendar_server import create_followup_reminder

def test_pipeline():
    print("====================================")
    print("STEP 1: Testing Gmail API (Fetch)")
    print("====================================")
    # NOTE: If this is the first run, a browser window will pop up on your Mac
    # asking you to log into the test Gmail account and grant permissions.
    print("Attempting to fetch unread emails...")
    email_list = fetch_unread_emails(limit=1)
    print("Result from Gmail:\n", email_list)
    
    print("\n====================================")
    print("STEP 2: Testing Notion API (Create)")
    print("====================================")
    print("Attempting to create a ticket in Notion...")
    notion_result = create_ticket(
        subject="Test: User cannot export CSV reports",
        category="Technical / IT Problems",
        priority="High",
        assigned_role="tech-support@techflow.gmbh"
    )
    print("Result from Notion:", notion_result)

    print("\n====================================")
    print("STEP 3: Testing Calendar API (Create)")
    print("====================================")
    # NOTE: First run will pop up a browser window for Calendar OAuth permission.
    print("Attempting to create a calendar follow-up...")
    # Generating an ISO timestamp for a meeting 2 hours from now
    current_year = 2026  # Aligning with project temporal context
    start_time = f"{current_year}-06-25T14:00:00+02:00"
    end_time = f"{current_year}-06-25T14:30:00+02:00"
    
    calendar_result = create_followup_reminder(
        summary="Follow up on CSV Export Ticket",
        description="Check if the engineering team resolved the user's report generation block.",
        start_time_iso=start_time,
        end_time_iso=end_time
    )
    print("Result from Calendar:", calendar_result)

    print("\n====================================")
    print("STEP 4: Testing Gmail API (Send Email)")
    print("====================================")
    print("Attempting to send a test email...")
    
    # Replace with your actual test email address
    TEST_RECIPIENT = "your_test_email@gmail.com"  
    
    send_result = send_email(
        to_address="sosa26ss@hotmail.com",
        subject="Test: Automated reply from TechFlow Support Agent",
        body="""Dear Customer,

This is a test email sent from the TechFlow Support Agent pipeline to verify that the email sending functionality is working correctly.

If you received this email, the Gmail MCP server is successfully configured and operational.

Best regards,
TechFlow Support Agent (Automated Test)
"""
    )
    print("Result from Send Email:", send_result)

    print("\n====================================")
    print("All tests completed!")
    print("====================================")

if __name__ == "__main__":
    test_pipeline()