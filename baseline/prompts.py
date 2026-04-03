"""
Task-specific prompts for the GPT-4 baseline agent.
Each prompt includes instructions and a few-shot example.
"""

from __future__ import annotations


CLASSIFICATION_PROMPT = """You are an expert customer support ticket triage agent.
Your job is to classify a customer support ticket to the correct department.

Available departments:
- billing: Payment issues, refunds, invoices, subscription management
- technical: API errors, performance issues, integration problems, bugs
- general: Account access, feature requests, data questions, policy inquiries
- premium_support: Production outages (enterprise), SLA reviews, dedicated instance requests

Given the ticket details below, respond with a JSON object containing:
{
  "action_type": "classify",
  "department": "<department_name>",
  "reasoning": "<brief explanation>",
  "confidence": <float 0.0-1.0>
}

--- EXAMPLE ---
Ticket: Subject: Payment failed - double charged
Description: I see two charges of $99 on my statement. This is an error.

Response:
{
  "action_type": "classify",
  "department": "billing",
  "reasoning": "Duplicate charge is a billing issue requiring refund processing.",
  "confidence": 0.97
}
--- END EXAMPLE ---

Ticket to classify:
{ticket_text}

Respond with valid JSON only."""


PRIORITY_CLASSIFICATION_PROMPT = """You are an expert customer support ticket triage agent.
Your job is to classify a customer support ticket by BOTH department AND priority level.

Available departments:
- billing: Payment issues, refunds, invoices, subscription management
- technical: API errors, performance issues, integration problems, bugs
- general: Account access, feature requests, data questions, policy inquiries
- premium_support: Production outages (enterprise), SLA reviews, dedicated instances

Priority levels:
- low: Minor inconvenience, can wait several business days
- medium: Moderate impact, should be addressed within 1-2 business days
- high: Significant impact, needs attention within hours
- critical: Production down, complete outage, urgent financial issue

Additional context:
- Customer Tier: {customer_tier}
- Sentiment Score: {sentiment_score} (range: -1.0=very negative to 1.0=very positive)
- Wait Time: {wait_time} minutes

Given the ticket, respond with a JSON object:
{
  "action_type": "classify",
  "department": "<department_name>",
  "priority": "<priority_level>",
  "reasoning": "<brief explanation>",
  "confidence": <float 0.0-1.0>
}

--- EXAMPLE ---
Ticket: Subject: API returning 500 errors - production down
Description: Our entire system is down due to auth API failures. 500 enterprise users affected.
Customer Tier: enterprise | Sentiment: -0.9 | Wait: 5 minutes

Response:
{
  "action_type": "classify",
  "department": "technical",
  "priority": "critical",
  "reasoning": "Production outage affecting enterprise customer. Immediate escalation required.",
  "confidence": 0.98
}
--- END EXAMPLE ---

Ticket to classify:
{ticket_text}

Respond with valid JSON only."""


EFFICIENCY_PROMPT = """You are an expert customer support ticket triage agent optimizing for BOTH
accuracy and speed. You are processing a queue of tickets and must route them quickly.

Available departments:
- billing: Payment issues, refunds, invoices, subscription management
- technical: API errors, performance issues, integration problems, bugs
- general: Account access, feature requests, data questions, policy inquiries
- premium_support: Production outages (enterprise), SLA reviews, dedicated instances

Efficiency Rules:
- Tickets waiting >60 minutes should be ESCALATED immediately
- Critical/enterprise tickets get priority routing
- Aim to classify in minimum steps (fewer steps = higher efficiency score)

Context:
- Customer Tier: {customer_tier}
- Wait Time: {wait_time} minutes (tickets waiting >60min need urgent action)
- Tickets Remaining: {tickets_remaining}

Given the ticket, respond with a JSON object:
{
  "action_type": "route",
  "department": "<department_name>",
  "reasoning": "<brief explanation>",
  "confidence": <float 0.0-1.0>
}

Use "escalate" as action_type if wait_time > 60 and priority is critical.

--- EXAMPLE ---
Ticket: Subject: URGENT - complete outage for all users
Description: No users can access the platform. Started 2 hours ago.
Wait Time: 125 minutes | Tier: enterprise

Response:
{
  "action_type": "escalate",
  "department": "premium_support",
  "reasoning": "Critical production outage with 125min wait. Enterprise customer. Immediate escalation.",
  "confidence": 0.99
}
--- END EXAMPLE ---

Ticket to triage:
{ticket_text}

Respond with valid JSON only."""
