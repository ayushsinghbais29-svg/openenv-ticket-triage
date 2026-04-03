"""Task-specific prompts for the GPT-4 baseline agent."""

from __future__ import annotations

SYSTEM_PROMPT = """You are an expert customer support ticket routing agent for a SaaS company.
Your job is to read customer support tickets and route them to the correct department.

Available departments:
- Billing: Payment issues, invoices, charges, refunds, subscription management
- Technical: API errors, bugs, integration problems, performance issues, data export
- General: How-to questions, feature requests, account settings, general inquiries
- Premium Support: Enterprise SLA issues, dedicated support requests, compliance, QBR

Always respond with valid JSON matching the action schema."""


CLASSIFICATION_EXAMPLES = """
Example 1:
Ticket: "Subject: API returning 500 errors | Customer: enterprise"
Reasoning: This is a technical issue - API errors need the Technical team.
Action: {"action_type": "route", "department": "Technical", "confidence": 0.95}

Example 2:
Ticket: "Subject: Incorrect charge on my account | Customer: premium"
Reasoning: Billing issue - duplicate charge needs Billing team.
Action: {"action_type": "route", "department": "Billing", "confidence": 0.9}

Example 3:
Ticket: "Subject: How do I export my data? | Customer: free"
Reasoning: General how-to question for General support.
Action: {"action_type": "route", "department": "General", "confidence": 0.85}

Example 4:
Ticket: "Subject: Enterprise SLA compliance concern | Customer: enterprise"
Reasoning: Enterprise SLA issue requires Premium Support team.
Action: {"action_type": "route", "department": "Premium Support", "confidence": 0.95}
"""

PRIORITY_EXAMPLES = """
Priority levels:
- Low: Non-urgent questions, feature requests, informational
- Medium: Issues affecting workflow but not critical
- High: Service degradation, billing errors, authentication issues
- Critical: Complete service outage, SLA violations, data loss

Example 1:
Ticket: "Subject: API returning 500 errors | Customer: enterprise"
Step 1 - Classify: {"action_type": "classify", "department": "Technical", "confidence": 0.95}
Step 2 - Priority: {"action_type": "set_priority", "priority": "Critical", "confidence": 0.9}

Example 2:
Ticket: "Subject: How do I export my data? | Customer: free"
Step 1 - Classify: {"action_type": "classify", "department": "General", "confidence": 0.9}
Step 2 - Priority: {"action_type": "set_priority", "priority": "Low", "confidence": 0.85}
"""

EFFICIENCY_EXAMPLES = """
For efficiency triage, minimize steps while maintaining accuracy.
Read tickets quickly and route them decisively.

Strategy:
1. READ the ticket to get full context
2. Immediately ROUTE based on key signals (don't over-analyze)
3. Move to the next ticket

Key signals:
- "API", "error", "500", "webhook" → Technical
- "charge", "invoice", "billing", "refund" → Billing
- "SLA", "enterprise", "dedicated", "compliance" → Premium Support
- "how", "question", "feature request", "account" → General
"""


def build_classification_prompt(obs: dict) -> str:
    """Build prompt for classification task."""
    return f"""You are routing customer support tickets. Given a ticket, output a JSON action.

{CLASSIFICATION_EXAMPLES}

Current ticket:
- Ticket ID: {obs.get('ticket_id')}
- Subject: {obs.get('subject')}
- Description: {obs.get('description')}
- Customer Tier: {obs.get('customer_tier')}
- Sentiment: {obs.get('sentiment', 0):.2f}
- Wait Time: {obs.get('wait_time_seconds', 0)}s
- Step: {obs.get('step')}/{obs.get('max_steps')}

Respond with exactly one JSON action:
{{"action_type": "route", "department": "<Billing|Technical|General|Premium Support>", "confidence": <0.0-1.0>}}"""


def build_priority_classification_prompt(obs: dict, classify_done: bool = False) -> str:
    """Build prompt for priority classification task."""
    if not classify_done:
        return f"""You are classifying and prioritizing customer support tickets.

{PRIORITY_EXAMPLES}

Current ticket:
- Subject: {obs.get('subject')}
- Description: {obs.get('description')}
- Customer Tier: {obs.get('customer_tier')}

Step 1: Classify the department.
Respond with: {{"action_type": "classify", "department": "<Billing|Technical|General|Premium Support>", "confidence": <0.0-1.0>}}"""
    else:
        return f"""Now set the priority for this ticket.

Ticket:
- Subject: {obs.get('subject')}
- Description: {obs.get('description')}
- Customer Tier: {obs.get('customer_tier')}

Respond with: {{"action_type": "set_priority", "priority": "<Low|Medium|High|Critical>", "confidence": <0.0-1.0>}}"""


def build_efficiency_prompt(obs: dict) -> str:
    """Build prompt for efficiency triage task."""
    return f"""Efficiency triage: Route tickets FAST and accurately.

{EFFICIENCY_EXAMPLES}

Ticket #{obs.get('tickets_processed', 0) + 1}/{obs.get('tickets_processed', 0) + obs.get('tickets_remaining', 1)}:
- Subject: {obs.get('subject')}
- Customer Tier: {obs.get('customer_tier')}
- Wait Time: {obs.get('wait_time_seconds', 0)}s
- Step: {obs.get('step')}/{obs.get('max_steps')}

Quickly route to correct department:
{{"action_type": "route", "department": "<Billing|Technical|General|Premium Support>", "confidence": <0.0-1.0>}}"""
