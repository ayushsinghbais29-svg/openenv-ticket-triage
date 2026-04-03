"""Ticket generation for the OpenEnv Ticket Triage environment."""

from __future__ import annotations

import random
import time
from typing import List, Optional

from .models import CustomerTierEnum, DepartmentEnum, PriorityEnum, Ticket

BILLING_TEMPLATES = [
    {
        "subject": "Incorrect charge on my account",
        "description": (
            "I was charged twice for my subscription this month. "
            "My account shows two identical charges of $99. "
            "Please refund the duplicate charge immediately."
        ),
        "sentiment": -0.7,
        "priority": PriorityEnum.HIGH,
    },
    {
        "subject": "Invoice discrepancy - wrong amount",
        "description": (
            "The invoice I received shows $250 but I should only be on the $99/month plan. "
            "I did not upgrade and need this corrected before the due date."
        ),
        "sentiment": -0.4,
        "priority": PriorityEnum.MEDIUM,
    },
    {
        "subject": "Cannot update payment method",
        "description": (
            "I'm trying to update my credit card but the payment page keeps showing an error. "
            "My current card expires soon and I need to update it to avoid service interruption."
        ),
        "sentiment": -0.3,
        "priority": PriorityEnum.HIGH,
    },
    {
        "subject": "Request for billing statement",
        "description": (
            "I need a detailed billing statement for the past 12 months for my tax records. "
            "Please send it to my registered email address."
        ),
        "sentiment": 0.1,
        "priority": PriorityEnum.LOW,
    },
    {
        "subject": "Unexpected annual fee charge",
        "description": (
            "I was automatically switched to an annual plan without my consent and charged $1200. "
            "I want to revert to monthly billing and get a refund for the unused months."
        ),
        "sentiment": -0.8,
        "priority": PriorityEnum.CRITICAL,
    },
    {
        "subject": "Promo code not applied to bill",
        "description": (
            "I used promo code SAVE20 during signup but my first invoice shows full price. "
            "Please apply the 20% discount retroactively."
        ),
        "sentiment": -0.2,
        "priority": PriorityEnum.MEDIUM,
    },
]

TECHNICAL_TEMPLATES = [
    {
        "subject": "API returning 500 errors",
        "description": (
            "Our integration is getting 500 Internal Server Error responses from the /v2/payments endpoint. "
            "This started 2 hours ago and is affecting all our transactions. "
            "Error ID: ERR_500_1234."
        ),
        "sentiment": -0.8,
        "priority": PriorityEnum.CRITICAL,
    },
    {
        "subject": "Webhook not triggering on payment events",
        "description": (
            "Webhooks configured for payment.completed events stopped firing as of yesterday. "
            "We've verified our endpoint is reachable and the configuration is unchanged."
        ),
        "sentiment": -0.5,
        "priority": PriorityEnum.HIGH,
    },
    {
        "subject": "SDK authentication failing",
        "description": (
            "After updating the Python SDK to v3.2.1, we're seeing AuthenticationError. "
            "The same API key worked with the previous version."
        ),
        "sentiment": -0.4,
        "priority": PriorityEnum.HIGH,
    },
    {
        "subject": "Rate limit questions",
        "description": (
            "We're approaching our API rate limits and want to understand the best practices "
            "for batching requests to stay within limits during peak hours."
        ),
        "sentiment": 0.2,
        "priority": PriorityEnum.LOW,
    },
    {
        "subject": "Data export not completing",
        "description": (
            "The CSV export for our Q4 transactions has been stuck at 45% for 3 hours. "
            "We need this data for an audit tomorrow morning."
        ),
        "sentiment": -0.6,
        "priority": PriorityEnum.HIGH,
    },
    {
        "subject": "Dashboard charts not loading",
        "description": (
            "The analytics dashboard shows blank charts for the revenue section. "
            "Other sections seem fine. This has been the case since the maintenance window last night."
        ),
        "sentiment": -0.3,
        "priority": PriorityEnum.MEDIUM,
    },
]

GENERAL_TEMPLATES = [
    {
        "subject": "How do I export my data?",
        "description": (
            "I'd like to export all my account data including transaction history, "
            "customer records, and analytics. What format options are available?"
        ),
        "sentiment": 0.2,
        "priority": PriorityEnum.LOW,
    },
    {
        "subject": "Account access question",
        "description": (
            "I'm trying to add a team member to my account but I can't find the option. "
            "Is this a feature on my current plan?"
        ),
        "sentiment": 0.0,
        "priority": PriorityEnum.LOW,
    },
    {
        "subject": "Feature request: bulk import",
        "description": (
            "It would be really helpful to have a bulk import feature for customer records. "
            "Currently I have to add them one by one which is very time consuming."
        ),
        "sentiment": 0.3,
        "priority": PriorityEnum.LOW,
    },
    {
        "subject": "Password reset not working",
        "description": (
            "I requested a password reset email but haven't received it after 30 minutes. "
            "I've checked my spam folder."
        ),
        "sentiment": -0.2,
        "priority": PriorityEnum.MEDIUM,
    },
    {
        "subject": "Question about data retention policy",
        "description": (
            "Can you tell me how long you retain customer transaction data? "
            "I need this information for our compliance documentation."
        ),
        "sentiment": 0.1,
        "priority": PriorityEnum.LOW,
    },
    {
        "subject": "Language support question",
        "description": (
            "Does the dashboard support multiple languages? "
            "Our team has members who prefer Spanish and French interfaces."
        ),
        "sentiment": 0.1,
        "priority": PriorityEnum.LOW,
    },
]

PREMIUM_TEMPLATES = [
    {
        "subject": "Enterprise SLA compliance concern",
        "description": (
            "Per our enterprise SLA, downtime should not exceed 0.1% monthly. "
            "Last month's incident resulted in 4 hours of downtime which violates this. "
            "I need a formal incident report and SLA credit."
        ),
        "sentiment": -0.7,
        "priority": PriorityEnum.CRITICAL,
    },
    {
        "subject": "Dedicated account manager request",
        "description": (
            "As a platinum enterprise customer spending $50K/year, "
            "I'd like to discuss having a dedicated account manager assigned to our account."
        ),
        "sentiment": 0.3,
        "priority": PriorityEnum.HIGH,
    },
    {
        "subject": "Custom integration support needed",
        "description": (
            "We need assistance integrating with our custom ERP system. "
            "Our enterprise agreement includes professional services hours. "
            "Please assign a solutions engineer."
        ),
        "sentiment": 0.1,
        "priority": PriorityEnum.HIGH,
    },
    {
        "subject": "Quarterly business review scheduling",
        "description": (
            "It's time for our Q1 QBR. Please connect me with our enterprise success team "
            "to schedule a review of our usage, performance, and roadmap alignment."
        ),
        "sentiment": 0.4,
        "priority": PriorityEnum.MEDIUM,
    },
    {
        "subject": "Compliance documentation request",
        "description": (
            "For our upcoming SOC 2 audit, I need your latest security certifications, "
            "penetration test results, and data processing agreements."
        ),
        "sentiment": 0.0,
        "priority": PriorityEnum.HIGH,
    },
]

DEPARTMENT_TEMPLATES = {
    DepartmentEnum.BILLING: BILLING_TEMPLATES,
    DepartmentEnum.TECHNICAL: TECHNICAL_TEMPLATES,
    DepartmentEnum.GENERAL: GENERAL_TEMPLATES,
    DepartmentEnum.PREMIUM_SUPPORT: PREMIUM_TEMPLATES,
}

TIER_DEPARTMENT_WEIGHTS = {
    CustomerTierEnum.FREE: {
        DepartmentEnum.BILLING: 0.25,
        DepartmentEnum.TECHNICAL: 0.30,
        DepartmentEnum.GENERAL: 0.45,
        DepartmentEnum.PREMIUM_SUPPORT: 0.00,
    },
    CustomerTierEnum.PREMIUM: {
        DepartmentEnum.BILLING: 0.30,
        DepartmentEnum.TECHNICAL: 0.35,
        DepartmentEnum.GENERAL: 0.25,
        DepartmentEnum.PREMIUM_SUPPORT: 0.10,
    },
    CustomerTierEnum.ENTERPRISE: {
        DepartmentEnum.BILLING: 0.20,
        DepartmentEnum.TECHNICAL: 0.25,
        DepartmentEnum.GENERAL: 0.10,
        DepartmentEnum.PREMIUM_SUPPORT: 0.45,
    },
}

TIER_DISTRIBUTION = {
    CustomerTierEnum.FREE: 0.50,
    CustomerTierEnum.PREMIUM: 0.35,
    CustomerTierEnum.ENTERPRISE: 0.15,
}


class TicketGenerator:
    """Generates realistic customer support tickets for the triage environment."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._ticket_counter = 0

    def _select_tier(self) -> CustomerTierEnum:
        tiers = list(TIER_DISTRIBUTION.keys())
        weights = [TIER_DISTRIBUTION[t] for t in tiers]
        return self._rng.choices(tiers, weights=weights, k=1)[0]

    def _select_department(self, tier: CustomerTierEnum) -> DepartmentEnum:
        dept_weights = TIER_DEPARTMENT_WEIGHTS[tier]
        departments = list(dept_weights.keys())
        weights = [dept_weights[d] for d in departments]
        return self._rng.choices(departments, weights=weights, k=1)[0]

    def generate_ticket(
        self,
        department: Optional[DepartmentEnum] = None,
        tier: Optional[CustomerTierEnum] = None,
    ) -> Ticket:
        """Generate a single realistic ticket."""
        self._ticket_counter += 1

        if tier is None:
            tier = self._select_tier()

        if department is None:
            department = self._select_department(tier)

        templates = DEPARTMENT_TEMPLATES[department]
        template = self._rng.choice(templates)

        wait_time = int(self._rng.expovariate(1 / 180))
        wait_time = max(0, min(wait_time, 3600))

        ticket_id = f"TKT-{self._ticket_counter:04d}"

        return Ticket(
            ticket_id=ticket_id,
            subject=template["subject"],
            description=template["description"],
            sentiment=float(template["sentiment"]),
            customer_tier=tier,
            correct_department=department,
            correct_priority=template["priority"],
            wait_time_seconds=wait_time,
            created_at=time.time(),
        )

    def generate_episode(self, n_tickets: int) -> List[Ticket]:
        """Generate a batch of tickets for an episode."""
        return [self.generate_ticket() for _ in range(n_tickets)]
