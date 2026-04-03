"""
Realistic ticket templates and TicketGenerator for the OpenEnv Ticket Triage environment.
Generates tickets across Billing, Technical, General, and Premium Support domains.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .models import CustomerTierEnum, DepartmentEnum, PriorityEnum


# ---------------------------------------------------------------------------
# Ticket Templates
# ---------------------------------------------------------------------------

BILLING_TEMPLATES: List[Dict[str, Any]] = [
    {
        "subject": "Unexpected charge on my account",
        "description": (
            "I was charged {amount} on {date} but I cancelled my subscription last month. "
            "Please refund this amount immediately. I have proof of cancellation in my email."
        ),
        "correct_department": DepartmentEnum.BILLING,
        "correct_priority": PriorityEnum.HIGH,
        "sentiment_base": -0.7,
    },
    {
        "subject": "Invoice discrepancy - overcharged",
        "description": (
            "My invoice #{invoice_num} shows a charge of {amount} but according to my plan "
            "I should only be paying {expected_amount}. Please investigate and correct this."
        ),
        "correct_department": DepartmentEnum.BILLING,
        "correct_priority": PriorityEnum.MEDIUM,
        "sentiment_base": -0.4,
    },
    {
        "subject": "Update payment method",
        "description": (
            "I need to update my credit card on file. My current card ending in {card_last4} "
            "is expiring soon and I want to add a new card before the next billing cycle."
        ),
        "correct_department": DepartmentEnum.BILLING,
        "correct_priority": PriorityEnum.LOW,
        "sentiment_base": 0.1,
    },
    {
        "subject": "Request annual plan upgrade",
        "description": (
            "I would like to upgrade from monthly to annual billing to save costs. "
            "Can you confirm the annual rate and apply any proration for the current month?"
        ),
        "correct_department": DepartmentEnum.BILLING,
        "correct_priority": PriorityEnum.LOW,
        "sentiment_base": 0.4,
    },
    {
        "subject": "Double charged this month",
        "description": (
            "I see two identical charges of {amount} on my credit card statement dated {date}. "
            "This is clearly an error. I need this resolved and a full refund for the duplicate charge."
        ),
        "correct_department": DepartmentEnum.BILLING,
        "correct_priority": PriorityEnum.HIGH,
        "sentiment_base": -0.8,
    },
    {
        "subject": "Tax exemption certificate submission",
        "description": (
            "Our organization is tax-exempt and I would like to submit our exemption certificate "
            "to ensure future invoices do not include sales tax. Please advise on the process."
        ),
        "correct_department": DepartmentEnum.BILLING,
        "correct_priority": PriorityEnum.MEDIUM,
        "sentiment_base": 0.1,
    },
]

TECHNICAL_TEMPLATES: List[Dict[str, Any]] = [
    {
        "subject": "API returning 500 error on authentication endpoint",
        "description": (
            "Since yesterday evening, our integration with your API has been failing with HTTP 500 "
            "errors on the /auth/token endpoint. Error: {error_msg}. This is blocking our entire "
            "production deployment. We need an urgent fix."
        ),
        "correct_department": DepartmentEnum.TECHNICAL,
        "correct_priority": PriorityEnum.CRITICAL,
        "sentiment_base": -0.9,
    },
    {
        "subject": "Slow performance on dashboard",
        "description": (
            "The dashboard has been loading very slowly for the past {days} days. "
            "Page load times are averaging {load_time} seconds instead of the usual 2 seconds. "
            "This is affecting productivity for our entire team."
        ),
        "correct_department": DepartmentEnum.TECHNICAL,
        "correct_priority": PriorityEnum.MEDIUM,
        "sentiment_base": -0.5,
    },
    {
        "subject": "Integration setup help needed",
        "description": (
            "I'm trying to set up the Zapier integration following your documentation but keep "
            "getting an authentication error at step 3. I've double-checked my API key and it "
            "appears valid. Can someone walk me through the setup?"
        ),
        "correct_department": DepartmentEnum.TECHNICAL,
        "correct_priority": PriorityEnum.LOW,
        "sentiment_base": 0.0,
    },
    {
        "subject": "Data export not working",
        "description": (
            "I've been trying to export our data using the bulk export feature for the last "
            "{hours} hours but the download always fails. I get an error saying 'Export failed: "
            "timeout'. We have a compliance deadline in 24 hours."
        ),
        "correct_department": DepartmentEnum.TECHNICAL,
        "correct_priority": PriorityEnum.HIGH,
        "sentiment_base": -0.7,
    },
    {
        "subject": "Mobile app crashing on iOS 17",
        "description": (
            "After updating to iOS 17, your mobile app crashes immediately on launch. "
            "I've tried reinstalling but same issue. Several colleagues have the same problem. "
            "App version: {app_version}, iPhone model: {device}."
        ),
        "correct_department": DepartmentEnum.TECHNICAL,
        "correct_priority": PriorityEnum.HIGH,
        "sentiment_base": -0.6,
    },
    {
        "subject": "Webhook not delivering events",
        "description": (
            "Our webhook endpoint is no longer receiving events. The last successful delivery "
            "was {date}. I've verified our endpoint is up and returning 200. Please check your "
            "delivery logs for our endpoint: {endpoint_url}."
        ),
        "correct_department": DepartmentEnum.TECHNICAL,
        "correct_priority": PriorityEnum.MEDIUM,
        "sentiment_base": -0.3,
    },
]

GENERAL_TEMPLATES: List[Dict[str, Any]] = [
    {
        "subject": "How do I add team members?",
        "description": (
            "I want to invite {num_users} new team members to our workspace. "
            "Can you explain the process and let me know if there are any additional costs "
            "for adding users to our current plan?"
        ),
        "correct_department": DepartmentEnum.GENERAL,
        "correct_priority": PriorityEnum.LOW,
        "sentiment_base": 0.3,
    },
    {
        "subject": "Feature request: dark mode",
        "description": (
            "I would love to see a dark mode option added to the platform. "
            "Many of us work late hours and the bright interface causes eye strain. "
            "Is this on your roadmap? Many people in our company would benefit from this."
        ),
        "correct_department": DepartmentEnum.GENERAL,
        "correct_priority": PriorityEnum.LOW,
        "sentiment_base": 0.5,
    },
    {
        "subject": "Account password reset not working",
        "description": (
            "I requested a password reset {hours} hours ago but haven't received the email. "
            "I've checked my spam folder. Please help me regain access to my account. "
            "My account email is {email}."
        ),
        "correct_department": DepartmentEnum.GENERAL,
        "correct_priority": PriorityEnum.MEDIUM,
        "sentiment_base": -0.3,
    },
    {
        "subject": "Question about data retention policy",
        "description": (
            "I need to understand your data retention policy for compliance purposes. "
            "Specifically, how long do you retain user activity logs and audit trails? "
            "We're preparing for a SOC2 audit next month."
        ),
        "correct_department": DepartmentEnum.GENERAL,
        "correct_priority": PriorityEnum.MEDIUM,
        "sentiment_base": 0.1,
    },
    {
        "subject": "How to export user data for GDPR",
        "description": (
            "One of our users has submitted a GDPR data subject access request. "
            "I need to export all data associated with their account. "
            "Can you guide me through the process? We have {days} days to respond legally."
        ),
        "correct_department": DepartmentEnum.GENERAL,
        "correct_priority": PriorityEnum.HIGH,
        "sentiment_base": 0.0,
    },
]

PREMIUM_SUPPORT_TEMPLATES: List[Dict[str, Any]] = [
    {
        "subject": "URGENT: Production outage affecting all users",
        "description": (
            "We are experiencing a complete production outage. None of our {num_users} users "
            "can access the platform. Started approximately {time_ago} ago. "
            "This is severely impacting our business operations and SLA commitments to our clients. "
            "We need immediate assistance from your engineering team."
        ),
        "correct_department": DepartmentEnum.PREMIUM_SUPPORT,
        "correct_priority": PriorityEnum.CRITICAL,
        "sentiment_base": -1.0,
    },
    {
        "subject": "Custom SLA review meeting request",
        "description": (
            "We're approaching our contract renewal and would like to schedule a meeting "
            "to review our current SLA terms and discuss enterprise-level options. "
            "Our account manager {manager_name} suggested reaching out to support. "
            "We have {num_users} users and are considering significant expansion."
        ),
        "correct_department": DepartmentEnum.PREMIUM_SUPPORT,
        "correct_priority": PriorityEnum.MEDIUM,
        "sentiment_base": 0.4,
    },
    {
        "subject": "Dedicated instance configuration",
        "description": (
            "As an enterprise customer, we require a dedicated instance with custom data "
            "residency in {region}. Please initiate the provisioning process and provide "
            "an estimated timeline. Our security team will need to review the architecture "
            "before go-live."
        ),
        "correct_department": DepartmentEnum.PREMIUM_SUPPORT,
        "correct_priority": PriorityEnum.HIGH,
        "sentiment_base": 0.1,
    },
    {
        "subject": "Security audit report request",
        "description": (
            "For our annual compliance review, we need your latest SOC2 Type II report, "
            "penetration testing results, and security whitepaper. "
            "Our compliance deadline is {deadline}. Please share these under NDA."
        ),
        "correct_department": DepartmentEnum.PREMIUM_SUPPORT,
        "correct_priority": PriorityEnum.HIGH,
        "sentiment_base": 0.1,
    },
]

ALL_TEMPLATES = (
    BILLING_TEMPLATES + TECHNICAL_TEMPLATES + GENERAL_TEMPLATES + PREMIUM_SUPPORT_TEMPLATES
)

# Template variable fillers
TEMPLATE_VARS: Dict[str, List[str]] = {
    "{amount}": ["$49.99", "$99.00", "$199.00", "$299.00", "$499.00"],
    "{date}": ["March 1st", "March 15th", "February 28th", "April 1st"],
    "{invoice_num}": ["INV-2024-001", "INV-2024-042", "INV-2024-117"],
    "{expected_amount}": ["$29.99", "$49.99", "$79.00"],
    "{card_last4}": ["4242", "1234", "5678", "9999"],
    "{error_msg}": [
        "Internal Server Error",
        "Token generation failed",
        "Authentication service unavailable",
    ],
    "{days}": ["2", "3", "5", "7"],
    "{load_time}": ["8", "12", "15", "20"],
    "{hours}": ["2", "4", "6", "12", "24"],
    "{app_version}": ["3.2.1", "3.2.0", "3.1.5"],
    "{device}": ["iPhone 14 Pro", "iPhone 13", "iPhone 15"],
    "{endpoint_url}": ["https://hooks.example.com/webhook", "https://api.myapp.com/events"],
    "{num_users}": ["15", "50", "200", "500", "1000"],
    "{email}": ["user@company.com", "admin@business.org"],
    "{time_ago}": ["30 minutes", "1 hour", "2 hours"],
    "{manager_name}": ["Sarah", "Mike", "Jennifer", "David"],
    "{region}": ["EU (Frankfurt)", "US-West", "Asia-Pacific (Singapore)"],
    "{deadline}": ["April 30th", "May 15th", "June 1st"],
}


def _fill_template(text: str) -> str:
    """Replace template placeholders with random values."""
    for placeholder, options in TEMPLATE_VARS.items():
        if placeholder in text:
            text = text.replace(placeholder, random.choice(options))
    return text


# ---------------------------------------------------------------------------
# TicketGenerator
# ---------------------------------------------------------------------------


class TicketGenerator:
    """
    Generates realistic customer support tickets with metadata.
    Supports controlled generation for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._ticket_counter = 0

    def _next_id(self) -> str:
        self._ticket_counter += 1
        return f"TKT-{self._ticket_counter:05d}"

    def _random_tier(self) -> CustomerTierEnum:
        """Weighted random tier: 60% free, 30% premium, 10% enterprise."""
        roll = self._rng.random()
        if roll < 0.60:
            return CustomerTierEnum.FREE
        elif roll < 0.90:
            return CustomerTierEnum.PREMIUM
        return CustomerTierEnum.ENTERPRISE

    def _sentiment_jitter(self, base: float) -> float:
        """Add small random jitter to base sentiment."""
        jitter = self._rng.uniform(-0.15, 0.15)
        return max(-1.0, min(1.0, base + jitter))

    def generate(
        self,
        department_filter: Optional[DepartmentEnum] = None,
        num_tickets: int = 1,
        wait_time_base: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Generate a list of ticket dictionaries.

        Args:
            department_filter: If provided, only generate tickets for this department.
            num_tickets: Number of tickets to generate.
            wait_time_base: Base wait time (minutes) added to each ticket.

        Returns:
            List of ticket dicts with all relevant fields.
        """
        pool = ALL_TEMPLATES
        if department_filter is not None:
            pool = [t for t in pool if t["correct_department"] == department_filter]
            if not pool:
                pool = ALL_TEMPLATES

        tickets = []
        for i in range(num_tickets):
            template = self._rng.choice(pool)
            subject = _fill_template(template["subject"])
            description = _fill_template(template["description"])
            tier = self._random_tier()
            wait_time = wait_time_base + (i * self._rng.uniform(5.0, 30.0))

            ticket = {
                "ticket_id": self._next_id(),
                "subject": subject,
                "description": description,
                "customer_tier": tier,
                "sentiment_score": self._sentiment_jitter(template["sentiment_base"]),
                "wait_time_minutes": round(wait_time, 1),
                "correct_department": template["correct_department"],
                "correct_priority": template["correct_priority"],
                "metadata": {
                    "template_department": template["correct_department"].value,
                    "template_priority": template["correct_priority"].value,
                },
            }
            tickets.append(ticket)

        return tickets

    def generate_episode_tickets(
        self, task_type: str, num_tickets: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate tickets for a full episode, with progressive wait times."""
        return self.generate(
            num_tickets=num_tickets,
            wait_time_base=self._rng.uniform(0.0, 10.0),
        )
