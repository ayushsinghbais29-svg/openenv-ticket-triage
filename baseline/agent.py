"""
GPT-4 baseline agent for the OpenEnv Ticket Triage environment.
Uses OpenAI API with chain-of-thought reasoning.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import Action, ActionTypeEnum, Observation, TaskTypeEnum
from src.utils import format_ticket_for_prompt
from baseline.prompts import (
    CLASSIFICATION_PROMPT,
    EFFICIENCY_PROMPT,
    PRIORITY_CLASSIFICATION_PROMPT,
)


class GPT4Agent:
    """
    GPT-4 powered agent for ticket triage.
    Reads OPENAI_API_KEY from environment variables.
    Uses task-specific prompts with chain-of-thought reasoning.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self._client = None

    def _get_client(self):
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore

                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai>=1.0.0")
        return self._client

    def _build_prompt(self, observation: Observation) -> str:
        """Build a task-specific prompt for the given observation."""
        ticket_text = format_ticket_for_prompt(observation)
        task_type = TaskTypeEnum(observation.task_type)

        if task_type == TaskTypeEnum.CLASSIFICATION:
            return CLASSIFICATION_PROMPT.format(ticket_text=ticket_text)

        elif task_type == TaskTypeEnum.PRIORITY_CLASSIFICATION:
            return PRIORITY_CLASSIFICATION_PROMPT.format(
                ticket_text=ticket_text,
                customer_tier=observation.customer_tier,
                sentiment_score=f"{observation.sentiment_score:.2f}",
                wait_time=f"{observation.wait_time_minutes:.1f}",
            )

        else:  # efficiency
            return EFFICIENCY_PROMPT.format(
                ticket_text=ticket_text,
                customer_tier=observation.customer_tier,
                wait_time=f"{observation.wait_time_minutes:.1f}",
                tickets_remaining=observation.tickets_remaining,
            )

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Extract JSON from model response, handling markdown code blocks."""
        # Strip markdown code blocks if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: return a default classify action
        return {
            "action_type": "classify",
            "department": "general",
            "reasoning": "Fallback due to parse error",
            "confidence": 0.3,
        }

    def decide(self, observation: Observation) -> Action:
        """
        Decide on an action given an observation.

        Args:
            observation: Current ticket observation.

        Returns:
            Action to take.
        """
        prompt = self._build_prompt(observation)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful customer support triage agent. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            # Fallback action on API error
            return Action(
                action_type=ActionTypeEnum.CLASSIFY,
                department=None,
                reasoning=f"API error: {e}",
                confidence=0.0,
            )

        parsed = self._parse_response(raw)
        return self._dict_to_action(parsed)

    def _dict_to_action(self, data: Dict[str, Any]) -> Action:
        """Convert parsed dict to Action model."""
        action_type_str = data.get("action_type", "classify")
        # Map 'route' -> 'route', 'escalate' -> 'escalate', else 'classify'
        try:
            action_type = ActionTypeEnum(action_type_str)
        except ValueError:
            action_type = ActionTypeEnum.CLASSIFY

        dept_str = data.get("department")
        priority_str = data.get("priority")

        from src.models import DepartmentEnum, PriorityEnum

        dept = None
        if dept_str:
            try:
                dept = DepartmentEnum(dept_str.lower())
            except ValueError:
                dept = None

        priority = None
        if priority_str:
            try:
                priority = PriorityEnum(priority_str.lower())
            except ValueError:
                priority = None

        return Action(
            action_type=action_type,
            department=dept,
            priority=priority,
            reasoning=data.get("reasoning", ""),
            confidence=float(data.get("confidence", 1.0)),
        )
