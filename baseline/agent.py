"""GPT-4 baseline agent for the OpenEnv Ticket Triage environment."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional

from .prompts import (
    SYSTEM_PROMPT,
    build_classification_prompt,
    build_efficiency_prompt,
    build_priority_classification_prompt,
)


class GPT4Agent:
    """
    GPT-4 based agent for ticket triage.

    Uses few-shot chain-of-thought prompting to route tickets.
    Reads OPENAI_API_KEY from environment variables.
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.1
    MAX_TOKENS = 256

    def __init__(
        self,
        model: Optional[str] = None,
        seed: int = 42,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the agent.

        Args:
            model: OpenAI model name (default: gpt-4o-mini)
            seed: Random seed for reproducibility
            api_key: OpenAI API key (default: reads OPENAI_API_KEY env var)
        """
        self.model = model or self.DEFAULT_MODEL
        self.seed = seed
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None
        self._classify_done: Dict[str, bool] = {}

    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "openai package is required for the baseline agent. "
                    "Install with: pip install openai"
                )
        return self._client

    def decide(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide on an action given the current observation.

        Args:
            obs: Observation dict from the environment

        Returns:
            Action dict with action_type, department, priority, confidence
        """
        task_type = obs.get("task_type", "classification")

        if task_type == "classification":
            prompt = build_classification_prompt(obs)
        elif task_type == "priority_classification":
            ticket_id = obs.get("ticket_id", "")
            classify_done = self._classify_done.get(ticket_id, False)
            prompt = build_priority_classification_prompt(obs, classify_done)
        elif task_type == "efficiency_triage":
            prompt = build_efficiency_prompt(obs)
        else:
            prompt = build_classification_prompt(obs)

        action = self._call_llm(prompt)

        if task_type == "priority_classification":
            ticket_id = obs.get("ticket_id", "")
            action_type = action.get("action_type", "")
            if action_type == "classify":
                self._classify_done[ticket_id] = True

        return action

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call the LLM API and parse the JSON response."""
        if not self._api_key:
            return self._heuristic_fallback({})

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.TEMPERATURE,
                max_tokens=self.MAX_TOKENS,
                seed=self.seed,
            )
            content = response.choices[0].message.content or ""
            return self._parse_json_response(content)
        except Exception:
            return self._heuristic_fallback({})

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with fallback."""
        content = content.strip()
        json_match = re.search(r"\{[^{}]+\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return self._heuristic_fallback({"content": content})

    def _heuristic_fallback(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based fallback when LLM is unavailable."""
        subject = str(obs.get("subject", "")).lower()
        description = str(obs.get("description", "")).lower()
        tier = str(obs.get("customer_tier", "free")).lower()
        text = subject + " " + description

        if tier == "enterprise" and any(
            kw in text for kw in ["sla", "compliance", "dedicated", "qbr", "review"]
        ):
            department = "Premium Support"
        elif any(
            kw in text
            for kw in ["api", "error", "500", "webhook", "sdk", "integration", "bug", "export"]
        ):
            department = "Technical"
        elif any(
            kw in text
            for kw in ["charge", "invoice", "billing", "refund", "payment", "fee", "plan"]
        ):
            department = "Billing"
        else:
            department = "General"

        return {
            "action_type": "route",
            "department": department,
            "confidence": 0.7,
            "reasoning": "heuristic fallback",
        }

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self._classify_done = {}
