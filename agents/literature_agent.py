"""Literature review agent for ResearchMind AI.

This module uses the shared LLMService and prompt templates to generate a
structured literature review for a given research topic.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from utils.llm import LLMService, LLMServiceError
from utils.prompts import literature_review_prompt


class LiteratureAgentError(Exception):
    """Raised when the literature review agent cannot complete its task."""


class LiteratureAgent:
    """Generate a structured literature review for a research topic."""

    def __init__(self, llm_service: Optional[LLMService] = None) -> None:
        """Initialize the agent with an LLM service.

        Args:
            llm_service: Optional shared LLMService instance. If not provided,
                a default LLMService is created.
        """
        self.llm_service = llm_service or LLMService()

    def review(self, topic: str) -> Dict[str, Any]:
        """Generate a structured literature review for the given topic.

        Args:
            topic: The research topic or question to analyze.

        Returns:
            A dictionary containing the structured literature review JSON.

        Raises:
            LiteratureAgentError: If the topic is invalid or the response cannot
                be parsed into JSON.
        """
        if not isinstance(topic, str) or not topic.strip():
            raise LiteratureAgentError("A non-empty research topic is required.")

        try:
            prompt = literature_review_prompt(topic.strip())
            response_text = self.llm_service.generate_response(prompt)
            data = json.loads(response_text)
        except LLMServiceError as exc:
            raise LiteratureAgentError("Failed to generate literature review using Azure OpenAI.") from exc
        except json.JSONDecodeError as exc:
            raise LiteratureAgentError("The model response was not valid JSON.") from exc

        if not isinstance(data, dict):
            raise LiteratureAgentError("The literature review response must be a JSON object.")

        return data


def create_literature_agent(llm_service: Optional[LLMService] = None) -> LiteratureAgent:
    """Create and return a LiteratureAgent instance."""
    return LiteratureAgent(llm_service=llm_service)
