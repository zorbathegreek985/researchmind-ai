from __future__ import annotations

import logging
import os

import dotenv
import google.generativeai as genai

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    pass


class LLMService:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise LLMServiceError("Missing GEMINI_API_KEY")

        genai.configure(api_key=self.api_key)

        self.model = genai.GenerativeModel("gemini-1.5-flash")

        logger.info("Gemini initialized successfully")

    def generate_response(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMServiceError("Prompt must be a non-empty string.")

        try:
            response = self.model.generate_content(prompt)

            if not response.text:
                raise LLMServiceError("Gemini returned an empty response.")

            return response.text.strip()

        except Exception as exc:
            logger.exception("Gemini request failed.")
            raise LLMServiceError(
                "Failed to generate response from Gemini."
            ) from exc