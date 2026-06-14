from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from utils.llm import LLMService, LLMServiceError, print_gemini_exception_diagnostics


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if match:
            cleaned = match.group(1)

    return json.loads(cleaned)


def _normalize_critiques(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        items = raw.get("critiques", [])
    else:
        items = raw

    if not isinstance(items, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(
                {
                    "hypothesis": item.get("hypothesis") or item.get("title") or "Hypothesis",
                    "critique": item.get("critique") or item.get("weakness") or "The proposal needs a clearer validation plan.",
                    "weaknesses": item.get("weaknesses") or ["Limited evidence base", "Unclear experimental controls"],
                    "improvements": item.get("improvements") or ["Add a baseline comparison", "Specify success metrics"],
                }
            )
        elif isinstance(item, str):
            normalized.append(
                {
                    "hypothesis": "Hypothesis under review",
                    "critique": item,
                    "weaknesses": ["Needs stronger justification"],
                    "improvements": ["Add a benchmark and clearer expected outcome"],
                }
            )

    return normalized


def _fallback_critiques(topic: str, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fallback = []
    for item in hypotheses[:3]:
        title = item.get("hypothesis", "This hypothesis") if isinstance(item, dict) else str(item)
        fallback.append(
            {
                "hypothesis": title,
                "critique": f"Estimated critique (fallback mode): The proposal for {topic} is promising, but it needs stronger baselines, clearer metrics, and a realistic evaluation plan.",
                "weaknesses": ["Potential over-claiming", "Limited measurement detail", "Possible bias in the evaluation setup"],
                "improvements": ["Define one primary metric", "Include a comparison baseline", "Add a pilot study before scaling"],
            }
        )
    return fallback


def critique_hypotheses(topic: str, hypotheses: Any, force_fallback: bool = False) -> List[Dict[str, Any]]:
    """Critique generated hypotheses and identify weaknesses."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("A non-empty research topic is required.")

    if isinstance(hypotheses, str):
        hypothesis_text = hypotheses
    else:
        hypothesis_text = json.dumps(hypotheses, ensure_ascii=False)

    if not force_fallback:
        try:
            llm = LLMService()
            prompt = f"""
You are a peer reviewer for an AI research proposal.

Topic: {topic}

Hypotheses to review:
{hypothesis_text}

Return valid JSON only with a top-level key named 'critiques'.
Each critique should contain:
- hypothesis
- critique
- weaknesses
- improvements
"""
            response_text = llm.generate_response(prompt, stage="critique")
            data = _extract_json(response_text)
            normalized = _normalize_critiques(data)
            if normalized:
                return normalized
        except LLMServiceError as e:
            print_gemini_exception_diagnostics(e, stage="critique", service_error=e)

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            print(f"Critique parsing failed: {repr(e)}")

        except Exception as e:
            print_gemini_exception_diagnostics(e, stage="critique")

    return _fallback_critiques(topic.strip(), hypotheses if isinstance(hypotheses, list) else [])
