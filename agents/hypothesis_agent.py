from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from utils.llm import LLMService, LLMServiceError, print_gemini_exception_diagnostics
from utils.prompts import hypothesis_generation_prompt


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if match:
            cleaned = match.group(1)

    return json.loads(cleaned)


def _normalize_hypotheses(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        raw_items = raw.get("hypotheses", [])
    else:
        raw_items = raw

    if not isinstance(raw_items, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, str):
            normalized.append(
                {
                    "hypothesis": item,
                    "rationale": "This hypothesis directly addresses a documented research gap.",
                    "novelty": "Moderate novelty; it combines an existing idea with a new validation angle.",
                    "expected_impact": "A practical contribution to the current literature and future study design.",
                }
            )
            continue

        if isinstance(item, dict):
            normalized.append(
                {
                    "hypothesis": item.get("hypothesis") or item.get("title") or "Untitled hypothesis",
                    "rationale": item.get("rationale") or "This idea targets the most important unresolved question in the topic.",
                    "novelty": item.get("novelty") or "High novelty with a clear experimental path.",
                    "expected_impact": item.get("expected_impact") or "Expected to improve understanding and guide future experiments.",
                }
            )

    return normalized


def _fallback_hypotheses(topic: str, gaps: str) -> List[Dict[str, Any]]:
    gap_summary = gaps if isinstance(gaps, str) and gaps.strip() else "No detailed gap analysis was available."

    return [
        {
            "hypothesis": f"Estimated hypothesis (fallback mode): A fairness- and interpretability-aware {topic} workflow will outperform a baseline approach in real-world evaluation.",
            "rationale": f"This hypothesis directly addresses the gap that current work often lacks transparent validation and robust evidence in {topic}.",
            "novelty": "High novelty because it combines methodological rigor with practical evaluation criteria.",
            "expected_impact": "It could improve trust, transferability, and evidence quality for future studies.",
        },
        {
            "hypothesis": f"Estimated hypothesis (fallback mode): A lightweight multi-agent review loop will reduce over-claiming and improve the quality of research plans for {topic}.",
            "rationale": gap_summary,
            "novelty": "Moderate novelty through a focused critique-and-revision cycle.",
            "expected_impact": "This can make research planning more reliable and easier to reproduce.",
        },
        {
            "hypothesis": f"Estimated hypothesis (fallback mode): Benchmarking {topic} systems across multiple domains will reveal which factors drive generalizability and failure modes.",
            "rationale": "This targets the common weakness of narrow, single-domain evaluation.",
            "novelty": "Moderate novelty through cross-domain comparison rather than single-task validation.",
            "expected_impact": "It can help future teams choose more robust experimental protocols.",
        },
    ]


def generate_hypotheses(topic: str, gaps: str, force_fallback: bool = False) -> List[Dict[str, Any]]:
    """Generate research hypotheses from a topic and gap analysis summary."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("A non-empty research topic is required.")

    if not isinstance(gaps, str):
        gaps = json.dumps(gaps, ensure_ascii=False)

    if not force_fallback:
        try:
            llm = LLMService()
            prompt = hypothesis_generation_prompt(topic.strip(), gaps)
            response_text = llm.generate_response(prompt, stage="hypothesis")
            data = _extract_json(response_text)
            normalized = _normalize_hypotheses(data)
            if normalized:
                return normalized
        except LLMServiceError as e:
            print_gemini_exception_diagnostics(e, stage="hypothesis", service_error=e)

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            print(f"Hypothesis parsing failed: {repr(e)}")

        except Exception as e:
            print_gemini_exception_diagnostics(e, stage="hypothesis")

    return _fallback_hypotheses(topic.strip(), gaps)
