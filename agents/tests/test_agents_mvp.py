import pytest

from agents.critic_agent import critique_hypotheses
from agents.gap_agent import find_research_gaps
from agents.hypothesis_agent import generate_hypotheses
from utils.llm import (
    GEMINI_QUOTA_EXCEEDED_MESSAGE,
    LLMService,
    LLMServiceError,
    format_gemini_error,
    get_gemini_call_counts,
    record_gemini_call,
    reset_gemini_call_counts,
)
from utils.summarizer import summarize_paper


def test_generate_hypotheses_fallback_returns_structured_output():
    result = generate_hypotheses(
        topic="AI for healthcare",
        gaps="Limited evidence on fairness and deployment in clinical settings.",
        force_fallback=True,
    )

    assert isinstance(result, list)
    assert len(result) >= 1
    assert all("hypothesis" in item for item in result)


def test_critique_hypotheses_fallback_returns_structured_output():
    result = critique_hypotheses(
        topic="AI for healthcare",
        hypotheses=[
            {
                "hypothesis": "A fairness-aware AI triage model will improve clinician trust.",
                "rationale": "It addresses current deployment gaps.",
            }
        ],
        force_fallback=True,
    )

    assert isinstance(result, list)
    assert len(result) >= 1
    assert all("critique" in item for item in result)


def test_gemini_call_tracking_records_each_stage():
    reset_gemini_call_counts()

    record_gemini_call("summarization")
    record_gemini_call("gap_analysis")
    record_gemini_call("hypothesis")
    record_gemini_call("critique")

    counts = get_gemini_call_counts()

    assert counts["summarization"] == 1
    assert counts["gap_analysis"] == 1
    assert counts["hypothesis"] == 1
    assert counts["critique"] == 1
    assert counts["total"] == 4


def test_format_gemini_error_returns_user_friendly_message():
    message = format_gemini_error(RuntimeError("429 quota exceeded for model gemini-2.5-flash"))

    assert message == GEMINI_QUOTA_EXCEEDED_MESSAGE


def test_llm_service_wraps_connect_timeout_as_retryable_error(monkeypatch):
    reset_gemini_call_counts()

    class ConnectTimeout(Exception):
        pass

    service = LLMService.__new__(LLMService)
    service.model_name = "gemini-test"
    service.max_retries = 1
    service.retry_backoff = 0

    attempts = {"count": 0}
    service.model_names = ["gemini-test"]

    def broken_request(prompt, model_name):
        attempts["count"] += 1
        raise ConnectTimeout("connect timed out")

    monkeypatch.setattr(service, "_request_once", broken_request)
    monkeypatch.setattr(service, "_sleep_before_retry", lambda attempt, error: None)

    with pytest.raises(LLMServiceError) as exc_info:
        service.generate_response("Summarize this paper.")

    assert attempts["count"] == 2
    assert exc_info.value.code == "GEMINI_TIMEOUT"
    assert exc_info.value.retryable is True


def test_llm_service_switches_to_fallback_model_after_retryable_failure(monkeypatch):
    reset_gemini_call_counts()

    class ServerUnavailable(Exception):
        pass

    service = LLMService.__new__(LLMService)
    service.model_name = "gemini-primary"
    service.model_names = ["gemini-primary", "gemini-fallback"]
    service.max_retries = 1
    service.retry_backoff = 0

    calls = []

    def request_with_fallback(prompt, model_name):
        calls.append(model_name)
        if model_name == "gemini-primary":
            raise ServerUnavailable("503 UNAVAILABLE")
        return "fallback response"

    monkeypatch.setattr(service, "_request_once", request_with_fallback)
    monkeypatch.setattr(service, "_sleep_before_retry", lambda attempt, error: None)

    result = service.generate_response("Summarize this paper.")

    assert result == "fallback response"
    assert calls == ["gemini-primary", "gemini-primary", "gemini-fallback"]


def test_llm_service_does_not_retry_or_wait_on_gemini_quota(monkeypatch):
    reset_gemini_call_counts()

    service = LLMService.__new__(LLMService)
    service.model_name = "gemini-primary"
    service.model_names = ["gemini-primary", "gemini-fallback"]
    service.max_retries = 3
    service.retry_backoff = 100

    calls = []
    sleeps = []

    def quota_exhausted(prompt, model_name):
        calls.append(model_name)
        raise RuntimeError("429 RESOURCE_EXHAUSTED. Retry in 30s")

    monkeypatch.setattr(service, "_request_once", quota_exhausted)
    monkeypatch.setattr(service, "_sleep_before_retry", lambda attempt, error: sleeps.append((attempt, error)))

    with pytest.raises(LLMServiceError) as exc_info:
        service.generate_response("Summarize this paper.")

    assert calls == ["gemini-primary"]
    assert sleeps == []
    assert exc_info.value.message == GEMINI_QUOTA_EXCEEDED_MESSAGE
    assert exc_info.value.retryable is False
    assert exc_info.value.retry_after is None

    calls.clear()
    with pytest.raises(LLMServiceError):
        service.generate_response("Try another stage.")

    assert calls == []


def test_find_research_gaps_fallback_returns_estimated_gaps(monkeypatch):
    class BrokenLLM:
        def __init__(self, *args, **kwargs):
            raise LLMServiceError("timeout")

    monkeypatch.setattr("agents.gap_agent.LLMService", BrokenLLM)

    result = find_research_gaps("Paper A discusses fairness and evaluation in AI literature review. Paper B studies explainability and domain adaptation.")

    assert "Estimated gaps (fallback mode)" in result
    assert "1." in result and "2." in result and "3." in result


def test_summarize_paper_fallback_uses_estimated_summary(monkeypatch):
    class BrokenLLM:
        def __init__(self, *args, **kwargs):
            raise LLMServiceError("quota exceeded")

    monkeypatch.setattr("utils.summarizer.LLMService", BrokenLLM)

    result = summarize_paper("AI literature review", "This paper studies bias, retrieval quality, and explainability in automated review systems.")

    assert "Estimated summary (fallback mode)" in result
    assert "bias" in result.lower() or "review" in result.lower()
