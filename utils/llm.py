from __future__ import annotations

import logging
import os
import re
import time
from collections import Counter

import dotenv

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

_GEMINI_CALL_COUNTS = {
    "summarization": 0,
    "gap_analysis": 0,
    "hypothesis": 0,
    "critique": 0,
}

_GEMINI_RUNTIME_STATUS = {
    "last_model": None,
    "fallback_model_used": False,
    "request_failed": False,
}


def reset_gemini_call_counts() -> dict:
    """Reset Gemini request counters for a new workflow run."""
    global _GEMINI_CALL_COUNTS
    _GEMINI_CALL_COUNTS = {
        "summarization": 0,
        "gap_analysis": 0,
        "hypothesis": 0,
        "critique": 0,
    }
    reset_gemini_runtime_status()
    return _GEMINI_CALL_COUNTS


def record_gemini_call(stage: str) -> dict:
    """Count a Gemini call for one workflow stage."""
    stage_key = stage.lower().replace(" ", "_")
    if stage_key in _GEMINI_CALL_COUNTS:
        _GEMINI_CALL_COUNTS[stage_key] += 1
        logger.info("Gemini request counted", extra={"stage": stage_key, "counts": get_gemini_call_counts()})
    return get_gemini_call_counts()


def get_gemini_call_counts() -> dict:
    """Return current call counts and total for this workflow run."""
    total = sum(_GEMINI_CALL_COUNTS.values())
    return {**_GEMINI_CALL_COUNTS, "total": total}


def reset_gemini_runtime_status() -> dict:
    """Reset Gemini model health indicators for a new workflow run."""
    global _GEMINI_RUNTIME_STATUS
    _GEMINI_RUNTIME_STATUS = {
        "last_model": None,
        "fallback_model_used": False,
        "request_failed": False,
    }
    return _GEMINI_RUNTIME_STATUS


def record_gemini_model_success(model_name: str, primary_model: str) -> dict:
    """Record which Gemini model successfully served a response."""
    _GEMINI_RUNTIME_STATUS["last_model"] = model_name
    if model_name != primary_model:
        _GEMINI_RUNTIME_STATUS["fallback_model_used"] = True
    return get_gemini_runtime_status()


def record_gemini_model_failure() -> dict:
    """Record that at least one Gemini request failed during this workflow."""
    _GEMINI_RUNTIME_STATUS["request_failed"] = True
    return get_gemini_runtime_status()


def get_gemini_runtime_status() -> dict:
    """Return current Gemini model health indicators."""
    return dict(_GEMINI_RUNTIME_STATUS)


def get_gemini_diagnostic() -> dict:
    """Return a non-secret startup diagnostic for Gemini configuration."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    print("Gemini API loaded:", bool(api_key))

    return {
        "model_name": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "api_key_loaded": bool(api_key),
        "api_key_suffix": api_key[-6:] if isinstance(api_key, str) and len(api_key) >= 6 else None,
    }


def format_gemini_error(exc: Exception) -> str:
    """Translate raw Gemini failures into a clean, user-friendly message."""
    text = str(exc).strip()
    lowered = text.lower()

    if "quota" in lowered or "resource_exhausted" in lowered or "rate limit" in lowered or "429" in text:
        return "Gemini quota or rate limits were reached. Please wait a moment and try again."

    if "timeout" in lowered or "timed out" in lowered or "deadline" in lowered or "408" in text:
        return "The Gemini request timed out. Please try again with a simpler prompt or a short pause."

    if "missing gemini_api_key" in lowered or "google_api_key" in lowered or "api key" in lowered:
        return "Gemini credentials are not available. Add GEMINI_API_KEY or GOOGLE_API_KEY and restart the app."

    if "package is not installed" in lowered or "google-genai" in lowered:
        return "The Gemini client library is not installed. Install the required package and restart the app."

    if "empty response" in lowered:
        return "Gemini returned no usable content. Please try the run again."

    if "service could not be reached" in lowered or "temporarily unavailable" in lowered or "unavailable" in lowered:
        return "The Gemini service is temporarily unavailable. Please try again shortly."

    return "The Gemini service is currently unavailable. Please wait a moment and try again."


def _get_env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Invalid integer env value; using default", extra={"name": name, "default": default})
        return default
    return max(minimum, parsed)


def _get_env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("Invalid float env value; using default", extra={"name": name, "default": default})
        return default
    return max(minimum, parsed)


def _parse_model_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _dedupe_models(models: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for model in models:
        if model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped


def build_fallback_summary(title: str, abstract: str) -> str:
    """Generate a demo-friendly summary when Gemini is unavailable."""
    clean_title = title.strip() or "this paper"
    clean_abstract = abstract.strip() or ""
    keywords = [token for token, _ in Counter(re.findall(r"[A-Za-z]{4,}", clean_abstract.lower())).most_common(3)]

    if not keywords:
        keywords = ["evaluation", "interpretability", "generalization"]

    keyword_text = ", ".join(keywords)
    return (
        "Estimated summary (fallback mode): "
        f"{clean_title} focuses on {keyword_text} and presents practical guidance for evaluating research quality, "
        "bias, and deployment trade-offs in a demo-ready workflow. This version is intentionally stable for presentations "
        "when the Gemini service is temporarily unavailable."
    )


def build_fallback_gaps(summaries: str) -> str:
    """Generate three realistic research gaps from the paper summaries."""
    lowered = summaries.lower()
    details = []

    if any(word in lowered for word in ("fair", "bias", "equity", "ethic")):
        details.append("Fairness, bias mitigation, and accountability are not consistently evaluated across the reviewed studies.")
    else:
        details.append("Evaluation quality and reproducibility are not consistently benchmarked across the reviewed studies.")

    if any(word in lowered for word in ("explain", "interpret", "trust", "transparency")):
        details.append("Interpretability and user trust signals remain under-explored in real-world deployment settings.")
    else:
        details.append("Model explainability and decision transparency are under-specified for practical adoption.")

    if any(word in lowered for word in ("domain", "generaliz", "adapt", "transfer")):
        details.append("Cross-domain transfer and robustness are insufficiently validated beyond narrow benchmark conditions.")
    else:
        details.append("Cross-domain validation and long-term robustness are not yet established in the current literature.")

    return (
        "Estimated gaps (fallback mode):\n"
        "1. " + details[0] + "\n"
        "2. " + details[1] + "\n"
        "3. " + details[2]
    )


class LLMServiceError(Exception):
    """Structured Gemini service error with retry metadata."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "GEMINI_REQUEST_FAILED",
        status_code: int | None = None,
        retryable: bool = False,
        retry_after: int | None = None,
        raw_error: Exception | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.raw_error = raw_error
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "ok": False,
            "error": {
                "message": self.message,
                "code": self.code,
                "status_code": self.status_code,
                "retryable": self.retryable,
                "retry_after": self.retry_after,
            },
        }


class LLMService:
    """Central Gemini access wrapper with retries, timeout handling, and logging."""

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise LLMServiceError("Missing GEMINI_API_KEY or GOOGLE_API_KEY", code="MISSING_API_KEY")

        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        fallback_models = _parse_model_list(os.getenv("GEMINI_FALLBACK_MODELS"))
        if not fallback_models:
            fallback_models = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
        self.model_names = _dedupe_models([self.model_name, *fallback_models])
        self.timeout_seconds = _get_env_int("GEMINI_TIMEOUT_SECONDS", 45)
        self.timeout_ms = self.timeout_seconds * 1000
        self.max_retries = _get_env_int("GEMINI_MAX_RETRIES", 3, minimum=0)
        self.retry_backoff = _get_env_float("GEMINI_RETRY_BACKOFF_SECONDS", 2.0)
        self.client = None
        self.model = None

        try:
            from google import genai as google_genai
            from google.genai import types as genai_types

            self.genai_types = genai_types
            self.client = google_genai.Client(
                api_key=self.api_key,
                http_options=genai_types.HttpOptions(timeout=self.timeout_ms),
            )

            logger.info(
                "Gemini client initialized successfully",
                extra={"models": self.model_names, "timeout_seconds": self.timeout_seconds},
            )
        except ImportError:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self.legacy_genai = genai
                logger.info("Gemini legacy client initialized successfully", extra={"models": self.model_names})
            except ImportError as exc:
                raise LLMServiceError(
                    "Gemini client package is not installed. Install google-genai or google-generativeai.",
                    code="MISSING_GEMINI_PACKAGE",
                    raw_error=exc,
                ) from exc

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        text = str(exc)
        match = re.search(r"\b(4\d{2}|5\d{2})\b", text)
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_retry_after(exc: Exception) -> int | None:
        text = str(exc).lower()
        retry_match = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", text)
        if retry_match:
            return max(1, int(float(retry_match.group(1))))
        return None

    @staticmethod
    def _is_quota_or_retryable(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(token in text for token in ("429", "quota", "resource_exhausted", "rate limit", "retry"))

    @staticmethod
    def _is_network_or_timeout(exc: Exception) -> bool:
        exc_type = type(exc).__name__.lower()
        text = str(exc).lower()
        return any(
            token in f"{exc_type} {text}"
            for token in (
                "connecttimeout",
                "readtimeout",
                "timeout",
                "timed out",
                "connecterror",
                "network",
                "temporarily unavailable",
                "unavailable",
                "503",
                "502",
                "500",
                "504",
            )
        )

    @staticmethod
    def _is_auth_or_request_error(exc: Exception) -> bool:
        status_code = LLMService._extract_status_code(exc)
        text = str(exc).lower()
        return bool(
            status_code in {400, 401, 403, 404}
            or "api key not valid" in text
            or "permission denied" in text
            or "invalid argument" in text
        )

    def _to_service_error(self, exc: Exception) -> LLMServiceError:
        if isinstance(exc, LLMServiceError):
            return exc

        status_code = self._extract_status_code(exc)
        retry_after = self._extract_retry_after(exc)

        if self._is_quota_or_retryable(exc):
            return LLMServiceError(
                "Gemini quota or rate limits were reached.",
                code="GEMINI_RATE_LIMITED",
                status_code=status_code,
                retryable=True,
                retry_after=retry_after,
                raw_error=exc,
            )

        if self._is_network_or_timeout(exc):
            return LLMServiceError(
                "Gemini request timed out or the service could not be reached.",
                code="GEMINI_TIMEOUT",
                status_code=status_code,
                retryable=True,
                retry_after=retry_after,
                raw_error=exc,
            )

        if self._is_auth_or_request_error(exc):
            return LLMServiceError(
                "Gemini rejected the request. Check the API key, model name, and request payload.",
                code="GEMINI_BAD_REQUEST",
                status_code=status_code,
                retryable=False,
                raw_error=exc,
            )

        return LLMServiceError(
            "Gemini request failed.",
            code="GEMINI_REQUEST_FAILED",
            status_code=status_code,
            retryable=status_code is not None and status_code >= 500,
            retry_after=retry_after,
            raw_error=exc,
        )

    def _sleep_before_retry(self, attempt: int, service_error: LLMServiceError) -> None:
        delay = service_error.retry_after or self.retry_backoff * (2**attempt)
        if delay > 0:
            time.sleep(delay)

    def _request_once(self, prompt: str, model_name: str) -> str:
        if self.client is not None:
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = getattr(response, "text", None)
        else:
            model = self.legacy_genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            text = getattr(response, "text", None)

        if not text or not isinstance(text, str) or not text.strip():
            raise LLMServiceError("Gemini returned an empty response.", code="EMPTY_RESPONSE")

        return text.strip()

    def generate_response(self, prompt: str, stage: str | None = None) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMServiceError("Prompt must be a non-empty string.", code="INVALID_PROMPT")

        if stage:
            record_gemini_call(stage)

        last_error: LLMServiceError | None = None
        model_names = getattr(self, "model_names", [self.model_name])

        for model_index, model_name in enumerate(model_names):
            for attempt in range(self.max_retries + 1):
                try:
                    logger.info("Calling Gemini model", extra={"attempt": attempt + 1, "model": model_name})
                    result = self._request_once(prompt, model_name)
                    record_gemini_model_success(model_name, self.model_name)
                    if model_name != self.model_name:
                        logger.info("Gemini fallback model succeeded", extra={"model": model_name})
                    return result
                except Exception as exc:
                    record_gemini_model_failure()
                    last_error = self._to_service_error(exc)
                    logger.warning(
                        "Gemini request failed",
                        extra={
                            "attempt": attempt + 1,
                            "max_attempts": self.max_retries + 1,
                            "model": model_name,
                            "code": last_error.code,
                            "status_code": last_error.status_code,
                            "retryable": last_error.retryable,
                            "raw_error_type": type(exc).__name__,
                            "raw_error": str(exc),
                        },
                    )

                    if not last_error.retryable:
                        raise last_error from exc

                    if attempt < self.max_retries:
                        self._sleep_before_retry(attempt, last_error)
                        continue

                    if model_index < len(model_names) - 1:
                        next_model = model_names[model_index + 1]
                        logger.warning(
                            "Gemini model retries exhausted; switching fallback model",
                            extra={
                                "failed_model": model_name,
                                "next_model": next_model,
                                "code": last_error.code,
                                "status_code": last_error.status_code,
                                "raw_error": str(exc),
                            },
                        )
                        break

                    raise last_error from exc

        raise last_error or LLMServiceError("Gemini request failed.")


def track_gemini_call(stage: str):
    """Convenience wrapper for the agent layer to record a Gemini request."""
    return record_gemini_call(stage)
