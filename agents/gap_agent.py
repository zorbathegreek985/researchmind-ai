from utils.llm import LLMService, LLMServiceError, build_fallback_gaps, print_gemini_exception_diagnostics


def find_research_gaps(summaries):
    prompt = f"""
You are a research assistant.

Below are summaries of research papers:

{summaries}

Identify:

1. Common themes
2. Unsolved problems
3. Research gaps
4. Future research opportunities

Give the answer in bullet points.
"""

    try:
        llm = LLMService()
        return llm.generate_response(prompt, stage="gap_analysis")

    except LLMServiceError as e:
        print_gemini_exception_diagnostics(e, stage="gap_analysis", service_error=e)
        return build_fallback_gaps(summaries)

    except Exception as e:
        print_gemini_exception_diagnostics(e, stage="gap_analysis")
        raise
