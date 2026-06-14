from utils.llm import LLMService, LLMServiceError, build_fallback_summary, print_gemini_exception_diagnostics

def summarize_paper(title, abstract):
    prompt = f"""
Paper Title:
{title}

Abstract:
{abstract}

Explain:

1. What is this paper about?
2. Main contribution
3. Key findings
4. Research limitations

Keep the summary simple.
"""

    try:
        llm = LLMService()
        return llm.generate_response(prompt, stage="summarization")

    except LLMServiceError as e:
        print_gemini_exception_diagnostics(e, stage="summarization", service_error=e)
        return build_fallback_summary(title, abstract)

    except Exception as e:
        print_gemini_exception_diagnostics(e, stage="summarization")
        raise
