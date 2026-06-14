from utils.llm import LLMService, LLMServiceError, build_fallback_summary

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
        print(f"Summarization failed: {e}")
        return build_fallback_summary(title, abstract)

    except Exception as e:
        print(f"Unexpected summarization error: {repr(e)}")
        raise