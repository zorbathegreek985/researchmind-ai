# ResearchMind AI

ResearchMind AI is a Streamlit MVP for turning a research topic into a judge-friendly research brief. It searches arXiv, summarizes relevant papers with Gemini, identifies research gaps, proposes hypotheses, critiques those hypotheses, and exports the results as Markdown or plain text.

## Features

- arXiv paper search for a user-provided research topic
- Gemini-powered paper summaries
- Research gap analysis
- Structured hypothesis generation
- Critique notes with weaknesses and improvements
- Final Research Brief that synthesizes findings, gaps, hypotheses, critiques, and next steps
- Gemini retry handling with fallback model support
- Workflow status indicators and Gemini model health status
- Streamlit UI with downloadable Markdown and TXT reports
- Friendly fallback content when Gemini is unavailable

## Installation

1. Create and activate a Python virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

## Environment Setup

1. Copy `.env.example` to `.env`.
2. Replace the placeholder API key with your Gemini API key.

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODELS=gemini-2.0-flash,gemini-2.0-flash-lite
```

`GEMINI_MODEL` is the primary model. `GEMINI_FALLBACK_MODELS` is a comma-separated list used when the primary model returns a retryable failure such as timeout, quota exhaustion, or temporary unavailability.

## Run

```powershell
streamlit run app.py
```

Open the local Streamlit URL, enter a research topic, and click **Run research workflow**.

## Screenshots

Add screenshots here before submission:

- Main workflow screen
- Final Research Brief section
- Downloaded report preview

## Project Structure

```text
app.py                         Streamlit application
utils/llm.py                   Gemini client, retries, fallback models, diagnostics
utils/paper_search.py          arXiv search helper
utils/summarizer.py            Paper summary helper
utils/prompts.py               Prompt templates
agents/gap_agent.py            Research gap analysis
agents/hypothesis_agent.py     Hypothesis generation
agents/critic_agent.py         Hypothesis critique
agents/tests/                  Pytest coverage for MVP behavior
utils/test_*.py                Demo scripts guarded by __main__
```

## Known Limitations

- Paper search currently uses arXiv only.
- Outputs are not yet citation-grounded at the sentence level.
- The final brief is a deterministic synthesis of generated workflow outputs, not a separate expert review pass.
- Gemini availability and quota can affect live output quality, though fallback models and fallback content reduce demo risk.
- Dependencies are not fully pinned yet, so reproducibility can vary across environments.
