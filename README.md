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
GEMINI_TIMEOUT_SECONDS=45
GEMINI_MAX_RETRIES=3
GEMINI_RETRY_BACKOFF_SECONDS=2.0
```

`GEMINI_MODEL` is the primary model. `GEMINI_FALLBACK_MODELS` is a comma-separated list used when the primary model returns a retryable failure such as timeout, quota exhaustion, or temporary unavailability.
The timeout, retry count, and backoff values can be tuned for live demos.

## Run

```powershell
streamlit run app.py
```

Open the local Streamlit URL, enter a research topic, and click **Run research workflow**.

## Demo Flow

1. Start the app with `streamlit run app.py`.
2. Enter a focused topic, such as `AI agents for scientific literature review`.
3. Run the workflow and confirm the top metrics, workflow status, and model status are visible.
4. Review the **Final Research Brief** synthesis.
5. Download the Markdown or TXT report for submission notes.

## Verification

Run the automated test suite before presenting:

```powershell
python -m pytest
```

Optional syntax check:

```powershell
python -m compileall app.py agents utils
```

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

## Polish Pass Summary

- Final Research Brief: deterministic synthesis of key findings, research gaps, hypotheses, critiques, and next steps.
- Model status indicator: active, fallback, unavailable, and cached states are surfaced in the workflow panel.
- Encoding cleanup: UI status labels use ASCII-safe text for reliable Windows console and browser rendering.
- Repository hygiene: local environment files, caches, bytecode, and generated demo output folders are ignored.

## Known Limitations

- Paper search currently uses arXiv only.
- Outputs are not yet citation-grounded at the sentence level.
- The final brief is a deterministic synthesis of generated workflow outputs, not a separate expert review pass.
- Gemini availability and quota can affect live output quality, though fallback models and fallback content reduce demo risk.
- Dependencies are not fully pinned yet, so reproducibility can vary across environments.
