import html as html_lib
import re

import streamlit as st

from agents.critic_agent import critique_hypotheses
from agents.gap_agent import find_research_gaps
from agents.hypothesis_agent import generate_hypotheses
from utils.llm import (
    format_gemini_error,
    get_gemini_call_counts,
    get_gemini_diagnostic,
    get_gemini_runtime_status,
    reset_gemini_call_counts,
)
from utils.paper_search import search_papers
from utils.summarizer import summarize_paper


st.set_page_config(page_title="ResearchMind AI", layout="wide")

st.markdown(
    """
    <style>
        :root { color-scheme: light; }
        .stApp {
            background: linear-gradient(135deg, #f8fbff 0%, #ffffff 45%, #eff6ff 100%);
            color: #111827;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            color: #111827;
        }
        h1, h2, h3, h4, p, li, div, span, label, .stTextInput, .stTextArea, .stMarkdown {
            color: #111827 !important;
        }
        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #dbe4ee;
        }
        [data-testid="stSidebar"] * {
            color: #111827 !important;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dbe4ee;
            border-radius: 16px;
            padding: 0.75rem;
            box-shadow: 0 8px 18px rgba(148, 163, 184, 0.18);
            color: #111827;
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: #374151 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #111827 !important;
            font-weight: 700;
        }
        .card {
            background: #ffffff;
            border: 1px solid #dbe4ee;
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(148, 163, 184, 0.18);
            color: #111827;
            margin-bottom: 1rem;
        }
        div[data-testid="stNotification"] {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #111827;
        }
        div[data-testid="stNotification"] p,
        div[data-testid="stNotification"] div,
        div[data-testid="stNotification"] span {
            color: #111827 !important;
        }
        div[data-testid="stAlert"] {
            border-radius: 12px;
        }
        .stDownloadButton > button,
        .stButton > button {
            border: 1px solid #cbd5e1;
            color: #111827;
            background: #ffffff;
            border-radius: 10px;
        }
        .stDownloadButton > button:hover,
        .stButton > button:hover {
            border-color: #2563eb;
            color: #111827;
            background: #eff6ff;
        }
        .stExpander {
            border: 1px solid #dbe4ee;
            border-radius: 12px;
            background: #ffffff;
        }
        .stExpander summary {
            color: #111827 !important;
            font-weight: 600;
        }
        .badge {
            display: inline-block;
            padding: 0.18rem 0.45rem;
            border-radius: 999px;
            font-size: 0.78rem;
            background: #eff6ff;
            color: #1d4ed8;
        }
        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.5rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.92rem;
            margin-right: 0.4rem;
        }
        .status-success { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
        .status-warning { background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }
        .status-failed { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
        .model-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }
        .model-active { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
        .model-fallback { background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }
        .model-unavailable { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
        .brief-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.85rem;
            margin-top: 0.75rem;
        }
        .brief-panel {
            border: 1px solid #dbe4ee;
            border-radius: 12px;
            padding: 0.85rem;
            background: #f8fafc;
        }
        .brief-panel h4 {
            margin: 0 0 0.45rem 0;
            font-size: 1rem;
        }
        .brief-panel ul {
            margin: 0;
            padding-left: 1.1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

startup_diag = get_gemini_diagnostic()

st.title("ResearchMind AI")
st.caption("Hackathon-ready literature review, gap analysis, hypothesis generation, and critique workflow")

st.markdown(
    """
    This interface has been polished for a demo presentation with a clearer workflow, live status cues,
    and simple download options for the final research brief.
    """
)

st.caption(
    "Startup diagnostic - model: "
    f"{startup_diag['model_name']} | api key loaded: {'yes' if startup_diag['api_key_loaded'] else 'no'} "
    f"| key suffix: {startup_diag['api_key_suffix'] or 'n/a'}"
)


def _is_service_failure(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("unavailable", "quota", "timed out", "timeout", "could not be reached", "please wait", "temporarily unavailable", "fallback mode"))


def _clean_item(text: str, max_length: int = 190) -> str:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", str(text).strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) > max_length:
        return cleaned[: max_length - 1].rstrip() + "..."
    return cleaned


def _extract_brief_items(text: str, limit: int = 3) -> list[str]:
    items = []
    for line in str(text).splitlines():
        cleaned = _clean_item(line)
        if cleaned:
            items.append(cleaned)
        if len(items) >= limit:
            break
    if not items and text:
        sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
        items = [_clean_item(sentence) for sentence in sentences[:limit] if sentence.strip()]
    return items[:limit]


def _build_final_research_brief(result: dict) -> dict:
    summary_items = []
    for entry in result.get("papers", [])[:3]:
        paper = entry.get("paper", {})
        title = paper.get("title", "Untitled paper")
        summary = _clean_item(entry.get("summary") or paper.get("summary") or "No summary available.")
        summary_items.append(f"{title}: {summary}")

    hypotheses = [
        _clean_item(item.get("hypothesis", "Untitled hypothesis"))
        for item in result.get("hypotheses", [])[:3]
        if isinstance(item, dict)
    ]

    critique_items = []
    next_steps = []
    for item in result.get("critiques", [])[:3]:
        if not isinstance(item, dict):
            continue
        critique = item.get("critique")
        if critique:
            critique_items.append(_clean_item(critique))
        for improvement in item.get("improvements", [])[:2]:
            next_steps.append(_clean_item(improvement))

    if hypotheses and not next_steps:
        next_steps.append(f"Design a small validation study for: {hypotheses[0]}")
    if result.get("papers") and not next_steps:
        next_steps.append("Compare the selected papers by datasets, metrics, assumptions, and deployment constraints.")
    if not next_steps:
        next_steps.append("Run the workflow with a narrower research topic and inspect the top papers manually.")

    return {
        "key_findings": summary_items or ["No paper summaries were available for synthesis."],
        "research_gaps": _extract_brief_items(result.get("gaps", ""), limit=3) or ["No research gaps were generated."],
        "hypotheses": hypotheses or ["No hypotheses were generated."],
        "critiques": critique_items or ["No critique notes were generated."],
        "next_steps": next_steps[:3],
    }


def _derive_model_status(stats: dict) -> dict:
    runtime_status = get_gemini_runtime_status()
    if runtime_status.get("fallback_model_used"):
        return {"label": "[Fallback] Gemini fallback active", "class": "model-fallback", "model": runtime_status.get("last_model")}
    if runtime_status.get("last_model"):
        return {"label": "[Active] Gemini active", "class": "model-active", "model": runtime_status.get("last_model")}
    if stats.get("total", 0) > 0 or runtime_status.get("request_failed"):
        return {"label": "[Unavailable] Gemini unavailable", "class": "model-unavailable", "model": None}
    return {"label": "[Unavailable] Gemini unavailable", "class": "model-unavailable", "model": None}


def _render_final_research_brief(brief: dict) -> None:
    sections = [
        ("Key findings", brief.get("key_findings", [])),
        ("Research gaps", brief.get("research_gaps", [])),
        ("Hypotheses", brief.get("hypotheses", [])),
        ("Critiques", brief.get("critiques", [])),
        ("Recommended next steps", brief.get("next_steps", [])),
    ]
    html = ["<div class='brief-grid'>"]
    for title, items in sections:
        html.append("<div class='brief-panel'>")
        html.append(f"<h4>{html_lib.escape(title)}</h4>")
        html.append("<ul>")
        for item in items[:3]:
            html.append(f"<li>{html_lib.escape(str(item))}</li>")
        html.append("</ul></div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _build_markdown_report(topic: str, result: dict) -> str:
    brief = result.get("final_brief") or _build_final_research_brief(result)
    lines = [
        f"# ResearchMind AI report for: {topic}",
        "",
        "## Overview",
        f"- Papers found: {len(result.get('papers', []))}",
        f"- Summaries generated: {len(result.get('papers', []))}",
        f"- Hypotheses generated: {len(result.get('hypotheses', []))}",
        f"- Critiques generated: {len(result.get('critiques', []))}",
        "",
        "## Final Research Brief",
        "",
        "### Key findings",
        *[f"- {item}" for item in brief.get("key_findings", [])],
        "",
        "### Research gaps",
        *[f"- {item}" for item in brief.get("research_gaps", [])],
        "",
        "### Hypotheses",
        *[f"- {item}" for item in brief.get("hypotheses", [])],
        "",
        "### Critiques",
        *[f"- {item}" for item in brief.get("critiques", [])],
        "",
        "### Recommended next steps",
        *[f"- {item}" for item in brief.get("next_steps", [])],
        "",
        "## Research gaps",
        result.get("gaps", "No gap analysis available."),
        "",
        "## Hypotheses",
    ]
    for index, item in enumerate(result.get("hypotheses", []), start=1):
        lines.append(f"{index}. {item.get('hypothesis', 'Hypothesis')}")
        lines.append(f"   - Rationale: {item.get('rationale', 'N/A')}")
        lines.append(f"   - Novelty: {item.get('novelty', 'N/A')}")
        lines.append(f"   - Expected impact: {item.get('expected_impact', 'N/A')}")
        lines.append("")

    lines.extend(["## Critiques", ""])
    for index, item in enumerate(result.get("critiques", []), start=1):
        lines.append(f"{index}. {item.get('hypothesis', 'Hypothesis')}")
        lines.append(f"   - Critique: {item.get('critique', 'N/A')}")
        lines.append("   - Weaknesses:")
        for weakness in item.get("weaknesses", []):
            lines.append(f"     - {weakness}")
        lines.append("   - Improvements:")
        for improvement in item.get("improvements", []):
            lines.append(f"     - {improvement}")
        lines.append("")

    return "\n".join(lines)


def _build_text_report(topic: str, result: dict) -> str:
    return _build_markdown_report(topic, result).replace("# ", "").replace("## ", "").replace("- ", "  - ")


def run_mvp(topic: str):
    cache = st.session_state.get("workflow_cache", {})
    if topic in cache:
        reset_gemini_call_counts()
        cached = dict(cache[topic])
        cached["cache_hit"] = True
        cached["gemini_stats"] = get_gemini_call_counts()
        cached.setdefault("final_brief", _build_final_research_brief(cached))
        cached.setdefault("model_status", {"label": "[Cached] Previous model result", "class": "model-active", "model": startup_diag["model_name"]})
        return cached

    reset_gemini_call_counts()
    workflow_status = []

    try:
        with st.spinner("Searching for relevant papers..."):
            papers = search_papers(topic, max_results=5)
        workflow_status.append({"agent": "Paper search", "status": "success", "details": f"Found {len(papers)} papers."})
    except Exception as exc:
        papers = []
        workflow_status.append({"agent": "Paper search", "status": "failed", "details": format_gemini_error(exc)})

    paper_summaries = []
    summary_failures = 0
    with st.spinner("Summarizing the paper abstracts..."):
        for paper in papers:
            summary = summarize_paper(paper["title"], paper["summary"])
            if _is_service_failure(summary):
                summary_failures += 1
            paper_summaries.append({"paper": paper, "summary": summary})

    if summary_failures:
        workflow_status.append({"agent": "Summarization", "status": "warning", "details": f"{summary_failures} summary task(s) used the friendly fallback path."})
    else:
        workflow_status.append({"agent": "Summarization", "status": "success", "details": f"Generated {len(paper_summaries)} summaries."})

    literature_text = "\n\n".join(
        f"Title: {entry['paper']['title']}\nSummary: {entry['paper']['summary']}\nAI summary: {entry['summary']}"
        for entry in paper_summaries
    )

    with st.spinner("Analyzing research gaps..."):
        gaps = find_research_gaps(literature_text)
    gap_status = "success"
    if _is_service_failure(gaps):
        gap_status = "failed"
    workflow_status.append({"agent": "Gap analysis", "status": gap_status, "details": "Generated the gap analysis summary." if gap_status == "success" else gaps})

    with st.spinner("Generating hypotheses..."):
        hypotheses = generate_hypotheses(topic, gaps)
    workflow_status.append({"agent": "Hypothesis generation", "status": "success", "details": f"Generated {len(hypotheses)} hypotheses."})

    with st.spinner("Critiquing the hypotheses..."):
        critiques = critique_hypotheses(topic, hypotheses)
    workflow_status.append({"agent": "Critique review", "status": "success", "details": f"Generated {len(critiques)} critique notes."})

    stats = get_gemini_call_counts()
    result = {
        "topic": topic,
        "papers": paper_summaries,
        "gaps": gaps,
        "hypotheses": hypotheses,
        "critiques": critiques,
        "gemini_stats": stats,
        "workflow_status": workflow_status,
        "cache_hit": False,
    }
    result["final_brief"] = _build_final_research_brief(result)
    result["model_status"] = _derive_model_status(stats)

    st.session_state["workflow_cache"] = {**cache, topic: result}
    return result


with st.sidebar:
    st.header("Research Setup")
    topic = st.text_input("Research topic", value="AI for scientific literature review")
    run_workflow = st.button("Run research workflow", use_container_width=True)
    st.caption("The workflow uses the paper search, summarization, gap, hypothesis, and critique stages in one flow.")


if run_workflow:
    topic = topic.strip() or "AI for scientific literature review"
    result = run_mvp(topic)

    st.success("Research workflow completed. Your results are ready for a hackathon demo.")

    if result.get("cache_hit"):
        st.info("Using cached results for this topic. Gemini calls were not repeated for this run.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Papers found", len(result.get("papers", [])))
    col2.metric("Summaries generated", len(result.get("papers", [])))
    col3.metric("Hypotheses generated", len(result.get("hypotheses", [])))
    col4.metric("Critiques generated", len(result.get("critiques", [])))

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Workflow status")
    model_status = result.get("model_status", {})
    model_detail = f" ({model_status.get('model')})" if model_status.get("model") else ""
    st.markdown(
        f"<div class='model-chip {model_status.get('class', 'model-unavailable')}'>{model_status.get('label', '[Unavailable] Gemini unavailable')}{model_detail}</div>",
        unsafe_allow_html=True,
    )
    for item in result.get("workflow_status", []):
        status_class = "status-success" if item["status"] == "success" else "status-warning" if item["status"] == "warning" else "status-failed"
        icon = "OK" if item["status"] == "success" else "WARN" if item["status"] == "warning" else "FAIL"
        agent = html_lib.escape(str(item["agent"]))
        details = html_lib.escape(str(item["details"]))
        st.markdown(f"<div style='margin-bottom: 0.35rem;'><span class='status-chip {status_class}'>{icon} {agent}</span> {details}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Final Research Brief")
    _render_final_research_brief(result.get("final_brief", {}))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Download results")
    md = _build_markdown_report(topic, result)
    txt = _build_text_report(topic, result)
    d1, d2 = st.columns(2)
    d1.download_button("Download Markdown", md, file_name=f"researchmind_{re.sub(r'[^a-z0-9]+', '_', topic.lower()).strip('_') or 'report'}.md", mime="text/markdown")
    d2.download_button("Download TXT", txt, file_name=f"researchmind_{re.sub(r'[^a-z0-9]+', '_', topic.lower()).strip('_') or 'report'}.txt", mime="text/plain")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("1. Relevant papers")
    for entry in result["papers"]:
        with st.expander(entry["paper"]["title"], expanded=True):
            st.markdown(f"[Open paper]({entry['paper']['url']})")
            st.write(entry["paper"]["summary"])
            st.info(entry["summary"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("2. Research gaps")
    st.write(result["gaps"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("3. Generated hypotheses")
    for index, item in enumerate(result["hypotheses"], start=1):
        st.markdown(f"**{index}. {item['hypothesis']}**")
        st.write(f"Rationale: {item['rationale']}")
        st.write(f"Novelty: {item['novelty']}")
        st.write(f"Expected impact: {item['expected_impact']}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("4. Critiques")
    for index, item in enumerate(result["critiques"], start=1):
        st.markdown(f"**{index}. {item['hypothesis']}**")
        st.write(item['critique'])
        st.write("Weaknesses:")
        for weakness in item.get("weaknesses", []):
            st.write(f"- {weakness}")
        st.write("Improvements:")
        for improvement in item.get("improvements", []):
            st.write(f"- {improvement}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "Gemini request counts for this run: "
        f"summarization={result['gemini_stats']['summarization']}, "
        f"gap_analysis={result['gemini_stats']['gap_analysis']}, "
        f"hypothesis={result['gemini_stats']['hypothesis']}, "
        f"critique={result['gemini_stats']['critique']}, "
        f"total={result['gemini_stats']['total']}"
    )
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.info("Enter a research topic and click 'Run research workflow' to generate papers, summaries, research gaps, hypotheses, and critiques.")
    st.markdown("</div>", unsafe_allow_html=True)
