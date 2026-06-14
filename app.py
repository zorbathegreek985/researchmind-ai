import streamlit as st

st.set_page_config(page_title="ResearchMind AI", page_icon="🧠", layout="wide")

st.title("ResearchMind AI")
st.caption("Agent Debate Mode for research planning and critique")

st.markdown("""
This mode forces the research pipeline to debate its assumptions before a final roadmap is produced.
Each agent contributes a role in the review cycle:
1. Literature Agent presents evidence.
2. Gap Agent challenges assumptions.
3. Hypothesis Agent proposes ideas.
4. Critic Agent attacks weaknesses.
5. Experiment Agent improves methodology.
6. Roadmap Agent creates the final plan.
""")

with st.sidebar:
    st.header("Research Setup")
    topic = st.text_input("Research topic", value="AI for scientific literature review")
    goal = st.text_area("Goal / question", value="What is the most promising research direction and how should we test it?")
    run_debate = st.button("Start Debate Mode", use_container_width=True)


def build_debate(topic: str, goal: str):
    return [
        {
            "agent": "Literature Agent",
            "role": "Evidence Presenter",
            "summary": f"The literature around {topic} suggests strong interest in synthesis, retrieval, and evaluation quality. Current evidence supports a workflow that combines structured prompts, citation grounding, and iterative review."
        },
        {
            "agent": "Gap Agent",
            "role": "Assumption Challenger",
            "summary": f"A key gap is that many existing systems over-focus on generation and under-explain why a proposed idea is novel. The current literature often lacks a transparent critique loop before a plan is finalized."
        },
        {
            "agent": "Hypothesis Agent",
            "role": "Idea Generator",
            "summary": f"Hypothesis: a multi-agent research assistant that debates its own evidence before producing a roadmap will produce more credible and less brittle research plans for {topic}."
        },
        {
            "agent": "Critic Agent",
            "role": "Weakness Attacker",
            "summary": "Potential weaknesses include prompt sensitivity, limited grounding quality, and uneven agent quality. The system must include explicit review checkpoints and confidence scoring to avoid overclaiming."
        },
        {
            "agent": "Experiment Agent",
            "role": "Methodology Improver",
            "summary": "To improve validity, test the workflow on multiple topics, compare outputs from single-agent vs multi-agent pipelines, and measure clarity, novelty, feasibility, and review quality."
        },
        {
            "agent": "Roadmap Agent",
            "role": "Final Plan Builder",
            "summary": f"Final plan for {topic}: phase 1 establish evidence collection, phase 2 run debate review, phase 3 test hypotheses with controlled experiments, and phase 4 publish a reusable roadmap and report."
        },
    ]


if run_debate:
    st.success("Debate Mode started. The agents are now presenting and challenging the research plan.")

    debate = build_debate(topic, goal)

    for item in debate:
        with st.expander(f"{item['agent']} — {item['role']}", expanded=True):
            st.write(item['summary'])

    st.markdown("---")
    st.subheader("Final Research Plan")
    st.info("The final plan is synthesized from the debate: evidence is reviewed, assumptions are challenged, hypotheses are proposed, weaknesses are addressed, methodology is refined, and the roadmap is finalized.")

else:
    st.info("Enter a research topic and click 'Start Debate Mode' to display the full agent debate in Streamlit.")
