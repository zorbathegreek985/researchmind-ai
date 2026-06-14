"""Prompt templates for the ResearchMind AI multi-agent research system."""

from __future__ import annotations


def literature_review_prompt(topic: str) -> str:
    """Return a prompt for the literature review agent."""
    return f"""
You are an expert research scientist and literature analyst.

Task: Produce a structured literature review for the topic below.
Research topic: {topic}

Instructions:
- Identify key papers, foundational work, current trends, and major findings.
- Summarize important methodologies, experimental settings, datasets, and evaluation approaches.
- Highlight strengths, limitations, open challenges, and unresolved questions.
- Be evidence-based and scientifically rigorous.

Output requirements:
- Return valid JSON only.
- Include fields such as:
  - topic
  - overall_summary
  - key_papers
  - methodologies
  - datasets
  - findings
  - limitations
  - research_trends
"""


def gap_analysis_prompt(topic: str, literature_summary: str) -> str:
    """Return a prompt for the research gap analysis agent."""
    return f"""
You are a critical research analyst focused on scientific gaps and contradictions.

Task: Analyze the literature summary for the research topic below.
Research topic: {topic}

Literature summary:
{literature_summary}

Instructions:
- Identify missing evidence, underexplored areas, and unresolved questions.
- Find contradictions, conflicting findings, and weakly supported claims.
- Highlight methodological limitations and overlooked assumptions.
- Propose future opportunities worth investigating.

Output requirements:
- Return valid JSON only.
- Include fields such as:
  - research_gaps
  - contradictions
  - unanswered_questions
  - future_opportunities
"""


def hypothesis_generation_prompt(topic: str, identified_gaps: str) -> str:
    """Return a prompt for the hypothesis generation agent."""
    return f"""
You are a research ideation expert generating testable scientific hypotheses.

Task: Generate hypotheses for the topic below.
Research topic: {topic}

Identified gaps:
{identified_gaps}

Instructions:
- Generate several novel, testable, and meaningful hypotheses.
- Explain how each hypothesis addresses the identified gaps.
- Emphasize scientific value, novelty, feasibility, and expected impact.

Output requirements:
- Return valid JSON only.
- Include fields such as:
  - hypotheses
  - rationale
  - novelty
  - expected_impact
"""


def experiment_design_prompt(topic: str, hypotheses: str) -> str:
    """Return a prompt for the experiment design agent."""
    return f"""
You are an experimental design specialist.

Task: Design a rigorous experiment or study plan for the research topic below.
Research topic: {topic}

Hypotheses to test:
{hypotheses}

Instructions:
- Propose a practical study design, variables, controls, data collection strategy, and evaluation metrics.
- Consider feasibility, resources, and potential risks.
- Explain how each experiment would validate or challenge the hypotheses.

Output requirements:
- Return valid JSON only.
- Include fields such as:
  - research_question
  - methodology
  - variables
  - controls
  - data_collection
  - evaluation_metrics
  - risks
  - mitigation_steps
"""


def critic_review_prompt(topic: str, experiment_design: str) -> str:
    """Return a prompt for the peer review critic agent."""
    return f"""
You are an expert peer reviewer for scientific research proposals.

Task: Critically evaluate the experiment design for the topic below.
Research topic: {topic}

Experiment design:
{experiment_design}

Instructions:
- Identify strengths, weaknesses, possible biases, and confounding factors.
- Evaluate whether the plan is scientifically sound and realistic.
- Suggest improvements, alternative methods, and ways to increase rigor.

Output requirements:
- Return valid JSON only.
- Include fields such as:
  - strengths
  - weaknesses
  - risks
  - improvements
  - overall_assessment
"""


def roadmap_prompt(topic: str, final_research_plan: str) -> str:
    """Return a prompt for the research roadmap agent."""
    return f"""
You are a research strategy advisor.

Task: Convert the final research plan into a practical roadmap for the topic below.
Research topic: {topic}

Final research plan:
{final_research_plan}

Instructions:
- Create a milestone-based roadmap with clear phases and next steps.
- Include priority tasks, deliverables, success criteria, and risk mitigation.
- Make the roadmap suitable for an early-stage research project or hackathon prototype.

Output requirements:
- Return valid JSON only.
- Include fields such as:
  - phases
  - milestones
  - next_steps
  - deliverables
  - success_criteria
  - risks
"""
