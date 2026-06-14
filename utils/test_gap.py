import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.gap_agent import find_research_gaps

sample_summaries = """
Paper 1:
AI helps diagnose diseases from medical images.

Paper 2:
AI improves patient monitoring.

Paper 3:
AI predicts treatment outcomes.
"""

result = find_research_gaps(sample_summaries)

print(result)