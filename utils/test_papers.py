import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.paper_search import search_papers

papers = search_papers("Artificial Intelligence")

for paper in papers:
    print("\nTITLE:")
    print(paper["title"])

    print("\nURL:")
    print(paper["url"])

    print("\nSUMMARY:")
    print(paper["summary"][:300])

    print("\n" + "="*80)