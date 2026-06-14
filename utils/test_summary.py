import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.summarizer import summarize_paper


def main():
    result = summarize_paper(
        "Artificial Intelligence in Healthcare",
        """
        Artificial intelligence is increasingly being used in healthcare.
        This paper reviews applications of AI in diagnosis, treatment,
        and patient monitoring.
        """,
    )

    print(result)


if __name__ == "__main__":
    main()
