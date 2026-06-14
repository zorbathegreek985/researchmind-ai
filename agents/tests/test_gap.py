from agents.gap_agent import find_research_gaps


def main():
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


if __name__ == "__main__":
    main()
