import arxiv


def search_papers(query, max_results=5):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    try:
        client = arxiv.Client()

        search = arxiv.Search(
            query=query.strip(),
            max_results=max_results
        )

        papers = []

        for result in client.results(search):
            papers.append({
                "title": result.title,
                "summary": result.summary,
                "url": result.entry_id
            })

        return papers

    except Exception as exc:
        raise RuntimeError(f"Paper search failed: {exc}") from exc