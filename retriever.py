import arxiv
import logging

logger = logging.getLogger(__name__)

def fetch_arxiv_papers(query: str, max_results: int = 5):
    """
    Returns a list of dicts: {"title", "summary", "url"}.
    """
    try:
        search = arxiv.Search(query=query, max_results=max_results)
        return [
            {"title": r.title, "summary": r.summary, "url": r.entry_id}
            for r in search.results()
        ]
    except Exception as e:
        logger.error(f"Arxiv retrieval failed: {e}")
        return []