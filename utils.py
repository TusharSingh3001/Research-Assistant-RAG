from langchain.schema import Document

def prepare_documents(papers):
    """
    Convert each paper dict into a LangChain Document,
    using the 'summary' as page_content and storing
    title/url in metadata for source tracking.
    """
    return [
        Document(
            page_content=p["summary"],
            metadata={
                "title": p["title"],
                "url": p["url"],
                "source": p["url"],
            }
        )
        for p in papers
    ]