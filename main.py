# main.py

import streamlit as st
import logging
import time
from dotenv import load_dotenv

from retriever import fetch_arxiv_papers
from utils import prepare_documents
from langgraph_pipeline import llm, embeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="RAG Research Assistant", layout="wide")
st.title("Scientific Research Question Answering")

# ————— Helper functions —————

def get_papers(query: str, max_results: int = 5):
    return fetch_arxiv_papers(query, max_results)

def build_vectorstore(docs):
    # split into chunks & index
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    split_docs = splitter.split_documents(docs)
    vs = FAISS.from_documents(split_docs, embeddings)

    # Optional: GPU-offload FAISS for faster search
    try:
        import faiss
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, vs.index)
        vs.index = gpu_index
    except Exception:
        logging.warning("GPU FAISS offload failed; using CPU.")

    return vs




def load_qa_chain(retriever):
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

# ————— UI —————

query = st.text_input(
    "Enter research topic:",
    "3D bioprinting in regenerative medicine"
)
question = st.text_area(
    "Ask a question based on this topic:",
    "What are the latest trends in 3D bioprinting?"
)

if st.button("Answer Question"):
    with st.spinner("Retrieving context and answering…"):
        t0 = time.perf_counter()

        papers = get_papers(query)
        if not papers:
            st.warning("No papers found for that topic.")
        else:
            # prepare, index, and run
            docs      = prepare_documents(papers)
            vs        = build_vectorstore(docs)
            retriever = vs.as_retriever()
            qa_chain  = load_qa_chain(retriever)

            result  = qa_chain({"question": question})
            answer  = result["answer"]
            sources = [doc.metadata for doc in result["source_documents"]]

            # display
            st.subheader("Answer")
            st.markdown(answer)
            st.subheader("Sources")
            for s in sources:
                st.markdown(f"- [{s['title']}]({s['url']})")

        st.write(f" Total time: {time.perf_counter() - t0:.2f}s")
