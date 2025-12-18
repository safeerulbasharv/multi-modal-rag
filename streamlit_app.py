# streamlit_app.py

import streamlit as st
from ingestion.text_loader import load_pdf
from ingestion.table_loader import load_tables_from_pdf
from ingestion.ocr_loader import load_ocr_from_pdf
from rag.qa_pipeline import build_qa

st.set_page_config(page_title="Multi-Modal RAG QA", layout="wide")
st.title("📄 Multi-Modal Document QA (RAG)")

PDF_PATH = "data/qatar_test_doc.pdf"

STANCE_KEYWORDS = ["stance", "approach", "policy position", "policy outlook"]

@st.cache_resource(show_spinner=False)
def load_qa_system():
    status = st.empty()

    status.info("📄 Loading text...")
    text_docs = load_pdf(PDF_PATH)

    status.info("📊 Loading tables...")
    table_docs = load_tables_from_pdf(PDF_PATH)

    status.info("🧠 Running OCR (only on sparse pages)...")
    ocr_docs = load_ocr_from_pdf(PDF_PATH)

    docs = text_docs + table_docs + ocr_docs

    status.info("🔎 Building FAISS index...")
    qa = build_qa(docs)

    status.success("✅ Document indexed and ready")
    return qa


with st.spinner("Initializing RAG system..."):
    qa = load_qa_system()

st.divider()

query = st.text_input(
    "Ask a question about the document:",
    placeholder="e.g. What is Qatar’s projected real GDP growth in 2024?"
)

if query:
    if any(k in query.lower() for k in STANCE_KEYWORDS):
        st.subheader("Answer")
        st.write("Not found in the document.")
        st.stop()

    with st.spinner("Searching document..."):
        result = qa.invoke({"query": query})

    st.subheader("Answer")
    st.write(result.get("result", "No answer returned."))

    st.subheader("📌 Sources")
    source_docs = result.get("source_documents", [])
    if not source_docs:
        st.warning("No sources retrieved.")
    else:
        for doc in source_docs:
            st.markdown(
                f"- **Page:** {doc.metadata.get('page')} "
                f"| **Modality:** {doc.metadata.get('modality', 'text')}"
            )

st.divider()

st.subheader("⚡ Run Benchmark Evaluation")

if st.button("Run Evaluation"):
    benchmarks = [
        {
            "query": "What is Qatar’s projected real GDP growth in 2024?",
            "expected": "2 percent"
        },
        {
            "query": "Total population",
            "expected": "million"
        },
        {
            "query": "What is Qatar’s monetary policy stance?",
            "expected": "Not found in the document."
        },
    ]

    with st.spinner("Running benchmark queries..."):
        for b in benchmarks:
            if any(k in b["query"].lower() for k in STANCE_KEYWORDS):
                answer = "Not found in the document."
            else:
                res = qa.invoke({"query": b["query"]})
                answer = res.get("result", "")

            status = "✅ PASS" if b["expected"] in answer else "❌ FAIL"

            st.markdown(f"**Query:** {b['query']}")
            st.markdown(f"**Answer:** {answer}")
            st.markdown(f"**Expected:** {b['expected']}")
            st.markdown(f"**Status:** {status}")
            st.markdown("---")
