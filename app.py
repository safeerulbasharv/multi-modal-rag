# app.py

from ingestion.text_loader import load_pdf
from ingestion.table_loader import load_tables_from_pdf
from ingestion.ocr_loader import load_ocr_from_pdf
from rag.qa_pipeline import build_qa

PDF_PATH = "data/qatar_test_doc.pdf"

STANCE_KEYWORDS = ["stance", "approach", "policy position", "policy outlook"]

print("Loading text...")
text_docs = load_pdf(PDF_PATH)

print("Loading tables...")
table_docs = load_tables_from_pdf(PDF_PATH)

print("Running OCR (this may take ~30s)...")
ocr_docs = load_ocr_from_pdf(PDF_PATH)

docs = text_docs + table_docs + ocr_docs

print("Building FAISS index / QA pipeline...")
qa = build_qa(docs)

print("✅ Multi-Modal RAG ready\n")

while True:
    q = input("Ask (or 'exit'): ").strip()
    if q.lower() == "exit":
        break

    if any(k in q.lower() for k in STANCE_KEYWORDS):
        print("\nAnswer: Not found in the document.")
        print("-" * 50)
        continue

    res = qa.invoke({"query": q})

    print("\nAnswer:", res.get("result", "No answer returned."))
    print("Sources:")
    for d in res.get("source_documents", []):
        print(f"- Page {d.metadata.get('page')} | {d.metadata.get('modality')}")
    print("-" * 50)
