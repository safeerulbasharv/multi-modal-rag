from ingestion.text_loader import load_pdf
from ingestion.table_loader import load_tables_from_pdf
from ingestion.ocr_loader import load_ocr_from_pdf
from rag.qa_pipeline import build_qa

PDF_PATH = "data/qatar_test_doc.pdf"

docs = (
    load_pdf(PDF_PATH)
    + load_tables_from_pdf(PDF_PATH)
    + load_ocr_from_pdf(PDF_PATH)
)

qa = build_qa(docs, rebuild_index=True)

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
    }
]

print("\nRunning Benchmark Evaluation\n")

for b in benchmarks:
    res = qa.invoke({"query": b["query"]})
    ans = res["result"]
    status = "✅ PASS" if b["expected"] in ans else "❌ FAIL"

    print(f"Query: {b['query']}")
    print(f"Answer: {ans}")
    print(f"Expected: {b['expected']}")
    print(f"Status: {status}")
    print("-" * 60)
