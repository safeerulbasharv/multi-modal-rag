#table_loader.py

import fitz
from langchain.schema import Document

def load_tables_from_pdf(path):
    docs = []
    pdf = fitz.open(path)

    for page_num, page in enumerate(pdf):
        text = page.get_text("text")

        if any(x in text for x in ["Table", "% of GDP", "Population", "Exports", "QAR"]):
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path,
                        "page": page_num,
                        "modality": "table"
                    }
                )
            )

    return docs
