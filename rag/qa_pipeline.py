# qa_pipeline.py

import os
import shutil
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

INDEX_PATH = "faiss_index"

def build_qa(docs, rebuild_index=False):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if rebuild_index and os.path.exists(INDEX_PATH):
        shutil.rmtree(INDEX_PATH)

    if os.path.exists(INDEX_PATH):
        db = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(INDEX_PATH)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a STRICT document-grounded QA assistant.

CRITICAL RULES (DO NOT BREAK):
- Use ONLY the provided context.
- Answer ONLY if the question can be answered by a SINGLE explicit sentence in the document.
- Do NOT infer, summarize, explain, or combine information.
- Do NOT describe policies, trends, or stances unless explicitly stated.
- If the answer is NOT explicitly stated verbatim, respond EXACTLY with:

Not found in the document.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="tngtech/deepseek-r1t2-chimera:free",
        temperature=0
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
