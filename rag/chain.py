"""
Step 2 — LangChain RAG chain using GPT-4o and Chroma retriever.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag.retriever import get_retriever

load_dotenv()

PROMPT_TEMPLATE = """You are an expert assistant for the UCI Machine Learning Repository.
Use the retrieved dataset information below to recommend the top 3 most relevant datasets for the user's question.
For each dataset, write exactly one sentence explaining why it fits.

Context:
{context}

Question: {question}

Answer:"""


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _parse_abstract(page_content: str) -> str:
    for line in page_content.splitlines():
        if line.startswith("Abstract:"):
            return line.replace("Abstract:", "").strip()
    return ""


# Module-level cache — chain and retriever built once per process
_retriever = None
_chain = None


def _get_chain(k: int = 5):
    global _retriever, _chain
    if _chain is None:
        _retriever = get_retriever(k=k)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        _chain = (
            {"context": _retriever | _format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    return _retriever, _chain


def query(question: str, k: int = 5) -> dict:
    retriever, chain = _get_chain(k=k)
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    sources = [
        {
            "name": doc.metadata.get("name"),
            "area": doc.metadata.get("area"),
            "tasks": doc.metadata.get("tasks"),
            "num_instances": doc.metadata.get("num_instances"),
            "num_features": doc.metadata.get("num_features"),
            "year": doc.metadata.get("year"),
            "abstract": _parse_abstract(doc.page_content),
            "url": f"https://archive.ics.uci.edu/dataset/{doc.metadata.get('id')}",
        }
        for doc in docs
    ]
    return {"answer": answer, "sources": sources}
