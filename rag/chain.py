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
Use the retrieved dataset information below to recommend the most relevant datasets for the user's question.
For each recommended dataset, briefly explain why it fits.

Context:
{context}

Question: {question}

Answer:"""


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_chain(k: int = 5):
    retriever = get_retriever(k=k)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return retriever, chain


def query(question: str, k: int = 5) -> dict:
    retriever, chain = get_chain(k=k)
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
        }
        for doc in docs
    ]
    return {"answer": answer, "sources": sources}
