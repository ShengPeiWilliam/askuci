"""
Step 2 — LangChain RAG chain using GPT-4o and Chroma retriever.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rag.retriever import get_retriever

load_dotenv()

PROMPT_TEMPLATE = """You are an expert assistant for the UCI Machine Learning Repository.
Use the retrieved dataset information below to recommend the most relevant datasets for the user's question.
For each recommended dataset, briefly explain why it fits.

Context:
{context}

Question: {question}

Answer:"""


def get_chain(k: int = 5):
    retriever = get_retriever(k=k)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


def query(question: str, k: int = 5) -> dict:
    chain = get_chain(k=k)
    result = chain.invoke({"query": question})
    sources = [
        {
            "name": doc.metadata.get("name"),
            "area": doc.metadata.get("area"),
            "tasks": doc.metadata.get("tasks"),
            "num_instances": doc.metadata.get("num_instances"),
            "num_features": doc.metadata.get("num_features"),
            "year": doc.metadata.get("year"),
        }
        for doc in result["source_documents"]
    ]
    return {"answer": result["result"], "sources": sources}
