from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rag.retriever import get_retriever

load_dotenv()

_retriever = None
_guard_llm = None


def _parse_abstract(page_content: str) -> str:
    for line in page_content.splitlines():
        if line.startswith("Abstract:"):
            return line.replace("Abstract:", "").strip()
    return ""


def _get_retriever(k: int = 5):
    global _retriever
    if _retriever is None:
        _retriever = get_retriever(k=k)
    return _retriever


def _is_dataset_query(question: str) -> bool:
    global _guard_llm
    if _guard_llm is None:
        _guard_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Is the following question asking about finding, recommending, or learning about machine learning datasets?
Answer only "yes" or "no".

Question: {question}"""
    result = _guard_llm.invoke(prompt).content.strip().lower()
    return result.startswith("yes")


def query(question: str, k: int = 5) -> dict:
    if not _is_dataset_query(question):
        return {
            "answer": "I can only help with finding UCI datasets. Please ask me about a dataset you need!",
            "sources": [],
        }

    retriever = _get_retriever(k=k)
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
    return {"answer": "", "sources": sources}
