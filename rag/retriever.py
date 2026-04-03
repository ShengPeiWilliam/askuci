"""
Step 2 — Build retriever from persisted Chroma vectorstore.
"""

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "uci_datasets"

# Module-level cache — model loads once per process
_embeddings = None
_vectorstore = None


def _get_vectorstore() -> Chroma:
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=_embeddings,
            collection_name=COLLECTION_NAME,
        )
    return _vectorstore


def get_retriever(k: int = 5):
    return _get_vectorstore().as_retriever(search_kwargs={"k": k})
