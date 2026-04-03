"""
Step 2 — Build retriever from persisted Chroma vectorstore.
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "uci_datasets"


def get_retriever(k: int = 5):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
