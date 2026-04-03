"""
Step 2 — Embed all dataset descriptions and store in Chroma.
Usage: python rag/embed.py
"""

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

load_dotenv()

CSV_PATH = "data/uci_datasets.csv"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "uci_datasets"


def build_document(row: dict) -> Document:
    text = (
        f"Name: {row['name']}\n"
        f"Abstract: {row['abstract']}\n"
        f"Area: {row['area']}\n"
        f"Tasks: {row['tasks']}\n"
        f"Instances: {row['num_instances']}\n"
        f"Features: {row['num_features']}\n"
        f"Missing values: {row['has_missing_values']}\n"
        f"Year: {row['year_of_dataset_creation']}"
    )
    metadata = {
        "id": str(row["id"]),
        "name": str(row["name"]),
        "area": str(row["area"]),
        "tasks": str(row["tasks"]),
        "num_instances": str(row["num_instances"]),
        "num_features": str(row["num_features"]),
        "has_missing_values": str(row["has_missing_values"]),
        "year": str(row["year_of_dataset_creation"]),
    }
    return Document(page_content=text, metadata=metadata)


def main():
    df = pd.read_csv(CSV_PATH).fillna("")
    print(f"Loaded {len(df)} datasets from {CSV_PATH}")

    docs = [build_document(row) for _, row in df.iterrows()]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"Embedded {len(docs)} documents → saved to {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
