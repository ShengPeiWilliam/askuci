# AskUCI

A RAG-powered chatbot that recommends datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) based on natural language queries.

## Demo

Ask questions like:
- *"I need a dataset for image classification"*
- *"Which datasets are good for medical diagnosis prediction?"*
- *"Find me a small tabular dataset for regression with no missing values"*

## Architecture

```
User Query
    │
    ▼
FastAPI /query endpoint (api/main.py)
    │
    ▼
RAG Chain (rag/chain.py)
    ├── Retriever: all-MiniLM-L6-v2 + Chroma (rag/retriever.py)
    │       └── Searches 689 UCI datasets embedded as vectors
    └── Generator: GPT-4o (OpenAI)
            └── Returns natural language recommendations
```

## Project Structure

```
askuci/
├── data/
│   └── uci_datasets.csv          # Scraped UCI dataset metadata
├── scripts/
│   └── scrape_uci.py             # Fetches all datasets from UCI REST API
├── rag/
│   ├── embed.py                  # Embeds dataset descriptions into Chroma
│   ├── retriever.py              # Loads Chroma vectorstore + retriever
│   └── chain.py                  # LangChain LCEL RAG chain
├── api/
│   └── main.py                   # FastAPI app with /query endpoint
├── frontend/
│   └── chatbox.html              # Embeddable chatbox (single HTML file)
├── tests/
│   └── test_queries.py           # Typical query tests
└── chroma_db/                    # Chroma vector DB (gitignored)
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/askuci.git
cd askuci

conda create -n askuci python=3.11 -y
conda activate askuci
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your OpenAI API key
```

`.env`:
```
OPENAI_API_KEY=sk-...
```

### 3. Scrape UCI datasets

```bash
python scripts/scrape_uci.py
# Saves 689 datasets to data/uci_datasets.csv
```

### 4. Embed datasets into Chroma

```bash
python rag/embed.py
# Downloads all-MiniLM-L6-v2 and saves vectors to chroma_db/
```

### 5. Start the API server

```bash
uvicorn api.main:app --reload
# Running at http://localhost:8000
```

### 6. Open the chatbox

Open `frontend/chatbox.html` in your browser. The chatbox appears in the bottom-right corner.

## API

### `POST /query`

**Request:**
```json
{
  "question": "I need a dataset for image classification",
  "k": 5
}
```

**Response:**
```json
{
  "answer": "For image classification, I recommend...",
  "sources": [
    {
      "name": "ImageNet",
      "area": "Computer Science",
      "tasks": "Classification",
      "num_instances": "14000000",
      "num_features": "0.0",
      "year": "2009.0"
    }
  ]
}
```

## Running Tests

```bash
PYTHONPATH=. python tests/test_queries.py
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embedding | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector DB | Chroma |
| LLM | GPT-4o (OpenAI) |
| RAG Framework | LangChain |
| Backend | FastAPI |
| Frontend | HTML / CSS / JS |
