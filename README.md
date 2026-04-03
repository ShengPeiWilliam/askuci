# AskUCI

A RAG-powered chatbot that recommends datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) based on natural language queries.

## Setup

```bash
git clone https://github.com/your-username/askuci.git
cd askuci

conda create -n askuci python=3.11 -y
conda activate askuci
pip install -r requirements.txt
```

Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=sk-...
```

Then run:
```bash
python scripts/scrape_uci.py   # fetch dataset metadata
python rag/embed.py            # embed into Chroma
uvicorn api.main:app --reload  # start API server
```

Open `frontend/chatbox.html` in your browser.

## Customization

| What | Where | Default |
|------|-------|---------|
| Datasets retrieved per query | `api/main.py` → `k` | `5` |
| Recommendations shown | `rag/chain.py` → prompt `"top 3"` | `3` |
| Embedding model | `rag/embed.py` + `rag/retriever.py` | `all-MiniLM-L6-v2` |
| LLM model | `rag/chain.py` → `ChatOpenAI(model=...)` | `gpt-4o` |

> If you change the embedding model, re-run `python rag/embed.py` to rebuild the vector DB.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embedding | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector DB | Chroma |
| LLM | GPT-4o (OpenAI) |
| RAG Framework | LangChain |
| Backend | FastAPI |
| Frontend | HTML / CSS / JS |
