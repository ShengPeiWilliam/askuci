# AskUCI
RAG-powered chatbot that recommends datasets from the UCI Machine Learning Repository based on natural language queries.

## Motivation

Taking statistics courses meant constantly searching for datasets, filtering by task type, scanning abstracts, and hoping something fit. The process worked, but it was slow and often missed relevant results. I wanted a tool where I could just describe what I needed and get the closest matches back instantly.

## Design Decisions

The core idea: embed all 689 dataset descriptions from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) into a vector space, then retrieve the closest matches to your query. No LLM needed for the retrieval itself, the semantic similarity does the work.


**Why `all-MiniLM-L6-v2` instead of OpenAI embeddings?**

It runs locally, costs nothing, and is fast enough to embed all 689 datasets in under a minute, with negligible quality difference vs. hosted models for short text like dataset abstracts.

**Why return 5 results?**

Enough to surface variety without overwhelming the user. In practice, the top 1–2 results are almost always relevant; results 3–5 provide useful alternatives when the query is ambiguous.

**Why a guard layer?**

Without it, the retriever returns results for any query. The guard (`gpt-4o-mini`) checks intent before hitting the vector DB and returns a refusal message for off-topic queries, keeping cost low since it only classifies intent, not generates content.

## Demo

*Dataset Recommendation*

<img src="figures/chatbox_query.png" width="480">

*Guard (Off-topic Query)*

<img src="figures/chatbox_guard.png" width="480">

## Reflections & Next Steps

Embedding similarity works well for this use case since dataset descriptions are short and consistent. The main gap is the guard, which occasionally misclassifies broad queries ("what datasets exist?") as off-topic.

If I were to continue:
- **From retrieval to training**: AskUCI relies entirely on pretrained embeddings. A natural evolution would be training task-specific embeddings that learn user preferences from interaction data, similar to the approach explored in [movierec-two-towers](https://github.com/ShengPeiWilliam/movierec-two-towers).

- **Reranking**: use a cross-encoder to score top-k results against the query, which would help with queries that match on surface keywords but not actual intent.

- **Attribute filtering**: let users narrow by dataset size, feature types, or task type before or after retrieval.

## References

- [UCI ML Repository](https://archive.ics.uci.edu/) — dataset source, 689 datasets scraped for embedding.
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — embedding model chosen for local inference and zero API cost.
- [Chroma](https://www.trychroma.com/) — vector store for semantic retrieval.
- [LangChain](https://python.langchain.com/) — orchestration layer for the RAG chain.
