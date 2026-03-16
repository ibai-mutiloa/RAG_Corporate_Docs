# RAG Corporate Docs

> Production-grade Retrieval-Augmented Generation pipeline for indexing and querying internal corporate documents using vector search and LLM integration.

Built and deployed at **Mondragon Unibertsitatea**, used daily by 200+ staff members to query internal regulations, policies and procedures in natural language — without reading hundreds of pages of PDFs.

---

## What it does

Employees type a question like *"What is the procedure to request a leave of absence?"* and get a direct, contextual answer grounded in the official documents — not a list of links, not a hallucination.

The system ingests PDFs, splits them into semantically meaningful chunks, embeds them, stores them in PostgreSQL with `pgvector`, and at query time retrieves the most relevant chunks to ground the LLM response.

```
PDF documents
      │
      ▼
  Text extraction + chunking
      │
      ▼
  Embedding generation (Azure OpenAI)
      │
      ▼
  pgvector (PostgreSQL)
      │
      ▼
  Semantic retrieval at query time
      │
      ▼
  LLM response synthesis with context grounding
      │
      ▼
  Answer via FastAPI endpoint
```

---

## Architecture

![System architecture](/intraneta.eps.mondragon.edu.png)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Ingestion pipeline                                                          │
│                                                                              │
│  Windows file server ──SMB──► Local folder ──► Chunker + Embedder           │
│       (IIS/MGEP)                                       │                    │
│                                                        ▼                    │
│                                               PostgreSQL + pgvector          │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  Query pipeline                                                              │
│                                                                              │
│  MGEP Intranet                                                               │
│  (embedded chatbot) ◄────────────────────────────────────────────┐          │
│         │                                                         │          │
│         │ user question                                  grounded │          │
│         ▼                                                response  │          │
│  Embed query ──► pgvector similarity search                        │          │
│                         │                                         │          │
│                         │ top-K chunks                            │          │
│                         ▼                                         │          │
│                  Azure OpenAI (GPT) ──────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### How documents reach the system

Documents live on a **Windows file server** (the existing university infrastructure). The ingestion pipeline accesses them over **SMB**, so there is no need to migrate or duplicate files — the source of truth stays where it already is.

### Key design decisions

- **SMB-based ingestion** — no need to change existing document workflows. Files stay on the university's Windows server; the pipeline reads them directly over the network share.
- **pgvector over dedicated vector DBs** — keeps the stack simple (one PostgreSQL instance), reduces operational overhead, and performs well at this scale.
- **Chunking strategy** — documents are split respecting section boundaries, not just fixed token counts. This preserves semantic coherence per chunk and improves retrieval precision.
- **Context grounding** — the LLM only uses retrieved chunks as context. No parametric knowledge for factual answers, reducing hallucination risk on regulatory content.
- **Embedded chatbot** — the assistant is integrated directly into the existing MGEP intranet UI, so users never leave the platform they already use daily.

---

## Tech stack

| Layer | Technology |
|---|---|
| Ingestion | Python, PyMuPDF |
| Embedding | Azure OpenAI (`text-embedding-ada-002`) |
| Vector store | PostgreSQL + pgvector |
| LLM | Azure OpenAI (GPT-4) |
| API | FastAPI |
| Deployment | Docker, Azure |

---

## Project structure

```
RAG_Corporate_Docs/
├── ingestion/
│   ├── extract.py          # PDF text extraction
│   ├── chunker.py          # Semantic chunking logic
│   └── embedder.py         # Embedding generation + pgvector insert
├── api/
│   ├── main.py             # FastAPI app + query endpoint
│   ├── retriever.py        # Similarity search against pgvector
│   └── synthesizer.py      # LLM prompt construction + response
├── db/
│   └── schema.sql          # PostgreSQL schema with pgvector extension
├── docker/
│   └── Dockerfile
├── .env.example
└── requirements.txt
```

---

## Getting started

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector) extension
- Azure OpenAI API access (embedding model + GPT-4 deployment)
- Docker (optional)

### Setup

```bash
# Clone the repo
git clone https://github.com/ibai-mutiloa/RAG_Corporate_Docs.git
cd RAG_Corporate_Docs

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Azure OpenAI keys and PostgreSQL connection string
```

### Database setup

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Run the schema
\i db/schema.sql
```

### Ingest documents

```bash
# Place your PDFs in /docs and run the ingestion pipeline
python ingestion/extract.py --input ./docs --output ./chunks
python ingestion/embedder.py --input ./chunks
```

### Run the API

```bash
uvicorn api.main:app --reload
```

Query the endpoint:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the procedure to request a leave of absence?"}'
```

### Docker

```bash
docker build -t rag-corporate-docs .
docker run -p 8000:8000 --env-file .env rag-corporate-docs
```

---

## Environment variables

```env
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_db
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
```

---

## Production context

This system is deployed on the intranet of **Mondragon Unibertsitatea** and used daily by 200+ employees. It replaced a manual process of manually searching through PDF documents for regulatory and administrative information — internal regulations, HR policies, academic procedures.

A separate system handles a different use case: a corporate telephony assistant built on **Azure AI Search** that lets staff query mobile phone contracts in natural language (roaming conditions, travel zones, usage abroad). Same RAG pattern, different stack and document domain — see [rag_using_azure](https://github.com/ibai-mutiloa/rag_using_azure).

---

## Roadmap / ideas

- [ ] Hybrid retrieval: combine BM25 keyword search with semantic search for better recall on exact terms
- [ ] Re-ranking layer: add a cross-encoder to re-rank top-K results before synthesis
- [ ] Streaming responses via SSE for better UX on longer answers
- [ ] Evaluation harness: automated RAG evaluation with precision/recall metrics
- [ ] Multi-document collections: namespace support for querying specific document sets

---

## Related projects

- **[rag_using_azure](https://github.com/ibai-mutiloa/rag_using_azure)** — Azure AI Search variant of this pipeline
- **[ibaimutiloa.es](https://ibaimutiloa.vercel.app/)** — Portfolio with full project writeup

---

## Author

**Ibai Mutiloa Aliaga** — Backend Engineer · AI Systems · Cloud Infrastructure

[LinkedIn](https://linkedin.com/in/ibai-mutiloa-aliaga) · [Portfolio](https://ibaimutiloa.vercel.app/) · [GitHub](https://github.com/ibai-mutiloa)
