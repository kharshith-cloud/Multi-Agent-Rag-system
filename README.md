# Multi-Agent RAG System

A multi-agent Retrieval-Augmented Generation (RAG) system that answers questions from uploaded PDF documents using hybrid retrieval, LLM-based research, and automated answer verification.

## System Architecture

The system combines a dual-engine hybrid retrieval layer with an autonomous multi-agent state graph orchestrated via **LangGraph**. Rather than relying on a single unverified LLM response, queries route through specialized agent nodes to ensure document relevance, factual grounding, and quality control.

- **Document Ingestion & Indexing**: PDF files are loaded using LlamaIndex `SimpleDirectoryReader`, split into nodes using `SentenceSplitter`, embedded locally with HuggingFace `BAAI/bge-base-en-v1.5`, and persisted to disk.
- **Hybrid Retrieval**: `LlamaIndexHybridRetriever` executes parallel queries over a dense vector index (`VectorIndexRetriever`) and a sparse lexical index (`BM25Retriever`), merging and deduplicating top results into LangChain Document objects.
- **Relevance Checker**: Evaluates top retrieved document chunks to classify whether the query can be answered (`CAN_ANSWER`, `PARTIAL`, or `NO_MATCH`) before wasting tokens on synthesis.
- **Research Agent**: Synthesizes factual, context-grounded answers strictly from retrieved documents and incorporates structured feedback during refinement loops.
- **Verification Agent**: Fact-checks draft answers against source context to identify unsupported claims, contradictions, and overall relevance.
- **Iterative Refinement**: When verification identifies gaps or unsupported claims, the workflow passes structured feedback back to the Research Agent for targeted revision (capped at 2 iterations to prevent infinite loops).
- **Streamlit Interface**: Provides a user dashboard with interactive chat, batch document upload, and toggles for Fast vs. Verification execution modes.

### Workflow & Data Flow Diagram

```text
               ┌──────────────────────────────────────────────┐
               │              PDF Document Upload             │
               └──────────────────────┬───────────────────────┘
                                      ▼
                       Document Ingestion & Chunking
                        (SimpleDirectoryReader & Splitter)
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                  Vector Index                BM25 Index
               (BAAI/bge-base-en-v1.5)        (Lexical Store)
                         └────────────┬────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │  Hybrid Retriever Engine │
                        └─────────────┬────────────┘
                                      │
User Question ────────────────────────┤ (Query + Document Retrieval)
                                      ▼
                           Relevance Checker Node
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                    [NO_MATCH]            [CAN_ANSWER / PARTIAL]
                         │                         │
                         ▼                         ▼
                   Return Warning            Research Agent Node
                                                   │
                                                   ▼
                                              Draft Answer
                                                   │
                                                   ▼
                                        Verification Agent Node
                                                   │
                                      ┌────────────┴────────────┐
                                      ▼                         ▼
                                 [Verified]             [Needs Refinement]
                                      │                         │
                                      ▼                         ▼
                                 Final Answer ◄────── Retry with Feedback
                                                      (Max 2 Iterations)
```

## Features

- **PDF Document Ingestion**: Supports batch uploading and chunking of single or multi-page PDF documents.
- **Hybrid Retrieval**: Dual-engine retrieval merging dense semantic embeddings (`BAAI/bge-base-en-v1.5`) and sparse keyword search (`BM25`).
- **Multi-Agent Orchestration**: LangGraph `StateGraph` workflow coordinating specialized agent nodes with conditional state routing.
- **Relevance Assessment**: Pre-evaluates retrieved document chunks to filter out irrelevant or unanswerable queries.
- **Automated Answer Verification**: Fact-checks draft answers against source context, flagging unsupported claims or contradictions.
- **Iterative Refinement Loop**: Feedback loop allowing the Research Agent to self-correct draft answers based on verification reports.
- **Qualitative Execution Modes**:
  - **Fast Mode**: Direct research synthesis optimized for lower-latency responses.
  - **Verification Mode**: Full multi-agent pipeline with relevance scoring and structured verification reports.
- **Interactive Web Interface**: Streamlit UI featuring chat history, upload progress tracking, and expandable verification details.

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | Streamlit |
| **Agent Orchestration** | LangGraph |
| **Retrieval & Indexing** | LlamaIndex, BM25Retriever |
| **Embeddings** | HuggingFace (`BAAI/bge-base-en-v1.5`) |
| **LLM Provider** | Groq (`openai/gpt-oss-20b`) |
| **Document Processing** | PyPDF, SentenceSplitter |
| **Language** | Python 3.10+ |

## Core Components

| Component | Responsibility |
|---|---|
| `app.py` | Streamlit user interface, session state management, and mode toggles |
| `config.py` | Directory paths, chunking settings, and environment configuration |
| `ingest.py` | PDF document loading, text splitting, batch node insertion, and vector index persistence |
| `retriever.py` | Hybrid retriever engine merging dense vector and sparse BM25 search results |
| `agents/relevance_checker.py` | Assesses if retrieved document passages contain sufficient details to answer the query |
| `agents/research_agent.py` | Synthesizes context-grounded answers and incorporates retry feedback |
| `agents/verification_agent.py` | Fact-checks draft answers against context for factual support, contradictions, and relevance |
| `agents/workflow.py` | Coordinates the multi-agent LangGraph workflow, state routing, and iteration controls |

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Edit `.env` and set your Groq API key:

```ini
GROQ_API_KEY=your_actual_api_key_here
```

Get an API key from [Groq Console](https://console.groq.com/keys).

### 3. Run the Application

```bash
streamlit run app.py
```

## Usage

1. **Upload Documents**: Browse and select one or more PDF files.
2. **Index Documents**: Click **📑 Index PDFs** to build and persist the vector and BM25 indexes.
3. **Select Execution Mode**: Choose between **Fast Mode** (quick answer generation) or **Verification Mode** (full multi-agent verification).
4. **Ask Questions**: Type your question into the chat input.
5. **Inspect Verification**: In Verification Mode, expand the **🔍 Verification Report** below any response to review factual support, unsupported claims, contradictions, and relevance evaluations.