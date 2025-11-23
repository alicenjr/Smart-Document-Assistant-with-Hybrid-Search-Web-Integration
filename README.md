# Agentic RAG System: Intelligent Document Q&A with Multi-Modal Processing

> A comprehensive Retrieval-Augmented Generation (RAG) system built with LangChain that combines document ingestion, vector search, and agentic workflows to provide intelligent question-answering capabilities over PDF documents.

**Multi-Modal RAG • Hybrid Search • Web-Augmented Intelligence • Agentic Workflows**

## 🎯 Features

- **PDF Document Ingestion**: Upload and process PDFs with support for:
  - Semantic text chunking
  - Image extraction and captioning (using Gemini Vision)
  - Table extraction and description
  - Multi-modal content indexing

- **Hybrid Search**: Combines keyword and semantic (vector) search for optimal retrieval
- **Agentic RAG Workflow**: Multi-step workflow with query enhancement, retrieval, summarization, and quality rating
- **Web Search Integration**: Augments document knowledge with real-time web search via Serper API
- **Conversation Memory**: Maintains context across multiple interactions
- **Modern Chat UI**: React-based frontend with dark mode support
- **OpenSearch Backend**: Scalable vector database for document storage

## 📋 Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Development](#development)

## 🏗️ Architecture

The system consists of three main components:

1. **Backend (FastAPI)**: Handles PDF ingestion, document processing, and RAG query execution
2. **Frontend (React/TypeScript)**: Modern chat interface for user interactions
3. **Infrastructure**: OpenSearch for vector storage, Ollama for embeddings

### Workflow Overview

```
PDF Upload → Partitioning → Chunking → Embedding → OpenSearch Index
                                                          ↓
User Query → Query Enhancement → Hybrid Search → Retrieval
                                                          ↓
                    Web Search (Serper) ← → Document Retrieval
                                                          ↓
                    Summarization → Merging → Quality Rating → Response
```

## 📦 Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for frontend)
- **Docker & Docker Compose** (for OpenSearch)
- **Ollama** (running locally on port 11434 with `nomic-embed-text` model)
- **API Keys**:
  - Google Gemini API key (for LLM and vision)
  - Serper API key (optional, for web search)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd langchain_tools
```

### 2. Set Up OpenSearch

Start OpenSearch and OpenSearch Dashboards using Docker Compose:

```bash
docker-compose up -d
```

This will start:
- OpenSearch on `http://localhost:9200`
- OpenSearch Dashboards on `http://localhost:5601`

### 3. Set Up Ollama

Install and run Ollama, then pull the embedding model:

```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai for installation instructions

# Pull the embedding model
ollama pull nomic-embed-text
```

Ensure Ollama is running on `http://localhost:11434`

### 4. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Additional dependencies may be needed:
pip install unstructured[pdf] langchain-google-genai langgraph opensearch-py python-dotenv
```

### 5. Install Frontend Dependencies

```bash
cd chatbot-ui
npm install
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory (optional, as some keys are hardcoded in the code):

```env
# Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key

# Serper API Key (for web search)
SERPER_API_KEY=your_serper_api_key

# PDF Upload Directory
PDF_UPLOAD_DIR=uploads

# OpenSearch Index Name
PDF_INDEX_NAME=pdf_content_index
```

**⚠️ Security Note**: The codebase currently contains hardcoded API keys. For production use, move all API keys to environment variables or a secure configuration system.

### Frontend Configuration

Create a `.env` file in the `chatbot-ui` directory:

```env
VITE_API_BASE=http://localhost:8000
```

## 🎮 Usage

### Start the Backend Server

```bash
# From the root directory
uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Start the Frontend

```bash
# From the chatbot-ui directory
cd chatbot-ui
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the port shown in the terminal)

### Using the System

1. **Upload a PDF**: Use the upload interface in the chat UI to add documents to the knowledge base
2. **Ask Questions**: Type your question in the chat interface
3. **View Responses**: The system will retrieve relevant information from your documents and web search, then provide a consolidated answer

## 📡 API Endpoints

### Health Check

```http
GET /health
```

Returns the health status of the service.

### Ingest PDF

```http
POST /ingest
Content-Type: multipart/form-data

file: <PDF file>
```

Uploads and processes a PDF file. The ingestion runs in the background.

**Response:**
```json
{
  "message": "Ingestion started",
  "original_filename": "document.pdf",
  "stored_filename": "unique_hash_document.pdf",
  "index": "pdf_content_index"
}
```

### Query RAG

```http
POST /query
Content-Type: application/json

{
  "query": "Your question here",
  "conversation_id": "optional-conversation-id"
}
```

Executes the agentic RAG workflow and returns a response.

**Response:**
```json
{
  "query": "enhanced query",
  "rag_answer": "retrieved chunks...",
  "google_answer": "web search results...",
  "r_summary": "document summary",
  "g_summary": "web search summary",
  "r_g_summary": "combined final answer",
  "rating": "approved",
  "conversation_id": "uuid"
}
```

### Get Conversation

```http
GET /conversations/{conversation_id}
```

Retrieves the conversation history for a given conversation ID.

### Delete Conversation

```http
DELETE /conversations/{conversation_id}
```

Deletes a conversation by its ID.

## 📁 Project Structure

```
langchain_tools/
├── app.py                 # FastAPI application with endpoints
├── workflow_2.py          # Agentic RAG workflow implementation
├── agent.py               # LangChain agent with tools
├── ingestion.py           # OpenSearch ingestion pipeline
├── retrieval.py           # Search functions (keyword, semantic, hybrid)
├── generation.py          # RAG response generation
├── chunker.py             # PDF chunking and processing
├── helper.py              # Utility functions (embeddings, OpenSearch client)
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # OpenSearch infrastructure
├── workflow_2.ipynb       # Jupyter notebook version of workflow
├── display_workflow_2.ipynb  # Workflow visualization
├── uploads/               # Temporary PDF storage
└── chatbot-ui/           # React frontend
    ├── src/
    │   ├── pages/chat/   # Chat interface
    │   ├── components/   # UI components
    │   └── ...
    └── package.json
```

## 🔧 Key Components

### PDF Processing Pipeline

1. **Partitioning**: Uses `unstructured` library to extract text, images, and tables
2. **Image Processing**: Extracts images, generates captions using Gemini Vision
3. **Table Processing**: Extracts tables and generates descriptions
4. **Semantic Chunking**: Creates meaningful text chunks based on document structure
5. **Embedding**: Generates 768-dimensional embeddings using Ollama's `nomic-embed-text`
6. **Indexing**: Stores all content in OpenSearch with vector search capabilities

### Agentic RAG Workflow

The workflow (`workflow_2.py`) implements a multi-step process:

1. **Query Enhancement**: Improves user queries for better retrieval
2. **Parallel Retrieval**: 
   - Document retrieval from OpenSearch (hybrid search)
   - Web search via Serper API
3. **Summarization**: Creates concise summaries of both sources
4. **Merging**: Combines document and web search summaries
5. **Quality Rating**: Evaluates answer quality (approved/rejected)
6. **Feedback Loop**: Re-retrieves if quality is rejected

### Search Methods

- **Keyword Search**: Traditional text matching
- **Semantic Search**: Vector similarity search using embeddings
- **Hybrid Search**: Combines both methods for optimal results

## 🛠️ Development

### Running Tests

```bash
# Test retrieval
python retrieval.py

# Test generation
python generation.py

# Test workflow
python workflow_2.py "your test query"
```

### Adding New Features

1. **New Tools**: Add to `agent.py` in the tools list
2. **New Workflow Nodes**: Add functions to `workflow_2.py` and wire them in the graph
3. **New Search Methods**: Extend `retrieval.py` with new search functions

### Frontend Development

```bash
cd chatbot-ui
npm run dev      # Development server
npm run build    # Production build
npm run lint     # Lint code
```

## 🔒 Security Considerations

- **API Keys**: Currently hardcoded in several files. Move to environment variables for production
- **CORS**: Currently allows all origins. Restrict in production
- **File Uploads**: Validate file types and sizes
- **OpenSearch**: Currently runs without security. Enable authentication for production

## 📝 Notes

- The system uses multiple Gemini API keys (hardcoded) for different LLM instances
- OpenSearch index is recreated on each ingestion (existing index is deleted)
- Conversation memory is stored in-memory (not persistent)
- PDF files are deleted after ingestion
- Smalltalk detection is implemented to handle casual greetings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- LangChain for the RAG framework
- OpenSearch for vector storage
- Unstructured for PDF processing
- Google Gemini for LLM capabilities
- Serper for web search API

