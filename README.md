# AI Learning Examples (Datacom AI Assessment)

A Python-based CLI application and Web Dashboard demonstrating interaction with Large Language Models (LLMs) using the OpenAI API. This project features autonomous agents (Planner, Healer), a RAG pipeline with ChromaDB, and an observability suite using Streamlit.

## 📖 Application Architecture

The application consists of several core components designed to demonstrate modern AI patterns:

* **Interactive CLI**: The primary entry point for chatting with the AI, executing plans, and healing code.
* **Autonomous Agents**:
  * **Planner**: A ReAct-style agent that breaks down travel requests into steps, utilizing tools to fetch flight prices and weather data.
  * **Healer**: A self-correcting agent that generates code, executes it in a local sandbox, captures errors, and iteratively fixes them.
* **RAG Pipeline**: Retrieval-Augmented Generation using **ChromaDB** for vector storage and **Sentence Transformers** for local embeddings. It ingests PDF/Text documents to ground LLM responses.
* **Observability Dashboard**: A Streamlit web application (`app/web/`) that visualizes usage metrics (tokens, cost, latency) and retrieval quality stored in a SQLite database.
* **Persistence**: All chat history and metrics are persisted in `data/chat.db` (SQLite) and `data/chroma_db/` (Vector Store).

## 🛠️ Setup

### Prerequisites

* **Python 3.13+**
* **Docker** & **Docker Compose** (Optional, for containerized execution)

### Environment Configuration

1. Create a `.env` file in the project root.
2. Add your OpenAI credentials and optional configuration:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
```

3. For testing, create a `.env.test` file with the same variables (or safe testing credentials).

## Installation (Local)

1. **Create a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```

## 💻 Usage

### Data Preparation & Ingestion

The application supports two datasets for RAG: a default **Standard Corpus** (`data/corpus`) and a **Large Corpus** (`data/corpus_large`) for performance testing.

1. **Download Large Corpus**:

    To download the extended dataset (>50MB of public domain books):

    ```bash

    python scripts/download_corpus.py

    ```

2. **Ingest Documents**:

    You can ingest documents using the helper script or the interactive CLI.

    * **Via Script**:

        ```bash

        # Ingest Standard Corpus

        python scripts/ingest_corpus.py



        # Ingest Large Corpus (>50MB)

        python scripts/ingest_corpus.py --large

        ```

    * **Via CLI**:

        Inside the application (`python -m app.main`):

        * `/ingest_all` - Ingests the standard corpus.

        * `/ingest_all --large` - Ingests the large corpus (>50MB).

### Running the CLI

Start the interactive chat session:

```bash

python -m app.main

```

**Commands:**

* `/plan <request>` - Ask the Planning Agent to organize a trip (e.g., "Plan a 3-day trip to Tokyo").
* `/heal <task>` - Ask the Healer Agent to write/fix code (e.g., "Write a script to calculate primes").
* `/ingest <path>` - Load a document into the RAG system.
* `/ingest_all` - Load all documents from `data/corpus`.
* `/exit` - Quit the application.

### Running the Dashboard

Launch the Streamlit observability dashboard:

```bash
streamlit run app/web/main.py
```

Open your browser to `http://localhost:8501`.

## 🐳 Docker Support

You can spin up all services using Docker Compose.

### 1. Run the Web Dashboard

This starts the Streamlit dashboard and the ChromaDB server.

```bash
docker-compose up -d ai-web
```

Access at: `http://localhost:8501`

### 2. Run the CLI

To run the interactive CLI within a container:

```bash
docker-compose --profile cli run --rm ai-cli
```

## 🧪 Testing

### Running Tests Locally

The project uses `pytest`.

**Note**: Before running tests locally, ensure that the Chroma DB service is running:

```bash

docker-compose up -d chromadb

```

* **Run all tests** (Requires API Key):

    ```bash
    python -m pytest
    ```

* **Run unit tests only** (No API cost):

    ```bash
    python -m pytest -m "not integration"
    ```

* **Run integration tests only**:

    ```bash
    python -m pytest -m integration
    ```

### Running Tests in Docker

To run tests inside a clean container environment:

```bash
docker-compose --profile test run --rm ai-cli-test
```

## 📂 Development

* **Linting**: Run `ruff check .` to ensure code quality.
* **Type Checking**: This project uses type hints extensively. Run `mypy .` to check
* **Logs**: Application logs are stored in `data/app.log`.
