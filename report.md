# Design Decisions and Trade-offs

A lot of the decisions boil down to time constraints and focusing on getting the logic working first, further exacerbated by my unfamiliarity with specific RAG tooling. I've created a list of steps that I intended to do next [next_steps.md](next_steps.md).

## Design Decisions & Trade-offs

### 1. CLI-First Approach

I built the project primarily as an interactive CLI tool, with the Web Dashboard strictly for observability, rather than starting as a standard Web API (FastAPI).
**Why:** Time constraints and a deliberate choice to focus on mastering the core Agentic logic (State loops, Tool calling, RAG) without the immediate overhead of HTTP layers, request serialization, and frontend state management.
**Trade-offs:** The application logic is currently coupled to the CLI interaction loop. It is not immediately consumable by other clients (like a mobile app or Slack bot).
**Future Path:** The clear next step is to wrap the core service layer in a **FastAPI** application. The CLI would then become just one of many lightweight clients consuming the API.

### 2. Synchronous Code

Synchronous Python code instead of `asyncio`.
**Why:** I've had issues with weird asyncio issues before and didn't want to get distracted debugging those issues until I finished the logic. I also didn't know what libraries I'd be using and if they were compatible.
**Trade-offs:** It is the main bottleneck. The app handles requests sequentially.
**Future Path:** "Async by default" would be the standard for production, especially for an app waiting on external IO (LLMs/DBs/Human prompts). This ties into the "Async Worker" pattern mentioned below.

### 3. Local Infrastructure

SQLite for the database and `sentence-transformers` (CPU) + ChromaDB (Local File) for the RAG pipeline.
**Why:** Easy to setup and it ensures the project runs on a standard laptop with zero recurring cloud costs or complex setup.
**Trade-offs:** SQLite struggles with concurrency (database locking) when multiple writes happen. Not horizontally scalable. Local embeddings consume application RAM.
**Future Path:** I would move to **PostgreSQL**. With the `pgvector` extension, it provides a single, robust solution for both relational data and vector embeddings, solving the concurrency and scale issues.

### 4. Manual Chunking vs. Frameworks

I used a custom manual script for text chunking rather than adopting a framework like LangChain's text splitters.
**Why:** A combination of factors: I was less familiar with the framework-specific abstractions, ran into some initial environment installation issues, and—most importantly—my benchmarks showed no reasonable improvement in retrieval metrics (Recall/Precision) for this specific corpus when using the more complex splitters.
**Trade-offs:** Manual chunking is brittle and requires maintenance. It lacks sophisticated features like semantic splitting or recursive character handling.
**Future Path:** Adopt standard libraries (like `langchain-text-splitters`) to reduce code maintenance and handle edge cases in document formatting better.

### 5. Strict Typing & Pydantic (Reliability over Velocity)

I used Pydantic models and Protocol interfaces extensively throughout the core logic.
**Why:** In the chaotic world of LLM string outputs, Pydantic brings structure. It enforces schemas for API responses and configuration, catching errors early.
**Trade-offs:** It requires more boilerplate code than using simple Python dictionaries.
**Future Path:** This puts us in a great position to adopt frameworks like **PydanticAI**, which treat the LLM as a function that returns a Pydantic model.

### 6. The "Healer" Sandbox Compromise

The "Healer" agent writes code and runs it directly on the host machine via `subprocess`.
**Why:** It allows for a demo of self-correction without requiring the user to install Docker.
**Trade-offs:** **Security Risk.** The AI has full access to the user's file system and network.
**Future Path:** This is the critical blocker for public deployment. The execution must be wrapped in a secure sandbox (like a Docker container or Firecracker micro-VM) with no network access.

### 7. Naive RAG vs. Agentic RAG

Linear RAG pipeline where every query searches the database.
**Why:** It guarantees the model always has context, which reduces hallucinations for simple Q&A.
**Trade-off/Issues:** It's inefficient for questions that don't need docs ("Hi there") or complex questions requiring multi-step research.
**For Production:** I would move to **Agentic RAG**, treating "Search" as a tool the agent can choose to use. Consider also adopting MCP to standardize how the agent connects to these data sources, rather than writing custom tool code.
