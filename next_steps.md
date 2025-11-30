## Pending Next Steps

1. **Code Hygiene & Modernization**
   - **Task:** Refactor project structure and consolidate configuration.
   - **Detail:** Move to a `src/` layout and migrate `pytest.ini`, `mypy.ini`, `requirements.txt` into a single `pyproject.toml`.
   - **Why:** Reduces clutter, standardizes tooling configuration, and improves build reproducibility.

2. **Test Suite Overhaul & E2E (Playwright)**
   - **Task:** Refactor existing tests and implement a tiered testing strategy.
   - **Detail:** Clean up current tests, restructure folders to clearly separate Unit, Integration, and E2E tests, and introduce **Playwright** for end-to-end CLI/Web testing.
   - **Why:** Ensures comprehensive coverage and reliable regressions testing across the entire stack, not just python functions.

3. **Safety: HITL & Guardrails**
   - **Task:** Implement a "Suspension" layer for human approval and Input/Output Guardrails.
   - **Detail:** Pause for user confirmation before critical actions (billing/execution) and use **NeMo/Llama Guard** to filter prompt injections and PII leakage.
   - **Why:** Prevents "Auto-Pilot" disasters, security jailbreaks, and sensitive data exposure.

4. **Reliability: Basic Hygiene (Retries & Token Management)**
   - **Task:** Implement `tenacity` for API retries and `tiktoken` for accurate context window management (evicting old messages based on token count, not message count).
   - **Why:** Prevents the app from crashing due to transient network blips or "Context Length Exceeded" errors.

5. **Architecture: Async Worker Pattern**
   - **Task:** Refactor the CLI/Web to submit "Jobs" to a Redis Queue, and implement a Worker process to execute the Agent logic.
   - **Why:** Decouples the UI from the heavy processing, fixing the "Frozen UI" issue and enabling concurrent users.

6. **Data: Unified Postgres (Standard & Vector)**
   - **Task:** Migrate from SQLite/Chroma to **PostgreSQL + pgvector**.
   - **Why:** Simplifies the stack (one DB for everything) and handles concurrency for the Async Worker pattern.

7. **Interface: FastAPI & Streaming**
   - **Task:** Wrap core logic in a REST API with **Streaming-First** architecture.
   - **Detail:** Expose endpoints that stream LLM tokens and execution logs in real-time to the client.
   - **Why:** Decouples the frontend and solves the "Black Box" waiting experience by giving immediate feedback.

8. **RAG Evolution: Hybrid Search & Web Agents**
   - **Task:** Implement Hybrid Search (Keyword + Vector) and a **Web Search** tool.
   - **Detail:** Combine keyword matching for precision with semantic search for understanding, and allow the agent to fetch live data (e.g., via Tavily) for up-to-date answers.
   - **Why:** Moves beyond static local files to a "Search-First" architecture that can answer current events.

9. **Critical Security: Containerized Sandboxing**
   - **Task:** Isolate the `Healer` agent so it runs code in a disposable Docker container, not the host OS.
   - **Why:** Prevents the AI from accidentally (or intentionally) wiping files or stealing API keys.

10. **Quality: Evaluation-Driven Development (EDD)**
    - **Task:** Implement an automated test suite where an LLM judges the quality of RAG retrieval and generated answers.
    - **Why:** Moves us from "Vibes-based testing" to systematic quality engineering.

11. **Optimization: DSPy & Semantic Caching**
    - **Task:** Adopt **DSPy** for prompt optimization and **Semantic Caching** (Redis/GPTCache).
    - **Detail:** "Compile" optimal prompts automatically using DSPy, and cache high-similarity queries to return instant, zero-cost responses.
    - **Why:** Reduces token costs, lowers latency, and eliminates brittle manual prompt engineering.
