#!/bin/bash
set -e

create_test_directories() {
    mkdir -p /app/data/test /app/.cache/test
    chmod -R 777 /app/data /app/.cache
}

wait_for_chromadb() {
    if [ -z "$CHROMA_HOST" ]; then
        return 0
    fi

    echo "Waiting for ChromaDB at ${CHROMA_HOST}:${CHROMA_PORT}..."

    local max_attempts=15
    local attempt=0
    local wait_seconds=2

    while [ $attempt -lt $max_attempts ]; do
        if python3 -c "import urllib.request; urllib.request.urlopen('http://${CHROMA_HOST}:${CHROMA_PORT}/api/v2/heartbeat', timeout=2.0)" 2>/dev/null; then
            echo "ChromaDB is ready!"
            return 0
        fi

        attempt=$((attempt + 1))
        echo "Attempt $attempt/$max_attempts - ChromaDB not ready yet, waiting ${wait_seconds}s..."
        sleep $wait_seconds
    done

    echo "WARNING: ChromaDB may not be ready after ${max_attempts} attempts, proceeding anyway..."
    return 0
}

main() {
    echo "Initializing test environment..."
    create_test_directories
    wait_for_chromadb
    echo "Starting tests..."
    exec "$@"
}

main "$@"
