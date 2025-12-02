#!/bin/bash
set -e

create_application_directories() {
    mkdir -p /app/data/corpus \
             /app/data/corpus_large \
             /app/data/chroma_db \
             /app/.cache/huggingface \
             /app/.cache/sentence_transformers \
             /app/.cache/torch \
             /home/appuser/.streamlit
}

fix_directory_permissions() {
    chown -R appuser:appuser /app/data /app/.cache /home/appuser/.streamlit 2>/dev/null || true
    chmod -R u+rw /app/data /app/.cache 2>/dev/null || true
}

main() {
    echo "Initializing application directories..."
    create_application_directories
    fix_directory_permissions
    echo "Directory initialization complete"
    exec "$@"
}

main "$@"
