#!/bin/bash
set -e

create_host_directories() {
    mkdir -p data/corpus data/corpus_large data/chroma_db data/chroma_server
    mkdir -p .cache/huggingface .cache/sentence_transformers .cache/torch
    mkdir -p .streamlit
}

set_permissive_permissions() {
    chmod -R 777 data .cache .streamlit
}

main() {
    echo "Setting up Docker environment..."
    create_host_directories
    set_permissive_permissions
    echo "Docker environment ready!"
    echo "You can now run: docker-compose up -d ai-web"
}

main
