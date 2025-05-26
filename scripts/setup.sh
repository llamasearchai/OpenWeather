#!/bin/bash
# Setup environment and install dependencies

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install UV for faster package installation
pip install uv

# Install project dependencies using UV
uv pip install -e .

# Install LLM plugins and tools
uv pip install llm-cli datasette datasette-llm-embed

# Initialize SQLite database for caching
sqlite-utils create-database data/weather_cache.db

# Set up LLM configuration
llm keys set openai
llm keys set huggingface

echo "Setup complete. Activate virtual environment with: source .venv/bin/activate" 