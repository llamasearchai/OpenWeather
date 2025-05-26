#!/bin/bash
# Activate the virtual environment and set up environment variables

source .venv/bin/activate

# Set default environment variables if not already set
export OPENWEATHER_DB_PATH=${OPENWEATHER_DB_PATH:-"data/weather_cache.db"}
export OPENWEATHER_LOG_LEVEL=${OPENWEATHER_LOG_LEVEL:-"INFO"}

# Check for required API keys
if [ -z "$OPENAI_API_KEY" ] && [ -f ~/.llm/keys.json ]; then
    export OPENAI_API_KEY=$(jq -r '.openai' ~/.llm/keys.json)
fi

if [ -z "$HF_API_KEY" ] && [ -f ~/.llm/keys.json ]; then
    export HF_API_KEY=$(jq -r '.huggingface' ~/.llm/keys.json)
fi

echo "Environment activated:"
echo "- OPENWEATHER_DB_PATH: $OPENWEATHER_DB_PATH"
echo "- OPENWEATHER_LOG_LEVEL: $OPENWEATHER_LOG_LEVEL"
echo "- OPENAI_API_KEY: ${OPENAI_API_KEY:0:4}..."
echo "- HF_API_KEY: ${HF_API_KEY:0:4}..." 