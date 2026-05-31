#!/bin/bash
# run_agent.sh — executed by HTCondor on the GPU node
# Activates venv, starts Ollama, and runs the agent

set -e

PROJECT_DIR=/nethome/fkhalid/smart-office-support-agent

echo "== Starting Smart Office Support Agent =="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"

# Activate virtual environment
source $PROJECT_DIR/venv/bin/activate

# Set Ollama to store models in scratch (not nethome — space limited)
export OLLAMA_MODELS=/scratch/fkhalid/models

# Start Ollama in the background
echo "Starting Ollama..."
ollama serve &
OLLAMA_PID=$!
sleep 5

# Pull model if not already cached
echo "Checking model..."
ollama pull qwen3:8b

# Run the classifier test
echo "Running classifier test..."
cd $PROJECT_DIR
python src/part1_llm/classifier.py

# Kill Ollama when done
kill $OLLAMA_PID 2>/dev/null || true
echo "== Done =="
