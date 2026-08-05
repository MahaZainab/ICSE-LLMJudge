#!/bin/bash
#SBATCH --job-name=gemma27b_codeqa
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/aiau010_scratch/maz0032/logs/gemma27b_codeqa_%j.out
#SBATCH --error=/aiau010_scratch/maz0032/logs/gemma27b_codeqa_%j.err

set -euo pipefail

mkdir -p /aiau010_scratch/maz0032/logs

module load python-v3.12

export HF_HOME="/aiau010_scratch/maz0032/.cache/huggingface"

cd /aiau010_scratch/maz0032
# adjust this to wherever the RQ_bigLLM_generate_codeqa_gemma2_27b.py script actually lives
cd icsellmjudge/bigllm

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "HF_HOME: $HF_HOME"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# swap this line for `source .venv/bin/activate && python ...` if you're not using uv
uv run python RQ_bigLLM_generate_codeqa_gemma2_27b.py

echo "Job finished: $(date)"
