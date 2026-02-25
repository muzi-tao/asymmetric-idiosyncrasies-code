#!/bin/bash -l
#SBATCH --job-name=flux-schnell_claude_cap2img_ddp
#SBATCH --output=slurm_history/claude_ddp_%j.out
#SBATCH --error=slurm_history/claude_ddp_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --export=ALL

# Configuration parameters - modify these as needed
NUM_GPUS=8
PROMPT_FILE="dataset/text/claude-3-5-sonnet_prompts.json"
MODEL_NAME="claude"
SD_MODEL_ID="flux-schnell"
BATCH_SIZE=4
IMAGE_SIZE=256
ADDITIONAL_ARGS=""  # e.g., "--resume"

# Ensure log directory exists for SBATCH output/error paths
mkdir -p slurm_history

# Conda init path fallback for portability
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh under \$HOME/miniconda3 or \$HOME/anaconda3"
    exit 1
fi

conda activate muzi-icap
PY=$(which python)
echo "Using Python: $PY"

# 4) Generate shared timestamp for all ranks
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Host: $(hostname)  Date: $(date)  Timestamp: $TIMESTAMP"

# 5) Set up distributed training environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN  # Set to WARN for less verbose output; or INFO for debug

# Note: With DDP, no manual sharding is needed - torchrun handles data distribution
# Each rank will automatically process its assigned portion of the dataset

# 6) Launch DDP training with torchrun
echo "Launching DDP training with $NUM_GPUS GPUs..."
echo "Model: $SD_MODEL_ID, Prompt source: $MODEL_NAME"
echo "Prompt file: $PROMPT_FILE"
echo "Batch size: $BATCH_SIZE, Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}, Additional args: $ADDITIONAL_ARGS"

srun torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    img_gen.py \
        --file $PROMPT_FILE \
        --model-name $MODEL_NAME \
        --sd-model-id $SD_MODEL_ID \
        --batch-size $BATCH_SIZE \
        --image-width $IMAGE_SIZE \
        --image-height $IMAGE_SIZE \
        $ADDITIONAL_ARGS

echo "DDP Job finished at $(date)"