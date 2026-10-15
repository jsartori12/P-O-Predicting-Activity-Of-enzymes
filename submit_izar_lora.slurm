#!/bin/bash -l
#SBATCH --job-name=esm2-lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00              # check your QOS max walltime; if the job is cut,
                                      # resubmit the same command -- it resumes from last.pt
#SBATCH --partition=gpu
#SBATCH --qos=<fill-in>              # same values as ESM2Classifier/submit_izar.slurm
#SBATCH --account=<fill-in>
#SBATCH --output=logs/%j.out

# Usage (from ~/ESM2LoRa/ on Izar):
#   mkdir -p logs && sbatch submit_izar_lora.slurm [run_name] [extra train.py args...]
#   e.g.  sbatch submit_izar_lora.slurm lora_r8
#         sbatch submit_izar_lora.slurm lora_pilot --max-train-samples 50000 --epochs 1
#
# Reuses ../ESM2Classifier/esm2.sif: it already has torch (cu121) + transformers
# 5.17.0 + peft 0.21.0, so no rebuild. Expected layout on Izar (same as local):
#   ~/ESM2Classifier/esm2.sif
#   ~/ESM2LoRa/                      (this directory)
#   ~/classifier_dataset/v2_no_wt_decoys/splits/{train,test}.csv

set -e
mkdir -p logs

RUN_NAME=${1:-lora_r8}
shift || true
CKPT=runs/$RUN_NAME/checkpoints/last.pt
SIF=../ESM2Classifier/esm2.sif

echo "job started: $(date) | run: $RUN_NAME | extra args: $*"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

export HF_HOME=/scratch/izar/$USER/hf_cache      # 650M weights already cached here by the extraction job
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PARENT_DIR=$(dirname "$PWD")

RESUME=""
if [ -f "$CKPT" ]; then
    RESUME="--resume $CKPT"
    echo "found $CKPT -- resuming"
fi

srun apptainer exec --nv --bind "$PARENT_DIR:$PARENT_DIR" "$SIF" \
    python3 train.py --run-name "$RUN_NAME" $RESUME "$@"

srun apptainer exec --nv --bind "$PARENT_DIR:$PARENT_DIR" "$SIF" \
    python3 evaluate.py --run-dir "runs/$RUN_NAME"

echo "job finished: $(date)"
