#!/bin/bash

#SBATCH --job-name=run_noise_cond_distilled
#SBATCH --output=outputs/logs/run_noise_cond_distilled_%A_%a.out
#SBATCH --error=outputs/logs/run_noise_cond_distilled_%A_%a.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3

set -e

source ~/miniforge3/bin/activate
conda activate /lus/lfs1aip2/projects/u6db/conda/longtail

# Find project root
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/syam/scripts"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
fi
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$BASE_DIR"

TRAIN_SCRIPT="$BASE_DIR/syam/scripts/classifier_train_noise_cond.py"
export PYTHONPATH="$BASE_DIR/syam:$PYTHONPATH"

# Arrays for iteration
DATASETS=("cardium" "isic" "mimic" "ctrate")

# Check if we are in an array job
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Please submit this script with sbatch."
    exit 1
fi

DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
MODEL_EXP_NAME="flux2"
CHECKPOINT="/home/u6db/b180dc10.u6db/pycharm/longtail/outputs/flux2distilled_best/{ds}/cxr-lt_ConvNeXt-Tiny_rw-cb_ldam-drw_cb-beta-0.9999_lr-0.0005_bs-256_fold_{fold}/best.pt"

echo "============================================================"
echo "RUNNING ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "SELECTED MODEL: $MODEL_EXP_NAME"
echo "SELECTED DATASET: $DS"
echo "STARTING CHECKPOINT TEMPLATE: $CHECKPOINT"
echo "============================================================"

CSV_PATH="$BASE_DIR/$DS.csv"
LATENTS_DIR="$BASE_DIR/outputs/outputs_${MODEL_EXP_NAME}_${DS}/latents"
OUT_DIR="$BASE_DIR/outputs/flux2noisecond_from_distilled/$DS"

if [ ! -d "$LATENTS_DIR" ]; then
    if [ -f "${LATENTS_DIR}.tar.gz" ]; then
        echo "Extracting ${LATENTS_DIR}.tar.gz..."
        mkdir -p "$LATENTS_DIR"
        tar -xzf "${LATENTS_DIR}.tar.gz" -C "$LATENTS_DIR"
    else
        echo "Error: Latent directory and tarball not found for $LATENTS_DIR."
        exit 1
    fi
fi

mkdir -p "$OUT_DIR"

python "$TRAIN_SCRIPT" \
    --data_dir "$LATENTS_DIR" \
    --filelist "$CSV_PATH" \
    --out_dir "$OUT_DIR" \
    --model_name ConvNeXt-Tiny \
    --model_path "$CHECKPOINT" \
    --loss ldam \
    --rw_method cb \
    --drw \
    --max_epochs 60 \
    --patience 15 \
    --batch_size 256 \
    --lr 5e-4 \
    --do_crossfold \
    --is_latent 

echo -e "\nFinished $DS / $MODEL_EXP_NAME / noise_cond_from_distilled\n"
echo "All done!"
