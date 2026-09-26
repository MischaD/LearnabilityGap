#!/bin/bash

#SBATCH --job-name=distill_downstream
#SBATCH --output=outputs/logs/distill_downstream_%A_%a.out
#SBATCH --error=outputs/logs/distill_downstream_%A_%a.err
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

TRAIN_SCRIPT="$BASE_DIR/syam/scripts/classifier_distill.py"
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

# Original dataset images
DATA_DIR="/home/u6db/b180dc10.u6db/pycharm/longtail/data/"

echo "============================================================"
echo "RUNNING ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "SELECTED DATASET: $DS"
echo "============================================================"

# Set paths
CSV_PATH="$BASE_DIR/$DS.csv"
LATENTS_DIR="$BASE_DIR/outputs/outputs_${MODEL_EXP_NAME}_${DS}/latents"
OUT_DIR="$BASE_DIR/outputs/flux2distilled_from_imagespace/${DS}"

TEACHER_DIR="$BASE_DIR/outputs/outputs_imagespace/image_space/${DS}"

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

echo "Running distillation training on latents from frozen imagespace teacher..."

python "$TRAIN_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --latent_dir "$LATENTS_DIR" \
    --filelist "$CSV_PATH" \
    --out_dir "$OUT_DIR" \
    --max_epochs 60 \
    --patience 15 \
    --batch_size 256 \
    --lr 5e-4 \
    --loss ldam \
    --rw_method cb \
    --drw \
    --model_name ConvNeXt-Tiny \
    --teacher_model resnet50 \
    --teacher_weights_dir "$TEACHER_DIR" \
    --alpha 0.5 \
    --do_crossfold \
    --is_latent 

echo -e "\nFinished Distillation for $DS\n"
