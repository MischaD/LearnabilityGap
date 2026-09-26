#!/bin/bash

#SBATCH --job-name=run_all_latents
#SBATCH --output=outputs/logs/run_all_latents_%A_%a.out
#SBATCH --error=outputs/logs/run_all_latents_%A_%a.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-19

set -e

CLASSIFIER_MODEL="${1:-resnet50}"
set --

source ~/miniforge3/bin/activate
conda activate /lus/lfs1aip2/projects/u6db/conda/longtail

if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/syam/scripts"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
fi
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$BASE_DIR"

TRAIN_SCRIPT="$BASE_DIR/syam/scripts/classifier_train.py"
export PYTHONPATH="$BASE_DIR/syam:$PYTHONPATH"

# Arrays for iteration
DATASETS=("cardium" "isic" "mimic" "ctrate")
MODELS=("flux1" "flux2" "medvae_xray" "medvae_ft_xray" "sdvae")

# Check if we are in an array job
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Please submit this script with sbatch."
    exit 1
fi

# Calculate which model and dataset this array task corresponds to
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 4))
DS_IDX=$((SLURM_ARRAY_TASK_ID % 4))

MODEL_EXP_NAME="${MODELS[$MODEL_IDX]}"
DS="${DATASETS[$DS_IDX]}"

echo "============================================================"
echo "RUNNING ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "SELECTED MODEL: $MODEL_EXP_NAME"
echo "SELECTED DATASET: $DS"
echo "============================================================"

# Loop through initializations: pretrained, then random initialization
for RAND_INIT_FLAG in "" "--rand_init"; do

    if [[ "$RAND_INIT_FLAG" == "--rand_init" ]]; then
        RAND_SUFFIX="_rand"
        echo "=== Random initialization enabled ==="
    else
        RAND_SUFFIX=""
        echo "=== Pretrained initialization enabled ==="
    fi

    echo "------------------------------------------------------------"
    echo "RUNNING: $DS | $MODEL_EXP_NAME | latent_space | Init: ${RAND_SUFFIX:-_pretrained}"
    echo "------------------------------------------------------------"

    # Set CSV path
    CSV_PATH="$BASE_DIR/$DS.csv"
    
    # Set Latent Directory and Output Directory
    LATENTS_DIR="$BASE_DIR/outputs/outputs_${MODEL_EXP_NAME}_${DS}/latents"
    OUT_DIR="$BASE_DIR/outputs/outputs_${MODEL_EXP_NAME}_${DS}/results/latent_space/${CLASSIFIER_MODEL}/$DS"

    if [ ! -d "$LATENTS_DIR" ]; then
        if [ -f "${LATENTS_DIR}.tar.gz" ]; then
            echo "Extracting ${LATENTS_DIR}.tar.gz..."
            mkdir -p "$LATENTS_DIR"
            tar -xzf "${LATENTS_DIR}.tar.gz" -C "$LATENTS_DIR"
        else
            echo "Warning: Latent directory and tarball not found for $LATENTS_DIR. Skipping!"
            continue
        fi
    fi

    mkdir -p "$OUT_DIR"

    # Execute the training script on the latents
    python "$TRAIN_SCRIPT" \
        --data_dir "$LATENTS_DIR" \
        --filelist "$CSV_PATH" \
        --out_dir "$OUT_DIR" \
        --loss ldam \
        --rw_method cb \
        --drw \
        --max_epochs 60 \
        --patience 15 \
        --batch_size 256 \
        --lr 1e-4 \
        --do_crossfold \
        --is_latent \
        --mask_ratio 0.2 \
        --model_name "$CLASSIFIER_MODEL" \
        $RAND_INIT_FLAG

    echo -e "\nFinished $DS / $MODEL_EXP_NAME / latent_space / Init: ${RAND_SUFFIX:-_pretrained}\n"

done

echo "All latent space trainings for $MODEL_EXP_NAME on $DS completed!"
