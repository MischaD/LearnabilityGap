#!/bin/bash

#SBATCH --job-name=train_noise_cond_imagenet
#SBATCH --output=outputs/logs/noise_cond_imagenet_%j.out
#SBATCH --error=outputs/logs/noise_cond_imagenet_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -e

# Setup environment
source ~/miniforge3/bin/activate
conda activate /lus/lfs1aip2/projects/u6db/conda/longtail

# Configuration
LATENTS_DIR="./outputs/imagenet_latents_flux2_dev"
OUTPUT_DIR="./outputs/imagenet_classifier_noise_cond_convnext"
CSV_FILE="imagenet_train.csv"
MEAN_PATH="${LATENTS_DIR}/imagenet_stats_channel_mean.pt"

# Make sure log dir exists
mkdir -p outputs/logs
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$(pwd)/syam:$PYTHONPATH"

# Run noise conditioned training
echo "Starting Noise Conditioned training on ImageNet latents..."
python syam/scripts/classifier_train_noise_cond.py \
    --data_dir "$LATENTS_DIR" \
    --filelist "$CSV_FILE" \
    --out_dir "$OUTPUT_DIR" \
    --mean_path "$MEAN_PATH" \
    --max_epochs 60 \
    --patience 15 \
    --batch_size 256 \
    --lr 0.0001 \
    --model_name ConvNeXt-Tiny \
    --is_latent

echo "Job finished."
