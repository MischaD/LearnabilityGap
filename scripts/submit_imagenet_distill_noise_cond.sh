#!/bin/bash

#SBATCH --job-name=distill_noise_cond_imagenet
#SBATCH --output=outputs/logs/distill_noise_cond_imagenet_%j.out
#SBATCH --error=outputs/logs/distill_noise_cond_imagenet_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -e

# Setup environment
source ~/miniforge3/bin/activate
conda activate /lus/lfs1aip2/projects/u6db/conda/longtail

# Configuration
DATA_DIR="/lus/lfs1aip2/projects/u6db/data/imagenet/ILSVRC/Data/CLS-LOC"
LATENT_DIR="./outputs/imagenet_latents_flux2_dev"
OUTPUT_DIR="./outputs/imagenet_classifier_distill_noise_cond_convnext_best"
CSV_FILE="imagenet_train.csv"

# Make sure log dir exists
mkdir -p outputs/logs
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$(pwd)/syam:$PYTHONPATH"

# Run noise-conditioned distillation training
echo "Starting Distillation + Noise Conditioned training on ImageNet latents..."
python syam/scripts/classifier_distill_noise_cond.py \
    --data_dir "$DATA_DIR" \
    --latent_dir "$LATENT_DIR" \
    --filelist "$CSV_FILE" \
    --out_dir "$OUTPUT_DIR" \
    --max_epochs 60 \
    --patience 15 \
    --batch_size 256 \
    --lr 0.0001 \
    --model_name ConvNeXt-Tiny \
    --teacher_model ConvNeXt-Tiny \
    --alpha 0.5

echo "Job finished."
