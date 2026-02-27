#!/bin/bash

source activate EndoVLM
cd EndoVLM

# Set output log directory and log file
SAFE_ID=${MLFLOW_TASK_ID//#/_}
SAFE_ID=${SAFE_ID:-"manual_run"}
OUTPUT_DIR="${SAFE_ID}"
LOG_FILE="$OUTPUT_DIR/log_train.log"

# Create the output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node=8 --master_port=29510 main_pretrain.py \
    --seed 42 \
    --batch_size 6 \
    --input_size 224 \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$OUTPUT_DIR" \
    --data_file_train dataset.jsonl \
    --transform_list_file dataset/configs/transform.yaml \
    --norm_pix_loss \
    --pooing_topk 3 \
    --model vit_base_patch16 \
    >> "$LOG_FILE" 2>&1

# Notify that training is complete
echo "Training completed. Logs are saved in $LOG_FILE"