#!/bin/bash
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# echo "======= Check Environment Variables ======="
# env | sort
# echo "==========================================="

# export CUDA_VISIBLE_DEVICES=0,1,2,3
source activate /mnt/data/xujianwei.xjw/conda_envs/ms-swift

# Set output log directory and log file
SAFE_ID=${MLFLOW_TASK_ID//#/_}
SAFE_ID=${SAFE_ID:-"manual_run"}
OUTPUT_DIR="/mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/output_dir_endovlp_new/$1/${SAFE_ID}"
# OUTPUT_DIR="/mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/output_dir_debug/$(date +'%Y-%m-%d-%H_%M_%S')"

# LOG_FILE="$OUTPUT_DIR/log_$(date +'%Y%m%d_%H%M%S').log"
LOG_FILE="$OUTPUT_DIR/log_train.log"

# Create the output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
# export PYTHONFAULTHANDLER=1
# LOG_DIR_PATH="/mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/torchrun_logs/job_$(date +%Y%m%d_%H%M%S)"
# mkdir -p $LOG_DIR_PATH

# Navigate to project directory
cd /mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# [endovlp] bs: 8; datasize 1000 -> max-memery: 61G 
# --data_file_train /mnt/data/xujianwei.xjw/胃镜/wzzx/dataset/Image_Text_wzzx_0126_addAbnFlags_Filtered_NoBoston.jsonl \
# --master_port=29505 
# --debug \
# --debug_datasize 200 \
# --resume /mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/output_dir_endovlp/2026-01-30-14_11_32/checkpoint-0.pth \
# --log-dir=$LOG_DIR_PATH \
torchrun --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$WORLD_SIZE \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv-backend=c10d \
    --rdzv-conf=join_timeout=36000 \
    main_pretrain_v1.py \
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
    --data_file_train /mnt/data/xujianwei.xjw/胃镜/total/dataset/Image_Text_wzzx_Final_lim7_addAllPaths.jsonl /mnt/data/xujianwei.xjw/胃镜/total/dataset/Image_Text_szy_Final_lim7_addAllPaths.jsonl \
    --transform_list_file /mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/dataset/configs/transform.yaml \
    --norm_pix_loss \
    --pooing_topk 3 \
    --vit_pretrain dinov3 \
    --model vit_base_patch16 \
    --train_from_scratch \
    >> "$LOG_FILE" 2>&1

# Notify that training is complete
echo "Training completed. Logs are saved in $LOG_FILE"