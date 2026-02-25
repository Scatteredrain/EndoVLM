#!/bin/bash

# 1. 禁用 P2P 访问和 NVLink 的某些通信路径（解决 Invalid access 报错）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1  # 如果是单机多卡，不需要 IB 通信，建议禁用

# 2. 增加通信超时检测阈值（可选）
export NCCL_BLOCKING_WAIT=1

# 3. 强制在发生 CUDA 错误时阻塞，方便定位具体哪行报错
export CUDA_LAUNCH_BLOCKING=1


DEBUG=false
for arg in "$@"; do
  if [ "$arg" == "--debug" ]; then
    DEBUG=true
    shift
  fi
done

cd /mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP
source activate /mnt/data/xujianwei.xjw/conda_envs/ms-swift

if [ "$DEBUG" = true ]; then
    echo "Debug mode"
    OUTPUT_DIR="/mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/output_dir_debug/$(date +'%Y-%m-%d-%H_%M_%S')"
    LOG_FILE="$OUTPUT_DIR/log_$(date +'%Y%m%d_%H%M%S').log"
    # Create the output directory if it does not exist
    mkdir -p "$OUTPUT_DIR"

    torchrun --nproc_per_node=1 --master_port=29511 main_pretrain_v1.py \
    --seed 42 \
    --batch_size 48 \
    --input_size 224 \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$OUTPUT_DIR" \
    --data_file_train /mnt/data/xujianwei.xjw/胃镜/total/dataset/Image_Text_wzzx_Final_lim7_addAllPaths.jsonl /mnt/data/xujianwei.xjw/胃镜/total/dataset/Image_Text_szy_Final_lim7_addAllPaths.jsonl \
    --transform_list_file /mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/dataset/configs/transform.yaml \
    --mode clip \
    --vit_pretrain clip \
    --debug \
    --debug_datasize 200 \
    
else
    echo "Normal training"
    source activate /mnt/data/xujianwei.xjw/conda_envs/ms-swift
    OUTPUT_DIR="/mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/output_dir_clip/$(date +'%Y-%m-%d-%H_%M_%S')"
    LOG_FILE="$OUTPUT_DIR/log_$(date +'%Y%m%d_%H%M%S').log"
    # Create the output directory if it does not exist
    mkdir -p "$OUTPUT_DIR"

    torchrun --nproc_per_node=8 --master_port=29511 main_pretrain_v1.py \
    --seed 42 \
    --batch_size 48 \
    --input_size 224 \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$OUTPUT_DIR" \
    --data_file_train /mnt/data/xujianwei.xjw/胃镜/total/dataset/Image_Text_wzzx_Final_lim7_addAllPaths.jsonl /mnt/data/xujianwei.xjw/胃镜/total/dataset/Image_Text_szy_Final_lim7_addAllPaths.jsonl \
    --transform_list_file /mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/dataset/configs/transform.yaml \
    --mode clip \
    --vit_pretrain clip \
    --resume /mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/output_dir_clip/2026-02-05-06_25_39/checkpoint-3.pth \
    >> "$LOG_FILE" 2>&1
fi


# [endovlp] bs: 64; datasize wzzx -> max-memery: 83G 
# --data_file_train /mnt/data/xujianwei.xjw/胃镜/wzzx/dataset/Image_Text_wzzx_0126_addAbnFlags_Filtered_NoBoston.jsonl \
# --master_port=29505 
# --debug \
# --debug_datasize 2000 \
    # --resume /mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/output_dir_clip/2026-01-30-11_52_41/checkpoint-0.pth \

# Notify that training is complete
echo "Training completed. Logs are saved in $LOG_FILE"