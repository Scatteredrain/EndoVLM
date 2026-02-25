#!/bin/bash

DEBUG=false
for arg in "$@"; do
  if [ "$arg" == "--debug" ]; then
    DEBUG=true
    shift
  fi
done

cd /mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP
source activate /mnt/data/xujianwei.xjw/conda_envs/ms-swift

# [endovlp] bs: 8; datasize 1000 -> max-memery: 61G 
# --data_file_train /mnt/data/xujianwei.xjw/胃镜/wzzx/dataset/Image_Text_wzzx_0126_addAbnFlags_Filtered_NoBoston.jsonl \
# --master_port=29505 
# --debug \
# --debug_datasize 200 \
# --resume /mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/output_dir_mae/2026-01-30-11_52_41/checkpoint-0.pth \
   
if [ "$DEBUG" = true ]; then
    echo "Debug mode"
    OUTPUT_DIR="/mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/output_dir_debug/$(date +'%Y-%m-%d-%H_%M_%S')"
    LOG_FILE="$OUTPUT_DIR/log_$(date +'%Y%m%d_%H%M%S').log"
    mkdir -p "$OUTPUT_DIR"
    torchrun --nproc_per_node=1 --master_port=29510 main_pretrain_v1.py \
        --seed 42 \
        --batch_size 256 \
        --input_size 224 \
        --epochs 100 \
        --warmup_epochs 5 \
        --blr 1.5e-4 \
        --weight_decay 0.05 \
        --mask_ratio 0.7 \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$OUTPUT_DIR" \
        --data_file_train /mnt/data/xujianwei.xjw/胃镜/total/dataset/Image_Text_wzzx_Final_lim7_addAllPaths.jsonl /mnt/data/xujianwei.xjw/胃镜/total/dataset/Image_Text_szy_Final_lim7_addAllPaths.jsonl \
        --transform_list_file /mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/dataset/configs/transform.yaml \
        --mode mae \
        --vit_pretrain mae \
        --norm_pix_loss \
        --debug \
        --debug_datasize 200 \

else
    echo "Normal mode"
    OUTPUT_DIR="/mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/output_dir_mae/$(date +'%Y-%m-%d-%H_%M_%S')"
    LOG_FILE="$OUTPUT_DIR/log_$(date +'%Y%m%d_%H%M%S').log"
    mkdir -p "$OUTPUT_DIR"
    torchrun --nproc_per_node=8 --master_port=29510 main_pretrain_v1.py \
        --seed 42 \
        --batch_size 512 \
        --input_size 224 \
        --epochs 100 \
        --warmup_epochs 20 \
        --blr 1.5e-4 \
        --weight_decay 0.05 \
        --mask_ratio 0.7 \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$OUTPUT_DIR" \
        --data_file_train /mnt/data/xujianwei.xjw/胃镜/total/dataset/Image_Text_wzzx_Final_lim7_addAllPaths.jsonl /mnt/data/xujianwei.xjw/胃镜/total/dataset/Image_Text_szy_Final_lim7_addAllPaths.jsonl \
        --transform_list_file /mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/dataset/configs/transform.yaml \
        --mode mae \
        --vit_pretrain mae \
        --norm_pix_loss \
        >> "$LOG_FILE" 2>&1
fi

# Notify that training is complete
echo "Training completed. Logs are saved in $LOG_FILE"