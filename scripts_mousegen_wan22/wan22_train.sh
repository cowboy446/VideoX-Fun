export MODEL_NAME="models/wan22/Wan2.2-TI2V-5B"
export DATASET_NAME="datasets/poor_dataset_01/"
export DATASET_META_NAME="datasets/poor_dataset_01/meta_text_video_pairs_qwen_8b.json"
export CUDA_VISIBLE_DEVICES=0,4,5,7
export WAN22_ACCELERATE="/home/zhangrong/miniconda3/envs/wan22/bin/accelerate"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

"$WAN22_ACCELERATE" launch \
  --mixed_precision="bf16" \
  --use_fsdp \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock \
  --fsdp_sharding_strategy "FULL_SHARD" \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_backward_prefetch "BACKWARD_PRE" \
  --fsdp_cpu_ram_efficient_loading False \
  scripts_mousegen_wan22/wan22_train.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=0 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=300 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="mousegen_wan22_5b_fsdp" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --boundary_type="full" \
  --train_mode="ti2v" \
  --trainable_modules "." \
  --validation_steps=100 \
  --validation_prompts \
    "一只黑色小鼠在实验台上探索，自由地移动，背景是实验室环境，光线明亮，细节丰富，高清晰度" \
    "一只黑色小鼠在野外草地环境下爬行，画面生动，高清晰度" \
  --validation_paths \
    "datasets/validset/view4_f00637.jpg" \
    "datasets/validset/vace_mouse_frame_001.jpg"


# The Training Shell Code for Image to Video
# You need to use "config/wan2.2/wan_civitai_i2v.yaml"
# 
# export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-I2V-A14B"
# export DATASET_NAME="datasets/internal_datasets/"
# export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# # NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# # export NCCL_IB_DISABLE=1
# # export NCCL_P2P_DISABLE=1
# NCCL_DEBUG=INFO

# accelerate launch --mixed_precision="bf16" scripts_mousegen_wan22/wan22_train.py \
#   --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATASET_NAME \
#   --train_data_meta=$DATASET_META_NAME \
#   --image_sample_size=640 \
#   --video_sample_size=640 \
#   --token_sample_size=640 \
#   --video_sample_stride=2 \
#   --video_sample_n_frames=81 \
#   --train_batch_size=1 \
#   --video_repeat=1 \
#   --gradient_accumulation_steps=1 \
#   --dataloader_num_workers=8 \
#   --num_train_epochs=100 \
#   --checkpointing_steps=50 \
#   --learning_rate=2e-05 \
#   --lr_scheduler="constant_with_warmup" \
#   --lr_warmup_steps=100 \
#   --seed=42 \
#   --output_dir="output_dir_wan2.2" \
#   --gradient_checkpointing \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --vae_mini_batch=1 \
#   --max_grad_norm=0.05 \
#   --random_hw_adapt \
#   --training_with_video_token_length \
#   --enable_bucket \
#   --uniform_sampling \
#   --low_vram \
#   --boundary_type="low" \
#   --train_mode="i2v" \
#   --trainable_modules "."