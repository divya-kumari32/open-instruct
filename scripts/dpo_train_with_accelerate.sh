if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <train_file> <output_dir>"
  exit 1
fi

# you need 8 GPUs for full finetuning
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

MODEL_NAME=$1
OUTPUT_DIR=$2

echo "Training model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
echo "Model: ${MODEL_NAME}"
echo "Output directory: ${OUTPUT_DIR}"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/dpo_tune.py \
    --model_name_or_path /gpfs/lchu/ckpt/mamba2_9b_dolma_2t/open_instruct/"${MODEL_NAME}" \
    --use_flash_attn \
    --tokenizer_name /gpfs/lchu/ckpt/mamba2_9b_dolma_2t/open_instruct/"${MODEL_NAME}" \
    --use_slow_tokenizer \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir /gpfs/lchu/ckpt/mamba2_9b_dolma_2t/open_instruct/dpo/"${OUTPUT_DIR}" \
    --report_to tensorboard \
    --logging_steps 1 \
    --try_launch_beaker_eval_jobs False \
    --push_to_hub False \
    # --max_train_samples 100 \
    # --gradient_checkpointing \