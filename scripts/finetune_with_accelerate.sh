
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <train_file> <output_dir>"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

CHECKPOINT_PATH=$1
TRAIN_FILE=$2
OUTPUT_DIR=$3
LEARNING_RATE=$4

echo "Training Mamba model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
echo "Checkpoint path: ${CHECKPOINT_PATH}"
echo "Training file: ${TRAIN_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "learning rate: ${LEARNING_RATE}"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path /datasets/lchu/ckpt/"${CHECKPOINT_PATH}" \
    --use_flash_attn \
    --tokenizer_name /datasets/lchu/ckpt/"${CHECKPOINT_PATH}" \
    --train_file /datasets/instruct_data/"${TRAIN_FILE}" \
    --max_seq_length 4096 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir /ckpt/lchu/ckpt/mamba2_9b_dolma_2t/open_instruct/"${OUTPUT_DIR}" \
    --report_to tensorboard \
    --logging_steps 1 \
    --try_launch_beaker_eval_jobs False \
    --push_to_hub False \
    # --max_train_samples 1000000 \