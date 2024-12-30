
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <train_file> <output_dir>"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CHECKPOINT_PATH=$1
TRAIN_FILE=$2
OUTPUT_DIR=$3
LEARNING_RATE=$4
GLOBAL_BATCH_SIZE=$5
WEIGHT_DECAY=$6
MACHINES=$7
EXP_NAME=$8

NUM_GPUS=$(($MACHINES * 8))
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=$(($GLOBAL_BATCH_SIZE))
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

echo "Training Mamba model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
echo "Checkpoint path: ${CHECKPOINT_PATH}"
echo "Training file: ${TRAIN_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "learning rate: ${LEARNING_RATE}"
echo "weight decay: ${WEIGHT_DECAY}"
echo "# Machines: ${MACHINES}"
echo "exp name: ${EXP_NAME}"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
accelerate launch \
    --mixed_precision bf16 \
    --num_machines $((MACHINES)) \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path "${CHECKPOINT_PATH}" \
    --use_flash_attn \
    --tokenizer_name "${CHECKPOINT_PATH}" \
    --train_file "${TRAIN_FILE}" \
    --max_seq_length 4096 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay "${WEIGHT_DECAY}" \
    --num_train_epochs 2 \
    --output_dir "${OUTPUT_DIR}" \
    --with_tracking \
    --wandb_entity project-avengers \
    --exp_name "${EXP_NAME}" \
    --report_to tensorboard \
    --logging_steps 1 \
    --try_launch_beaker_eval_jobs False \
    --push_to_hub False \
    # --max_train_samples 1000000 \
