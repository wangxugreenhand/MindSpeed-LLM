#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export HCCL_CONNECT_TIMEOUT=600

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6003
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="../Qwen3-0.6B-mcore"
CKPT_SAVE_DIR="../Qwen3-0.6B-finetune"
DATA_PATH="./finetune_dataset/alpaca_en"
TOKENIZER_PATH="../Qwen3-0.6B-hf"

TP=1
PP=4
MBS=1
GBS=4

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --transformer-impl local \
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 28 \
    --hidden-size 1024 \
    --ffn-hidden-size 3072 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 8 \
    --position-embedding-type rope \
    --kv-channels 128 \
    --qk-layernorm \
    --max-position-embeddings 40960 \
    --seq-length 512 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-iters 50 \
    --init-method-std 0.01 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
	--normalization RMSNorm \
    --no-gradient-accumulation-fusion \
    --attention-softmax-in-fp32 \
    --no-masked-softmax-fusion \
    --min-lr 1.25e-7 \
    --lr 1.25e-6 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --initial-loss-scale 4096 \
    --disable-bias-linear \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --fp16 \
    --swiglu \
    --no-bias-swiglu-fusion \
	--no-rope-fusion
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --prompt-type qwen \
    --variable-seq-lengths
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
	 2>&1 | tee "logs/sft_$(date +%Y%m%d_%H%M%S).log"
