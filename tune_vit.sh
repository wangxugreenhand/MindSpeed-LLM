#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1200

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# 路径配置
CKPT_LOAD_DIR="../vit_base_patch16_224_in21k"
CKPT_SAVE_DIR="../vit_base_finetune"
DATA_PATH="../data"

# 模型配置
TP=1
PP=1
MBS=50
GBS=400

RC=1024
strategy="topk"
LR=0.1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# 增加了位置编码类参数与绕过Megatron底层的Tokenizer参数
VIT_ARGS="
    --use-mcore-models \
    --dataloader-type cyclic \
    --spec mindspeed_llm.tasks.models.spec.vit_spec layer_spec \
    --epoch-log-file logs/sft_vit_${strategy}_${RC}_${LR}_log.txt \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --ffn-hidden-size 3072 \
    --seq-length 197 \
    --max-position-embeddings 197 \
    --position-embedding-type learned_absolute \
    --tokenizer-type NullTokenizer \
    --make-vocab-size-divisible-by 1 \Claude 
    --vocab-size 1 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-iters 2500 \
    --num-classes 100 \
    --img-h 224 \
    --img-w 224 \
    --patch-dim 16 \
    --optimizer sgd \
    --sgd-momentum 0.0 \
    --min-lr 1e-6 \
    --lr ${LR} \
    --lr-decay-style cosine \
    --weight-decay 1e-5 \
    --clip-grad 0.0 \
    --no-load-optim \
    --no-load-rng \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --seed 42 \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --transformer-impl local 
"
GMC_ARGS="
    --use-gmc \
    --gmc-compression-strategy ${strategy} \
    --gmc-sparsity-rate ${RC} \
    --gmc-beta 0.9
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 10 \
    --eval-interval 125 \
    --eval-iters 25
"

# 将阶段修正为了标准的 sft
TUNE_ARGS="
    --finetune \
    --stage sft_vit
"

torchrun $DISTRIBUTED_ARGS posttrain_vit.py \
    $VIT_ARGS \
    $DATA_ARGS \
    $GMC_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    --distributed-backend nccl \
    --converted-vit-ckpt ${CKPT_LOAD_DIR} \
     2>&1 | tee "logs/tune_vit_${strategy}_${RC}_${LR}.log"