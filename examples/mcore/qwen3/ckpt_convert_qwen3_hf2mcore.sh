# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 2
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir ../Qwen3-8B/ \
    --save-dir ../Qwen3-8B-mcore-tp4-pp2/ \
    --tokenizer-model ../Qwen3-8B/tokenizer.json \
    --params-dtype bf16 \
    --model-type-hf qwen3
