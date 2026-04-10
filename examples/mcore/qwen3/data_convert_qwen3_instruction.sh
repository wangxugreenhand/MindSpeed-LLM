# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ../alpaca_en/alpaca_data_en_52k.json \
    --tokenizer-name-or-path ../Qwen3-8B \
    --output-prefix ./finetune_dataset/alpaca_en \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --prompt-type qwen
