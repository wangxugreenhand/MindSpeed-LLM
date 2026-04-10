import torch
import os
from safetensors.torch import load_file

# --- 1. 配置路径 ---
safetensors_path = 'model.safetensors'  # 请确认你的文件名
output_pth = 'vit-base-mcore.pth'

if not os.path.exists(safetensors_path):
    print(f"Error: 找不到 {safetensors_path}")
    exit(1)

print(f"Loading weights from {safetensors_path}...")
original_weights = load_file(safetensors_path)

converted_weights = {}

# --- 2. 映射权重 (Embedding & Classifier) ---
if 'vit.embeddings.patch_embeddings.projection.weight' in original_weights:
    converted_weights['conv1.weight'] = original_weights['vit.embeddings.patch_embeddings.projection.weight']
    converted_weights['conv1.bias'] = original_weights['vit.embeddings.patch_embeddings.projection.bias']

if 'vit.embeddings.position_embeddings' in original_weights:
    pos_embed = original_weights['vit.embeddings.position_embeddings']
    if pos_embed.dim() == 3 and pos_embed.shape[0] == 1:
        pos_embed = pos_embed.squeeze(0)
    converted_weights['position_embeddings.weight'] = pos_embed

if 'vit.embeddings.cls_token' in original_weights:
    converted_weights['class_token'] = original_weights['vit.embeddings.cls_token']

if 'vit.layernorm.weight' in original_weights:
    converted_weights['layernorm.weight'] = original_weights['vit.layernorm.weight']
    converted_weights['layernorm.bias'] = original_weights['vit.layernorm.bias']

if 'classifier.weight' in original_weights:
    converted_weights['classifier.weight'] = original_weights['classifier.weight']
    converted_weights['classifier.bias'] = original_weights['classifier.bias']

# --- 3. 映射 Transformer 层 ---
num_layers = 12
print(f"Converting {num_layers} layers...")

for i in range(num_layers):
    hf_prefix = f'vit.encoder.layer.{i}'
    mg_prefix = f'encoder.layers.{i}'

    # Attention QKV
    q_w = original_weights[f'{hf_prefix}.attention.attention.query.weight']
    k_w = original_weights[f'{hf_prefix}.attention.attention.key.weight']
    v_w = original_weights[f'{hf_prefix}.attention.attention.value.weight']
    q_b = original_weights[f'{hf_prefix}.attention.attention.query.bias']
    k_b = original_weights[f'{hf_prefix}.attention.attention.key.bias']
    v_b = original_weights[f'{hf_prefix}.attention.attention.value.bias']

    converted_weights[f'{mg_prefix}.self_attention.linear_qkv.weight'] = torch.cat([q_w, k_w, v_w], dim=0)
    converted_weights[f'{mg_prefix}.self_attention.linear_qkv.bias'] = torch.cat([q_b, k_b, v_b], dim=0)

    # Attention Output
    converted_weights[f'{mg_prefix}.self_attention.linear_proj.weight'] = original_weights[f'{hf_prefix}.attention.output.dense.weight']
    converted_weights[f'{mg_prefix}.self_attention.linear_proj.bias'] = original_weights[f'{hf_prefix}.attention.output.dense.bias']

    # MLP
    converted_weights[f'{mg_prefix}.mlp.linear_fc1.weight'] = original_weights[f'{hf_prefix}.intermediate.dense.weight']
    converted_weights[f'{mg_prefix}.mlp.linear_fc1.bias'] = original_weights[f'{hf_prefix}.intermediate.dense.bias']
    converted_weights[f'{mg_prefix}.mlp.linear_fc2.weight'] = original_weights[f'{hf_prefix}.output.dense.weight']
    converted_weights[f'{mg_prefix}.mlp.linear_fc2.bias'] = original_weights[f'{hf_prefix}.output.dense.bias']

    # LayerNorms
    converted_weights[f'{mg_prefix}.input_layernorm.weight'] = original_weights[f'{hf_prefix}.layernorm_before.weight']
    converted_weights[f'{mg_prefix}.input_layernorm.bias'] = original_weights[f'{hf_prefix}.layernorm_before.bias']
    converted_weights[f'{mg_prefix}.pre_mlp_layernorm.weight'] = original_weights[f'{hf_prefix}.layernorm_after.weight']
    converted_weights[f'{mg_prefix}.pre_mlp_layernorm.bias'] = original_weights[f'{hf_prefix}.layernorm_after.bias']

torch.save(converted_weights, output_pth)
print(f"Success! Saved to {output_pth}")