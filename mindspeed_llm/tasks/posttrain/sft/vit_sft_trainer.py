import sys
import time
from functools import partial
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.distributed as dist
import megatron
from megatron.training import get_args, print_rank_0, get_timers
from megatron.training.training import (
    print_datetime,
    get_one_logger,
    append_to_progress_log,
    evaluate_and_print_results
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.spec_utils import import_module
from megatron.training.checkpointing import save_checkpoint
from megatron.training.utils import average_losses_across_data_parallel_group


from mindspeed_llm.training import build_train_args
from mindspeed_llm.training import train as megatron_train
from mindspeed_llm.training.initialize import set_jit_fusion_options
from dirichlet_data import get_dataset
from megatron.core.transformer.enums import ModelType
from transformers import ViTForImageClassification, ViTConfig

_TRAIN_START_TIME = time.time()

class MegatronHFViTWrapper(torch.nn.Module):
    """包裹 HuggingFace 模型，使其兼容 Megatron 的 schedules.py API"""
    def __init__(self, hf_model, config):
        super().__init__()
        # 必须作为模块的属性，以便正常注册参数
        self.model = hf_model
        self.model_type = ModelType.encoder_or_decoder
        self.config = config

    def set_input_tensor(self, input_tensor):
        """Megatron 流水线并行必需的接口"""
        self.input_tensor = input_tensor

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """占位函数：直接返回空字典，彻底跳过实际的权重收集和保存"""
        return {}

    def set_is_first_microbatch(self):
        """Sets the is_first_microbatch flag if it exists. When this flag is set, TE modules will update their fp8 parameter cache.
        
        """
        for m in self.modules():
            if hasattr(m, "is_first_microbatch"):
                m.is_first_microbatch = True

    def forward(self, images):
        # Megatron 传入 images，HF 模型接收 pixel_values
        # HF 模型返回 SequenceClassifierOutput，我们只需要取出 logits
        return self.model(images).logits


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """
        构建 PyTorch Dataset。
        Megatron 会自动处理 DataLoader, Sampler 和 Batching。
    """
    args = get_args()
    print_rank_0('> building datasets for ViT ...')

    # 1. 确定数据集名称和路径
    # dirichlet_data.py 里硬编码了 "cifar100" 和 "cifar10"
    # 通过命令行传入 --data cifar100
    dataset_name = 'cifar100'
    data_dir = args.data_path[0] if args.data_path else "./data"

    # 2. 构建 Dataset
    train_dataset = get_dataset(dataset_name, data_dir, split='train')
    val_dataset = get_dataset(dataset_name, data_dir, split='val')

    print_rank_0(f"> Loaded Train Dataset: {len(train_dataset)} samples")
    print_rank_0(f"> Loaded Val Dataset: {len(val_dataset)} samples")

    return train_dataset, val_dataset, None # 测试集暂不使用





class ViTSFTTrainer(ABC):
    def __init__(self):
        self.args = get_args()
        self.timers = get_timers()
        self.train_args = None
        self.test_data_iterator_list = None
        self.model_type = None  # ViT 使用默认 ModelType 即可
        self.train_valid_test_datasets_provider = train_valid_test_datasets_provider
        self.process_non_loss_data_func = None

        
        self.epoch_idx = 1
        self._train_steps, self._train_loss_sum, self._train_acc_sum = 0, 0.0, 0.0
        self._val_steps, self._val_loss_sum, self._val_acc_sum = 0, 0.0, 0.0
        self.epoch_log_file = getattr(self.args, "epoch_log_file", "Epoch_Log_ViT.txt")

        self.initialize()

    def initialize(self):
        """Sets up necessary configurations and logging."""
        self.train_valid_test_datasets_provider.is_distributed = True
        self.log_initialization()

        set_jit_fusion_options()
        self.synchronize_start_time()
        print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(time.time() - _TRAIN_START_TIME))

        app_metrics = {}
        app_metrics['app_start_time'] = round(_TRAIN_START_TIME * 1000.0)
        app_metrics['app_model_init_start_time'] = round(_TRAIN_START_TIME * 1000.0)

        # 构建训练参数 (核心步骤)
        self.train_args, self.test_data_iterator_list = build_train_args(
            self.args,
            self.timers,
            self.train_valid_test_datasets_provider, # 数据集回调
            self.model_provider,                     # 模型回调
            self.model_type,
            self.forward_step,                       # 前向回调
            self.process_non_loss_data_func,
            app_metrics
        )

    def log_initialization(self):
        if self.args.log_progress:
            append_to_progress_log("Starting ViT job")

    def synchronize_start_time(self):
        """Synchronize training start time across all distributed processes."""
        global _TRAIN_START_TIME
        start_time_tensor = torch.tensor([_TRAIN_START_TIME], dtype=torch.float, device='cuda')
        torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
        _TRAIN_START_TIME = start_time_tensor.item()
     
    def model_provider(self, pre_process=True, post_process=True):
        """Build Megatron ViT and load converted Megatron-format checkpoint (rank0 load + broadcast)."""
        args = get_args()
        print_rank_0("building Megatron ViT model ...")

        # # 构造 Megatron ViT
        # transformer_config = core_transformer_config_from_args(args)
        # transformer_layer_spec = import_module(args.spec) if getattr(args, "spec", None) else None
        # model = ViTForImageClassification(
        #     transformer_config=transformer_config,
        #     transformer_layer_spec=transformer_layer_spec,
        #     patch_dim=getattr(args, "patch_dim", 16),
        #     img_h=getattr(args, "img_h", 224),
        #     img_w=getattr(args, "img_w", 224),
        #     num_classes=getattr(args, "num_classes", 100),
        # )

        # # rank0 读取已转换好的 Megatron-format checkpoint（torch.save 的 dict）
        # converted_ckpt_path = getattr(args, "converted_vit_ckpt", None)
        # ckpt = None
        # rank = dist.get_rank()
        # if converted_ckpt_path and rank == 0:
        #     ckpt = torch.load(converted_ckpt_path, map_location="cpu")
        #     print_rank_0(f"rank0 loaded converted checkpoint from {converted_ckpt_path}")

        # # 广播 ckpt（所有进程必须调用）
        # if dist.is_initialized():
        #     obj = [ckpt]
        #     dist.broadcast_object_list(obj, src=0)
        #     ckpt = obj[0]

        # # 把 ckpt 中匹配的参数拷贝到模型
        # if ckpt is not None:
        #     my_sd = model.state_dict()

        #     matched_keys = 0
        #     used_ckpt_keys = set()
        #     sum_before = sum(p.sum().item() for p in my_sd.values() if p is not None and torch.is_tensor(p) and p.is_floating_point())

        #     for k in list(my_sd.keys()):
        #         if k in ckpt:
        #             if ckpt[k].shape == my_sd[k].shape:
        #                 my_sd[k].copy_(ckpt[k])
        #                 matched_keys += 1
        #                 used_ckpt_keys.add(k)
        #             else:
        #                 # 【新增预警 1】：名字匹配上了，但形状不对！
        #                 print_rank_0(f" [Warning] Shape mismatch for '{k}': model {my_sd[k].shape} vs ckpt {ckpt[k].shape}")
        #         else:
        #             # 【新增预警 2】：模型有这个层，但 ckpt 里名字不同或不存在！
        #             if "layernorm" in k:  # 专门捕捉包含 layernorm 名字的层
        #                 print_rank_0(f" [Warning] '{k}' is in model but NOT found in ckpt!")

        #     model.load_state_dict(my_sd, strict=False)           
        #     print_rank_0("Loaded converted Megatron weights into model")

        # if hasattr(model, 'classifier'):
        #     with torch.no_grad():
        #         model.classifier.weight.zero_()
        #         if model.classifier.bias is not None:
        #             model.classifier.bias.zero_()
        #     print_rank_0("Zero-initialized the classifier head.")


        # Megatron expect an iterable of modules

        hf_model = ViTForImageClassification.from_pretrained(
            getattr(args, "converted_vit_ckpt", None),
            num_labels=100,
            ignore_mismatched_sizes=True,
            local_files_only=True,
            attn_implementation="eager"
        )
        hf_model.classifier = nn.Linear(hf_model.config.hidden_size, 100)
        hf_model.classifier.weight.data.zero_()
        hf_model.classifier.bias.data.zero_()
        transformer_config = core_transformer_config_from_args(args)
        model = MegatronHFViTWrapper(hf_model, config=transformer_config)

        # 禁用后续 Megatron 的自动 data-parallel 广播（已经手动广播）
        args.data_parallel_random_init = False
        return model

    def get_batch(self, data_iterator):
        """Build the batch."""
        # Broadcast data.
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        # Data format from dirichlet_data/CIFAR: (images, labels)
        # images: [Batch, C, H, W] -> GPU Float
        # labels: [Batch] -> GPU Long
        
        # 这里的 None 检查是为了处理可能的空迭代器或者 Test 阶段
        if data is None:
            return None, None

        images = data[0].cuda()
        labels = data[1].long().cuda()
        
        return images, labels

    def loss_func(self, output, labels):
        """Loss function.
        Args:
            labels: 真实标签 [Batch]
            output_tensor: 模型输出 Logits [Batch, NumClasses]
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        logits = output.float()  # 确保 logits 是 float 类型
        loss = loss_fn(logits, labels)
        
        # 计算 Top-1 Accuracy 
        with torch.no_grad():
            # 获取预测的类别索引
            # logits: [B, num_classes] -> pred: [B]
            _, pred = logits.topk(1, 1, True, True)
            pred = pred.t()
            
            # 比较预测结果与真实标签
            # correct: [1, B]
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            
            # 计算正确总数并归一化为百分比
            # FloatTensor [1]
            correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
            acc1 = correct_k.mul_(100.0 / logits.size(0))
        
        averaged_stats = average_losses_across_data_parallel_group([loss, acc1])
        # mode = "Train" if torch.is_grad_enabled() else "Val"
        # current_step = self._train_steps + 1 if torch.is_grad_enabled() else self._val_steps + 1
        # print_rank_0(f"[{mode}] Epoch: {self.epoch_idx} | Iteration: {current_step} | loss: {averaged_stats[0].item():.4f}, acc1: {averaged_stats[1].item():.2f}%")

        c_loss = averaged_stats[0].item()
        c_acc = averaged_stats[1].item()
        if torch.is_grad_enabled():
            self._train_steps += 1
            self._train_loss_sum += c_loss
            self._train_acc_sum += c_acc
            if self._train_steps == 125:
                avg_train_loss = self._train_loss_sum / 125
                avg_train_acc = self._train_acc_sum / 125

                if dist.get_rank() == 0:
                    with open(self.epoch_log_file, "a") as f:
                        # 写入训练指标，不要换行，等待验证指标补齐
                        f.write(f"Epoch {self.epoch_idx} \t Train Loss: {avg_train_loss:.4f} \t Train Acc: {avg_train_acc:.2f}% ")
                    
                self._train_steps, self._train_loss_sum, self._train_acc_sum = 0, 0.0, 0.0
        else:
            self._val_steps += 1
            self._val_loss_sum += c_loss
            self._val_acc_sum += c_acc

            if self._val_steps == 25:
                avg_val_loss = self._val_loss_sum / 25
                avg_val_acc = self._val_acc_sum / 25

                # 计算距离训练开始的time，单位秒
                global _TRAIN_START_TIME
                elapsed_time = time.time() - _TRAIN_START_TIME

                if dist.get_rank() == 0:
                    with open(self.epoch_log_file, "a") as f:
                        # 补齐验证数据，并加 \n 回车换行
                        f.write(f"\t Val Loss: {avg_val_loss:.4f} \t Val Acc: {avg_val_acc:.2f}% \t Time: {elapsed_time:.2f}s\n")
                
                # 重置验证累加器并增加 Epoch 计数
                self._val_steps, self._val_loss_sum, self._val_acc_sum = 0, 0.0, 0.0
                self.epoch_idx += 1 

        return loss, {"cls_loss": averaged_stats[0], "top1_acc": averaged_stats[1]}

    def forward_step(self, data_iterator, model):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (ViTModel): The ViT Model
        """
        timers = get_timers()
        # 获取 Batch
        # 这是 ViT 特定的 get_batch，只返回 images 和 labels
        timers('batch-generator', log_level=2).start()
        images, labels = self.get_batch(data_iterator)
        timers('batch-generator').stop()

        output_tensor = model(images)
        return output_tensor, partial(self.loss_func, labels=labels)

    def train(self):
        """Main training loop."""
        args = get_args()
        test_data_iterator = self.test_data_iterator_list[0]
        forward_step_func, model, optimizer, opt_param_scheduler, train_data_iterator, valid_data_iterator, process_non_loss_data_func, config = self.train_args
        
        if not args.skip_train:
            print_rank_0('training ...')
            iteration = 0
            if args.do_train and args.train_iters > 0:
                iteration, num_floating_point_operations_so_far = megatron_train(*self.train_args)

            print_datetime('after training is done')

            if args.save and iteration != 0 and iteration % args.save_interval != 0:
                save_checkpoint(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far
                )

        # if args.do_valid and valid_data_iterator is not None:
        #     prefix = f'iteration {iteration} on validation set'
        #     evaluate_and_print_results(prefix, forward_step_func,
        #                                valid_data_iterator, model,
        #                                iteration, process_non_loss_data_func, config,
        #                                verbose=True, write_to_tensorboard=not args.skip_train)

        # # 5. 最终测试集评估 (Final Testing)
        # if args.do_test and test_data_iterator is not None:
        #     prefix = f'iteration {iteration} on test set'
        #     evaluate_and_print_results(prefix, forward_step_func,
        #                                test_data_iterator, model,
        #                                iteration, process_non_loss_data_func, config,
        #                                verbose=True, write_to_tensorboard=not args.skip_train)