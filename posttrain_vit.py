from mindspeed_llm import megatron_adaptor
from mindspeed_llm.tasks.posttrain.launcher import AutoTrainer


def launch():
    trainer = AutoTrainer()
    trainer.train()


if __name__ == '__main__':
    launch()