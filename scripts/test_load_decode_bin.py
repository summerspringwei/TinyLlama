from pathlib import Path
from lit_gpt.model import GPT, Block, Config
from lit_gpt import Tokenizer
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from pytorch_lightning.loggers import WandbLogger
import lightning as L
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pretrain.tinyllama import create_dataloaders

model_name = "tiny_LLaMA_1b"
name = "tinyllama_1b"
out_dir = Path("out") / name

# Hyperparameters
num_of_devices = 1
global_batch_size = 128
learning_rate = 4e-4
micro_batch_size = 8
# max_step = 7152 * 2
number_of_samples = 1000000
max_step = number_of_samples // global_batch_size
warmup_steps = 200
log_step_interval = 10
eval_iters = 10
save_step_interval = 1000
eval_step_interval = 1000


weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 4e-5

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps


max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps

logger = step_csv_logger("out", name, flush_logs_every_n_steps=log_iter_interval)
precision = get_default_supported_precision(training=True, tpu=False)
print(precision)
devices = 1
strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
wandb_logger = WandbLogger()
fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])

train_data_config = [
    ("AnghaBench", 1.0),
]

train_data_dir = Path("/workspace/Dataset/decompilation-dataset/AnghaBench-C-bin")
val_data_dir = train_data_dir
config = Config.from_name(model_name)

train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
    )

train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

checkpoint_dir = Path("/workspace/Dataset/TinyLlama-1.1B-step-50K-105b/")
tokenizer = Tokenizer(checkpoint_dir)
for train_data in train_dataloader:
    # print(train_data)
    input_ids = train_data[:, 0 : 2048].contiguous()
    targets = train_data[:, 1 : 2048 + 1].contiguous()
    print(input_ids.shape)
    for i in range(input_ids.shape[0]):
        print("xx"*30)
        print(tokenizer.decode(input_ids[i]))
        print("xx"*30)
    # print(input_ids)
    break
