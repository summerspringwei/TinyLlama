import math
import matplotlib.pyplot as plt

# Hyperparameters
num_of_devices = 2
global_batch_size = 128
learning_rate = 4e-4
micro_batch_size = 16
max_step = 7000
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

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

for it in range(0, max_iters, 100):
    print(it, get_lr(it))

it_list = range(0, max_iters, 100)
lr_list = [get_lr(it) for it in range(0, max_iters, 100)]

# Plotting the line chart
plt.plot(it_list, lr_list, label='TinyLlama Pretrain Learning Rate Line Chart', marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Simple Line Chart')

# Adding a legend
plt.legend()
plt.savefig('TinyLlama-Pretrain-Learning-Rate-Line-Chart.png')