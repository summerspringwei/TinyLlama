import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from lit_gpt.model import GPT, Block, Config, CausalSelfAttention

# config = Config.from_name("tiny_LLaMA_1b")
# print(config)
# exit(0)

def print_dict_recursive(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print("  " * indent + f"{k}:")
            print_dict_recursive(v, indent + 1)
        elif isinstance(v, torch.Tensor):
            print("  " * indent + f"{k}: {v.shape}, {v.dtype}")


def load_model(model_name = "tiny_LLaMA_1b", checkpoint_path = 'out/tinyllama_1b/iter-024000-ckpt.bin'):
    state = torch.load(checkpoint_path)
    print_dict_recursive(state)
    config = Config.from_name(model_name)
    model = GPT(config)
    model.load_state_dict(state["model"])
    model.eval().to("cuda")
    return model


def load_tokenizer(model_path = "/workspace/Dataset/TinyLlama-1.1B-step-50K-105b/"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return tokenizer

code = '''
#define NULL ((void*)0)
typedef unsigned long size_t;  // Customize by platform.
typedef long intptr_t; typedef unsigned long uintptr_t;
typedef long scalar_t__;  // Either arithmetic or pointer type.
/* By default, we understand bool (as a convenience). */
typedef int bool;
#define false 0
#define true 1

/* Forward declarations */

/* Type definitions */
struct ring_buffer {int /*<<< orphan*/  poll; } ;
struct perf_event {int /*<<< orphan*/  mmap_mutex; struct ring_buffer* rb; int /*<<< orphan*/  waitq; } ;
struct file {struct perf_event* private_data; } ;
typedef  int /*<<< orphan*/  poll_table ;
typedef  int /*<<< orphan*/  __poll_t ;

/* Variables and functions */
 int /*<<< orphan*/  EPOLLHUP ; 
 int /*<<< orphan*/  atomic_xchg (int /*<<< orphan*/ *,int /*<<< orphan*/ ) ; 
 scalar_t__ is_event_hup (struct perf_event*) ; 
 int /*<<< orphan*/  mutex_lock (int /*<<< orphan*/ *) ; 
 int /*<<< orphan*/  mutex_unlock (int /*<<< orphan*/ *) ; 
 int /*<<< orphan*/  poll_wait (struct file*,int /*<<< orphan*/ *,int /*<<< orphan*/ *) ; 

__attribute__((used)) static __poll_t perf_poll(struct file *file, poll_table *wait)
{
	struct perf_event *event = file->private_data;
	struct ring_buffer *rb;
	__poll_t events = EPOLLHUP;

	poll_wait(file, &event->waitq, wait);

	if (is_event_hup(event))
		return events;

	/*
	 * Pin the event->rb by taking event->mmap_mutex; otherwise
	 * perf_event_set_output() can swizzle our rb and make us miss wakeups.
	 */
	mutex_lock(&event->mmap_mutex);
'''

# 	rb = event->rb;
# 	if (rb)
# 		events = atomic_xchg(&rb->poll, 0);
# 	mutex_unlock(&event->mmap_mutex);
# 	return events;
# }

# code = '''
# 0000000000000000 tool_mime_new:
# ; {
#        0:       pushq   %rbp
#        1:       pushq   %rbx
#        2:       pushq   %rax
#        3:       movl    %esi, %ebp
#        5:       movq    %rdi, %rbx
# ;   tool_mime *m = (tool_mime *) calloc(1, sizeof(*m));
#        8:       movl    $1, %edi
#        d:       movl    $32, %esi
#       12:       callq   0 <tool_mime_new+0x17>
# ;   if(m) {
#       17:       testq   %rax, %rax
#       1a:       je      22 <tool_mime_new+0x32>
# ;     m->kind = kind;
#       1c:       movl    %ebp, 24(%rax)
# ;     m->parent = parent;
#       1f:       movq    %rbx, 16(%rax)
# '''
# ;     if(parent) {
#       23:       testq   %rbx, %rbx
#       26:       je      10 <tool_mime_new+0x32>
# ;       m->prev = parent->subparts;
#       28:       movq    (%rbx), %rcx
#       2b:       movq    %rcx, 8(%rax)
# ;       parent->subparts = m;
#       2f:       movq    %rax, (%rbx)
# ;   return m;
#       32:       addq    $8, %rsp
#       36:       popq    %rbx
#       37:       popq    %rbp
#       38:       retq

tokenizer = load_tokenizer()
# ids = tokenizer(code, return_tensors="pt")
# print(tokenizer)
# print(tokenizer.tokenize(code))
# model = load_model().to(torch.bfloat16)
# print(ids["input_ids"])
# out = model(ids["input_ids"].to("cuda"))
# print(out)
# out = tokenizer.batch_decode(out)
# print(out)


def run_model(model, code):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # checkpoint_path = 'out/tinyllama_1b_c_assembly_scratch/iter-030000-ckpt.bin'
    # pipeline.model.load_state_dict(torch.load(checkpoint_path), strict=False)
    sequences = pipeline(
        # 'The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens. With some proper optimization, we can achieve this within a span of "just" 90 days using 16 A100-40G GPUs 🚀🚀. The training has started on 2023-09-01.',
        code,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2048,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


if __name__=="__main__":
    run_model("/workspace/Dataset/TinyLlama-1.1B-C-assembly/", code)