import json

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

# code = "assembly: \n0000000000000000 <_start>:\n0: \tpushq\t%rbp\n1: \tpushq\t%rbx\n2: \tpushq\t%rax\n3: \tmovq\t%rsi, %rbx\n6: \tmovl\t%edi, %ebp\n8: \tmovl\t$0, %esi\nd: \txorl\t%edi, %edi\nf: \tcallq\t0x14 <_start+0x14>\n14: \tmovq\t%rax, (%rip)  # 0x1b <_start+0x1b>\n1b: \tmovl\t$0, %esi\n20: \tmovl\t$1, %edi\n25: \tcallq\t0x2a <_start+0x2a>\n2a: \tmovq\t%rax, (%rip)  # 0x31 <_start+0x31>\n31: \tmovl\t$0, %esi\n36: \tmovl\t$2, %edi\n3b: \tcallq\t0x40 <_start+0x40>\n40: \tmovq\t%rax, (%rip)  # 0x47 <_start+0x47>\n47: \tmovq\t$0, (%rip)    # 0x52 <_start+0x52>\n52: \tmovq\t$0, (%rip)    # 0x5d <_start+0x5d>\n5d: \tmovl\t%ebp, %edi\n5f: \tmovq\t%rbx, %rsi\n62: \tcallq\t0x67 <_start+0x67>\n67: \tmovl\t%eax, %edi\n69: \tcallq\t0x6e <_start+0x6e> \n###"

def read_codes(file_path):
    code_list = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            com = json.loads(line)["text"].split("### c:")
            if len(com) > 1:
                code_list.append((com[0], com[1]))
    return code_list


            


# ids = tokenizer(code, return_tensors="pt")
# print(tokenizer)
# print(tokenizer.tokenize(code))
# model = load_model().to(torch.bfloat16)
# print(ids["input_ids"])
# out = model(ids["input_ids"].to("cuda"))
# print(out)
# out = tokenizer.batch_decode(out)
# print(out)



def run_model(model,tokenizer, code):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        # device_map="auto"
        device=1
    )
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


def main():
    model_path = "/workspace/Dataset/TinyLlama-1.1B-C-assembly/"
    code_list = read_codes("/workspace/Dataset/decompilation-dataset/AnghaBench_text_train_paired_assembly-g-O2_C_2K_sample128.json")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    for code in code_list:
        print("="*20)
        print(code[1])
        run_model(model, tokenizer, code[0])
        print("="*20)

if __name__=="__main__":
    # run_model("/workspace/Dataset/TinyLlama-1.1B-C-assembly/", code)
    main()