python scripts/convert_hf_checkpoint.py --checkpoint_dir  out/TinyLlama-1.1B-900B --model_name tiny_LLaMA_1b

python test_weight.py --checkpoint_dir out/TinyLlama-1.1B-intermediate-900B


python pretrain/tinyllama_code.py --devices 8 --train_data_dir data/code_specialist_python_java_javascript_c_go_8192



python scripts/prepare_starcoder.py --source_path data/starcoderdata/ --tokenizer_path data/llama --destination_path data/code_specialist_python_java_javascript_c_go_8192 --split train --percentage 1.0 --filenames_subset ["python","cpp","go","java","javascript"] --chunk_size 4194816




/data/TinyLlama/out/code_tiny_LLaMA_1b_python_java_go_cpp_javascript/iter-032000-ckpt.pth

python scripts/convert_lit_checkpoint.py --out_dir /data/TinyLlama/out/tiny_LLaMA_1b/ --checkpoint_name iter-100000-ckpt.pth --model_name tiny_LLaMA_1b


python3 scripts/prepare_redpajama.py \
    --source_path /data/xiachunwei/Dataset/RedPajama-Data-1T-Sample \
    --checkpoint_dir /home/xiachunwei/Dataset/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --destination_path /home/xiachunwei/Dataset/RedPajama-Data-1T-Sample-Bin


lightning run model \
    --node-rank=0  \
    --main-address=127.0.0.1 \
    --accelerator=cuda \
    --devices=1 \
    --num-nodes=1 \
    pretrain/tinyllama.py --devices 1 \
    --train_data_dir /home/xiachunwei/Dataset/RedPajama-Data-1T-Sample-Bin  \
    --val_data_dir /home/xiachunwei/Dataset/RedPajama-Data-1T-Sample-Bin


lightning run model \
    --node-rank=0  \
    --main-address=127.0.0.1 \
    --accelerator=cuda \
    --devices=1 \
    --num-nodes=1 \
    pretrain/tinyllama.py --devices 1 \
    --train_data_dir /workspace/Dataset/RedPajama-Data-1T-Sample-Bin  \
    --val_data_dir /workspace/Dataset/RedPajama-Data-1T-Sample-Bin
