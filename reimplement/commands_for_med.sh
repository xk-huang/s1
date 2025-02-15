lm_eval --model dummy --tasks medqa_4options --batch_size auto --output_path dummy --log_samples --gen_kwargs "max_gen_toks=32768"



lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,tokenizer=Qwen/Qwen2.5-0.5B-Instruct,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.4 --tasks medqa_4options --batch_size 10 --apply_chat_template --output_path outputs/qwen --log_samples --gen_kwargs "max_gen_toks=32768"

lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-32B-Instruct,tokenizer=Qwen/Qwen2.5-32B-Instruct,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=500 --tasks medqa_4options --batch_size 2 --apply_chat_template --output_path outputs/qwen --log_samples --gen_kwargs "max_gen_toks=32768"

lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-32B-Instruct,tokenizer=Qwen/Qwen2.5-32B-Instruct,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=500,enforce_eager=True --tasks medqa_4options --batch_size 2 --apply_chat_template --output_path outputs/qwen --log_samples --gen_kwargs "max_gen_toks=32768"