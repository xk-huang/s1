from transformers import AutoTokenizer, HfArgumentParser


model_name = "Qwen/Qwen2.5-32B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)


dialog = [{"role" : "user", "content": "What is the capital of France?"}]

prompt = f"{tokenizer.apply_chat_template(dialog, tokenize=False)}<|im_start|>assistant\n"

text = tokenizer.apply_chat_template(
    dialog,
    tokenize=False,
    add_generation_prompt=True
)

system_dialog =[{"role": "system", "content": "You are ET."},{"role" : "user", "content": "What is the capital of France?"}]

system_text = tokenizer.apply_chat_template(
    system_dialog,
    tokenize=False,
    add_generation_prompt=True
)

tokenizer.apply_chat_template(
    system_dialog,
    tokenize=False,
    add_generation_prompt=False
)

# fmt: off
import IPython; IPython.embed()
# fmt: on

