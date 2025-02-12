# RUN

## Prerequisite

Use a `.env` file

```bash
WANDB_API_KEY=?
WANDB_PROJECT=s1
WANDB_MODE=online  # online, offline, disabled; ref: https://docs.wandb.ai/ref/python/init/
WANDB_ENTITY=xk-huang

HF_TOKEN=? # read-only
HF_HOME=cache/
```

To use them without `dotenv`:

```bash
set -a
. .env
set +a
```

## Run SFT

```bash
bash train/sft.sh
```

Edit `base_model` to try smaller one, e.g., `Qwen/Qwen2.5-0.5B-Instruct`.


### Code Design

Use `transformers`, `datasets`, and `trl`.

1. config: `transformers.HfArgumentParser`
2. dataset: huggingface `datasets`
3. tokenizer: `transformers`
4. data collator, applying target template: `trl.DataCollatorForCompletionOnlyLM`
5. SFT training: `trl.SFTTrainer`


TODO: add flash-attn v2
