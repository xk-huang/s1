# DATA

change data and model directory of huggingface
https://stackoverflow.com/questions/63312859/how-to-change-huggingface-transformers-default-cache-directory

```bash
cache_dir=cache/
cache_dir="$(realpath $cache_dir)"

mkdir -p "${cache_dir}"
export HF_HOME="${cache_dir}"
```


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
echo $HF_HOME
```

## s1K Sample Format

Example https://huggingface.co/datasets/simplescaling/s1K_tokenized
- update version: https://huggingface.co/datasets/simplescaling/s1K-1.1_tokenized

```python
from datasets import load_dataset
ds = load_dataset("simplescaling/s1K-1.1_tokenized")["train"]
ds[0]

import json
from pathlib import Path
output_dir = Path("misc/")
output_dir.mkdir(exist_ok=True, parents=True)
output_path = output_dir / "s1K-1.1_tokenized-sample.json"

with open(output_path, "w") as f:
    json.dump(ds[0], f, indent=4)
print(output_path)
```