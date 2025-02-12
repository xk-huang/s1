# DATA

change data and model directory of huggingface
https://stackoverflow.com/questions/63312859/how-to-change-huggingface-transformers-default-cache-directory

```bash
cache_dir=cache/
cache_dir="$(realpath $cache_dir)"

mkdir -p "${cache_dir}"
export HF_HOME="${cache_dir}"
```
