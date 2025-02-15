# EVAL

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

## Env

Clone the original lm-evluation-harness to see the difference compared with `eval/lm-evaluation-harness`

```bash
mkdir -p third_party/
git clone --depth 1 git@github.com:EleutherAI/lm-evaluation-harness.git third_party/lm-evaluation-harness
cd third_party/lm-evaluation-harness/
git fetch --depth 1 origin 4cec66e4e468d15789473d6d63c3a61a751fa524
git reset --hard 4cec66e4e468d15789473d6d63c3a61a751fa524

rsync -avp eval/lm-evaluation-harness/ third_party/lm-evaluation-harness/
code third_party/lm-evaluation-harness/
```

Changes:
- TBD


## Run

~~All of these evaluations you should be in the `eval/lm-evaluation-harness` directory and have the dependencies there installed.~~

Check commands in `eval/commands.sh`

Remove in `eval/commands.sh`
```
OPENAI_API_KEY=YOUR_OPENAI_KEY 
```