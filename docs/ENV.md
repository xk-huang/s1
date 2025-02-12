# Env

## Conda

Install mini conda
```shell
BASE_DIR=

cd $BASE_DIR/misc
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -p $BASE_DIR/misc/miniconda3

# source ~/.bashrc
source activate base
rm Miniconda3-latest-Linux-x86_64.sh
```


Follow `nvcr.io/nvidia/pytorch:24.06-py3` in https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-06.html

We assume the cuda is 12.4.

```bash
# Install python
conda create -n xiaoke-m1 python=3.10
source activate base
conda activate xiaoke-m1
which python
# should be `$BASE_DIR/misc/miniconda/envs/xiaoke-m1/bin/python`

# Install pytorch 2.4.0
# https://pytorch.org/get-started/locally/
# We use cuda 12.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
# conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt
```

Install flash-attn https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features.

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

pip install packaging ninja
pip uninstall -y ninja && pip install ninja
pip install flash-attn --no-build-isolation
```

## Eval

```bash
mkdir -p third_party/
git clone --depth 1 git@github.com:EleutherAI/lm-evaluation-harness.git third_party/lm-evaluation-harness
cd third_party/lm-evaluation-harness/
git fetch --depth 1 origin 4cec66e4e468d15789473d6d63c3a61a751fa524
git checkout 4cec66e4e468d15789473d6d63c3a61a751fa524
```