# draftbench — A100 GCP Setup TODO

## Step 1: Create GCP Instance

- Machine type: `a2-highgpu-1g` (1x A100 40GB)
- OS: Ubuntu 22.04 LTS
- Boot disk: 200GB SSD (models are large)
- Firewall: allow SSH, optionally port 8000 if benchmarking remotely

```bash
gcloud compute instances create draftbench-a100 \
  --machine-type=a2-highgpu-1g \
  --zone=us-central1-c \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --maintenance-policy=TERMINATE
```

---

## Step 2: Install Docker + NVIDIA Container Toolkit

SSH into the machine, then:

```bash
# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access works
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## Step 3: Set Up draftbench + Download Models

```bash
# draftbench — no vLLM venv needed, just requests
git clone https://github.com/djpinger/draftbench
cd draftbench
pip install requests

# Pull vLLM Docker image
docker pull vllm/vllm-openai:latest

# Create models directory
mkdir -p ~/draftbench/models

# Download models (hf = huggingface-cli)
pip install huggingface-hub

# Target: Qwen3-14B-AWQ (~8 GB)
# https://huggingface.co/Qwen/Qwen3-14B-AWQ
hf download Qwen/Qwen3-14B-AWQ --local-dir ~/draftbench/models/Qwen3-14B-AWQ

# Draft: EAGLE3 speculative model (~1 GB)
# https://huggingface.co/yuhuili/EAGLE3-Qwen3-14B-Instruct
hf download yuhuili/EAGLE3-Qwen3-14B-Instruct --local-dir ~/draftbench/models/Qwen3-14B-eagle3

# Draft: Qwen3-1.7B (~4 GB)
# https://huggingface.co/Qwen/Qwen3-1.7B
hf download Qwen/Qwen3-1.7B --local-dir ~/draftbench/models/Qwen3-1.7B

# Draft: Qwen3-4B-AWQ (~3 GB)
# https://huggingface.co/Qwen/Qwen3-4B-AWQ
hf download Qwen/Qwen3-4B-AWQ --local-dir ~/draftbench/models/Qwen3-4B-AWQ

# Draft: Qwen3-8B-AWQ (~5 GB)
# https://huggingface.co/Qwen/Qwen3-8B-AWQ
hf download Qwen/Qwen3-8B-AWQ --local-dir ~/draftbench/models/Qwen3-8B-AWQ
```

---

## Step 4: Configure and Run Sweep

Copy `configs/qwen3-14b-vllm.json` to the new machine (or sync via git pull).
Update `"hardware"` to match the new machine label, and add `"docker_image"` to settings:

```json
{
  "name": "qwen3-14b-vllm",
  "hardware": "a100-40g",
  "backend": "vllm",
  "model_family": "Qwen3 14B",

  "targets": [
    {"label": "14B AWQ", "path": "/home/paul/draftbench/models/Qwen3-14B-AWQ"}
  ],
  "drafts": [
    {"label": "EAGLE3",  "path": "/home/paul/draftbench/models/Qwen3-14B-eagle3", "method": "eagle3"},
    {"label": "1.7B",    "path": "/home/paul/draftbench/models/Qwen3-1.7B",        "method": "draft_model"},
    {"label": "4B AWQ",  "path": "/home/paul/draftbench/models/Qwen3-4B-AWQ",      "method": "draft_model"},
    {"label": "8B AWQ",  "path": "/home/paul/draftbench/models/Qwen3-8B-AWQ",      "method": "draft_model"}
  ],
  "settings": {
    "port": 8000,
    "runs": 3,
    "max_tokens": 512,
    "temperature": 0.0,
    "num_speculative_tokens": 5,
    "docker_image": "vllm/vllm-openai:latest",
    "extra_args": ["--max-model-len", "8192", "--max-num-seqs", "8"]
  }
}
```

> Note: `--enforce-eager` and tight memory flags are no longer needed on A100 40GB.
> Adjust `max-model-len` and `max-num-seqs` upward as appropriate.

```bash
cd ~/draftbench
python sweep.py --config configs/qwen3-14b-vllm.json
```

Results and HTML chart saved to `results/`.
