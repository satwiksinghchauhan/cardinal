# Cardinal v1.4.0 — Installation Guide

**Linux only. Offline builds. Ubuntu 24.04 LTS recommended.**

Cardinal v1.4.0 adds the SEAL self-improvement system (three-layer: self-model, meta-cognition, LoRA fine-tuning). The build remains fully offline. The training pipeline requires an additional Python venv with PEFT — this is only needed if you want Layer 3 (LoRA fine-tuning) to actually run. Layers 1 and 2 work with no Python dependency.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Detailed Installation](#detailed-installation)
  - [System Dependencies](#step-1--system-dependencies)
  - [CUDA Toolkit](#step-2--cuda-toolkit-126)
  - [Vendored Dependencies](#step-3--vendored-dependencies)
  - [Download Models](#step-4--download-models)
  - [Build llama.cpp](#step-5--build-llamacpp-with-cuda-and-mtmd)
  - [Build Cardinal](#step-6--build-cardinal)
  - [Training Venv (Layer 3)](#step-7--training-python-venv-layer-3-optional)
- [First Run](#first-run)
- [Verify Self-Improvement (v1.4.0)](#verify-self-improvement-v140)
- [Verify Vision (v1.3.0)](#verify-vision-v130)
- [Verify the HTTP API](#verify-the-http-api)
- [Troubleshooting](#troubleshooting)
- [Changing the API Key](#changing-the-api-key)
- [Directory Reference](#directory-reference)

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 22.04 LTS | **Ubuntu 24.04 LTS** |
| CPU | Any x64, 4 cores | AMD Ryzen 7 or better |
| RAM | 8GB | 16GB (32GB for training) |
| GPU | NVIDIA with 4GB VRAM | RTX 3050 or better |
| Storage | 25GB free | 50GB free (HF weights add ~8GB) |
| CUDA Compute | 7.5+ | 8.6 (RTX 30 series) |
| Python | 3.10+ | 3.12 (for training venv) |

Cardinal does **not** run on AMD or Intel GPUs. NVIDIA CUDA is required.

**Windows is not supported.**

---

## Quick Start

```bash
# Install system dependencies
sudo apt update
sudo apt install -y build-essential cmake libsqlite3-dev libssl-dev swi-prolog python3 python3-venv

# Clone Cardinal
git clone https://github.com/satwiksinghchauhan/cardinal ~/cardinal
cd ~/cardinal

# Populate vendor dependencies
./scripts/populate_vendor.sh   # or clone manually (see Step 3)

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# (Optional) Set up training venv for Layer 3
python3 -m venv ~/cardinal/cardinal-train-venv
source ~/cardinal/cardinal-train-venv/bin/activate
pip install peft transformers torch accelerate
deactivate

# Run
cd ~/cardinal
./build/bin/cardinal
```

---

## Detailed Installation

### Step 1 — System Dependencies

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libssl-dev \
    libsqlite3-dev \
    swi-prolog \
    python3 \
    python3-venv
```

Verify:

```bash
cmake --version    # must be 3.28 or higher
swipl --version    # SWI-Prolog 9.2.x or newer
openssl version    # 3.0.x or newer
python3 --version  # 3.10 or newer
```

If your distro ships CMake older than 3.28:

```bash
sudo apt remove cmake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt update
sudo apt install cmake
```

---

### Step 2 — CUDA Toolkit 12.6

Install the NVIDIA driver first if not already installed:

```bash
ubuntu-drivers autoinstall
sudo reboot
```

After reboot, install CUDA:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6
```

Add CUDA to your PATH:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
nvcc --version    # release 12.6
nvidia-smi        # shows your GPU
```

---

### Step 3 — Vendored Dependencies

| Dependency | URL | Clone as |
|:---|:---|:---|
| llama.cpp | https://github.com/ggerganov/llama.cpp | `vendor/llama.cpp` |
| nlohmann/json | https://github.com/nlohmann/json | `vendor/nlohmann_json` |
| cpp-httplib | https://github.com/yhirose/cpp-httplib | `vendor/cpp-httplib` |
| muparser | https://github.com/beltoforion/muparser | `vendor/muparser` |
| tokenizers-cpp | https://github.com/mlc-ai/tokenizers-cpp | `vendor/tokenizers-cpp` |


```bash
cd ~/cardinal
mkdir -p vendor && cd vendor

git clone https://github.com/ggerganov/llama.cpp.git
git clone https://github.com/nlohmann/json.git nlohmann_json
git clone https://github.com/yhirose/cpp-httplib.git
git clone https://github.com/beltoforion/muparser.git
git clone https://github.com/mlc-ai/tokenizers-cpp
```

Or use the provided script:

```bash
cd ~/cardinal
chmod +x scripts/populate_vendor.sh
./scripts/populate_vendor.sh
```

---

### Step 4 — Download Models

#### Primary LLM (required)

- **Qwen3.5 4B Q4_K_M**
- From: [bartowski/Qwen_Qwen3.5-4B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF)
- File: `Qwen_Qwen3.5-4B-Q4_K_M.gguf` (~2.5GB)
- Place at: `~/cardinal/models/Qwen_Qwen3.5-4B-Q4_K_M.gguf`

#### Vision encoder (v1.3.0, optional but recommended)

```bash
mkdir -p ~/cardinal/models/vision
cd ~/cardinal/models/vision

wget https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-text-model-f16.gguf
wget https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-mmproj-f16.gguf
```

If those links don't work, try: https://huggingface.co/salivosa/moondream2-gguf

#### HuggingFace weights for training (Layer 3, optional)

Layer 3 fine-tuning requires the HuggingFace format weights (not GGUF) to run PEFT:

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-4B', local_dir='models/qwen3.5-4b-hf')
"
```

This downloads ~8GB. Only needed if you want LoRA fine-tuning to actually execute. Cardinal works fully without it — Layers 1 and 2 have no Python or HF dependency.

---

### Step 5 — Build llama.cpp with CUDA and mtmd

```bash
cd ~/cardinal/vendor/llama.cpp
mkdir -p build && cd build
cmake .. \
    -DGGML_CUDA=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Verify:

```bash
ls build/bin/
# Should include libllama.so, libggml.so, libmtmd.so (or similar)
```

---

### Step 6 — Build Cardinal

```bash
cd ~/cardinal
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

A successful build prints:

```
-- =============================================================
-- Cardinal v1.4.0 Configuration
-- =============================================================
-- Build type:         Release
-- CUDA arch:          75
-- TensorRT backend:   OFF
-- Features:           tools + agent + explainability + vision
--                     + self-improvement (layers 1-3)
-- Output dir:         .../build/bin
-- =============================================================
```

#### Optional: TensorRT backend

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCARDINAL_ENABLE_TENSORRT=ON \
    -DTRT_LLM_INCLUDE_DIR=/path/to/TensorRT-LLM/include \
    -DTRT_LLM_LIB_DIR=/path/to/TensorRT-LLM/lib
make -j$(nproc)
```

With TensorRT, Layer 3 runs in script-export mode: instead of launching a training subprocess, Cardinal writes a ready-to-run shell script to `data/training/scripts/` for your cluster to execute.

---

### Step 7 — Training Python Venv (Layer 3, optional)

This step is only needed if you want Cardinal to actually run LoRA fine-tuning locally (Layer 3, `LlamaCppTrainer`). Skip if you're using TensorRT backend (script-export mode) or want only Layers 1 and 2.

```bash
python3 -m venv ~/cardinal/cardinal-train-venv
source ~/cardinal/cardinal-train-venv/bin/activate

pip install peft transformers torch accelerate

# Install the cardinal_train module (runs the actual PEFT training)
pip install -e ~/cardinal/scripts/cardinal_train   # if provided
# or manually install your PEFT training script as a module

deactivate
```

Verify the venv is configured correctly in `config.json`:

```json
"training": {
    "python_venv": "~/cardinal/cardinal-train-venv",
    "hf_model_path": "models/qwen3.5-4b-hf",
    "convert_lora_script": "vendor/llama.cpp/convert_lora_to_gguf.py"
}
```

---

## First Run

Run from the Cardinal root directory so relative paths in `config.json` resolve:

```bash
cd ~/cardinal
./build/bin/cardinal
```

Expected startup output:

```
[INFO ] SelfModel: opened database at data/self_model/self_model.db
[INFO ] MetaCognition: initialised (enabled=true, trigger_every=20)
[INFO ] LlamaCppTrainer: initialised (venv=~/cardinal/cardinal-train-venv)
[INFO ] SelfImprovementLoop: started
[INFO ] VisionEncoder: ready (CPU, 4 threads)
[INFO ] LlamaCppBackend ready - vocab: 151936, ctx: 8192
...
  +===========================================+
  |        C A R D I N A L  v1.4.0            |
  |   Neurosymbolic AGI Architecture          |
  +===========================================+
```

---

## Verify Self-Improvement (v1.4.0)

### Layer 1 — Self-Model

Send a few messages, then query the self-model endpoint:

```bash
curl -s -X POST http://127.0.0.1:8080/api/chat \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","message":"What is the capital of France?"}'

curl -s http://127.0.0.1:8080/api/self_model \
  -H "Authorization: Bearer secret_api_key" | python3 -m json.tool
```

Expected response includes `weakest_domain`, `strongest_domain`, `total_domain_stats`.

You can also inspect the SQLite database directly:

```bash
sqlite3 data/self_model/self_model.db \
  "SELECT domain, total_inferences, confidence_sum/total_inferences AS avg_conf \
   FROM domain_stats;"
```

### Layer 2 — Meta-Cognition

Trigger on-demand reflection:

```bash
curl -s -X POST http://127.0.0.1:8080/api/reflect \
  -H "Authorization: Bearer secret_api_key" | python3 -m json.tool
```

If `ran=false`, not enough failure episodes yet (need `min_failures_to_reflect=5`). To test quickly, temporarily lower the threshold in `config.json`:

```json
"meta_cognition": { "min_failures_to_reflect": 1 }
```

After a successful reflection, check for corrective rules:

```bash
curl -s http://127.0.0.1:8080/api/rules \
  -H "Authorization: Bearer secret_api_key" | \
  python3 -c "
import sys, json
rules = json.load(sys.stdin)
for r in rules:
    if r.get('reasoning_type') == 'meta_correction':
        print(r['domain'], '|', r['condition'][:60])
"
```

### Layer 3 — Training

Trigger a training cycle (runs asynchronously):

```bash
curl -s -X POST http://127.0.0.1:8080/api/train \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"domain_hint":""}' | python3 -m json.tool
```

Watch the log:

```bash
tail -f logs/cardinal.log | grep -E "Trainer|SelfImprovement|Curriculum|Dataset"
```

The training cycle requires `min_episodes_for_training=50` by default. To test pipeline plumbing with fewer episodes:

```json
"training": { "min_episodes_for_training": 2 }
```

---

## Verify Vision (v1.3.0)

Place a test image in `data/` and ask:

```
You: describe the image at data/test.jpg
```

---

## Verify the HTTP API

```bash
# Health check — no auth required
curl http://127.0.0.1:8080/api/health

# Chat
curl -X POST http://127.0.0.1:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer secret_api_key" \
  -d '{"session_id":"test","message":"Hello!"}'

# Self-model status (v1.4.0)
curl http://127.0.0.1:8080/api/self_model \
  -H "Authorization: Bearer secret_api_key"

# Stats (includes self_improvement block in v1.4.0)
curl http://127.0.0.1:8080/api/stats \
  -H "Authorization: Bearer secret_api_key"
```

---

## Troubleshooting

### Self-model DB not created

Ensure `config.json` has `self_improvement.enabled=true` and `self_improvement.self_model.enabled=true`. The DB is created on first chat, not on startup.

### Reflection returns `ran=false`

Not enough failure episodes. Lower `min_failures_to_reflect` to 1 for testing. Failure episodes are those where `contradiction=true` or `uncertainty=true`.

### Training returns `accepted=false`

Either training is disabled in config, or a cycle is already running. Check the log for `training_in_progress`.

### `cardinal_train` module not found

The PEFT training subprocess expects a `cardinal_train` Python module installed in the venv. This is the PEFT training script. Install it:

```bash
source ~/cardinal/cardinal-train-venv/bin/activate
pip install peft transformers torch accelerate
# Install your cardinal_train wrapper script as a module
```

### Training fails: `hf_model_path not found`

The HuggingFace weights must be present at the path in `config.json`. Download them (see Step 4) or disable Layer 3: `"training": {"enabled": false}`.

### `mtmd.h not found` during CMake

Build llama.cpp with `-DLLAMA_BUILD_EXAMPLES=ON`. See Step 5.

### `libmtmd.so not found` at runtime

```bash
export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/bin:$LD_LIBRARY_PATH
```

Make it permanent:

```bash
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### `CUDA out of memory` on startup

Reduce `gpu_layers` in `config.json`. Vision and training run on CPU.

---

## Changing the API Key

```json
"api": {
    "auth_enabled": true,
    "api_key": "your-strong-random-key-here"
}
```

To disable auth for local development:

```json
"api": {
    "auth_enabled": false
}
```

---

## Directory Reference (v1.4.0)

```
~/cardinal/
    build/
        bin/
            cardinal
    data/
        memory/
            rules.json
            knowledge.json
            episodes.db
            agent_working_memory
        self_model/                    ← new in v1.4.0
            self_model.db              ← Layer 1 SQLite accumulator
        training/                      ← new in v1.4.0
            adapters/                  ← trained GGUF adapters
            datasets/                  ← curated JSONL datasets
            scripts/                   ← TensorRT training scripts
        explainability/
            audit.db
            cardinal_private.pem
            cardinal_public.pem
            exports/
        vision_cache/
    logs/
        cardinal.log
        episodic.log
    models/
        Qwen_Qwen3.5-4B-Q4_K_M.gguf
        qwen3.5-4b-hf/                 ← HF weights for training (Layer 3)
        Llama-3.2-1B-Instruct-Q4_K_M.gguf   (optional)
        vision/
            moondream2-text-model-f16.gguf
            moondream2-mmproj-f16.gguf
    src/
        self_model/                    ← new in v1.4.0
            self_model_types.h
            self_model.h/.cpp          ← Layer 1
            meta_cognition.h/.cpp      ← Layer 2
        training/                      ← new in v1.4.0
            i_training_backend.h
            llama_cpp_trainer.h/.cpp
            tensorrt_trainer.h/.cpp
            training_factory.h/.cpp
            curriculum_builder.h/.cpp
            dataset_curator.h/.cpp
            adapter_evaluator.h/.cpp
            self_improvement_loop.h/.cpp
        vision/
        tools/builtin/
        ... (rest of source)
    vendor/
        llama.cpp/
        nlohmann_json/
        cpp-httplib/
        muparser/
    cardinal-train-venv/               ← Python venv for Layer 3
    config.json
    CMakeLists.txt
    README.md
    INSTALL.md
    DOCUMENTATION.md
```

---

## Next Steps

- Read `README.md` for the full architecture overview.
- Read `DOCUMENTATION.md` for the complete API, self-improvement internals, and configuration reference.
- Use `GET /api/self_model` to monitor Cardinal's self-knowledge as it accumulates inferences.
- Use `POST /api/reflect` to trigger an on-demand reflection pass after loading failure data.
- Use `POST /api/train` to manually kick off a LoRA training cycle.

---

*If something in this guide is wrong or out of date, the source of truth is always `CMakeLists.txt`, `config.json`, and the source code.*
