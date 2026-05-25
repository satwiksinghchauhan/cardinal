# Cardinal v1.5.0 — Installation Guide

**Linux only. Ubuntu 24.04 LTS recommended. No git submodules — all vendor dependencies cloned manually.**

v1.5.0 adds the Scheduler, Computer Use, and Watch subsystems. These require additional system packages (display tools, Python venv for Playwright, system control utilities) on top of the v1.4.0 base.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Step 1 — System Dependencies](#step-1--system-dependencies)
- [Step 2 — CUDA Toolkit](#step-2--cuda-toolkit)
- [Step 3 — Vendor Dependencies](#step-3--vendor-dependencies)
- [Step 4 — Build llama.cpp](#step-4--build-llamacpp)
- [Step 5 — Download Models](#step-5--download-models)
- [Step 6 — Build Cardinal](#step-6--build-cardinal)
- [Step 7 — Browser Venv (Playwright)](#step-7--browser-venv-playwright)
- [Step 8 — Training Venv (Layer 3 LoRA)](#step-8--training-venv-layer-3-lora-optional)
- [Step 9 — Email Setup](#step-9--email-setup)
- [Step 10 — Wayland Input (ydotool)](#step-10--wayland-input-ydotool)
- [Step 11 — config.json](#step-11--configjson)
- [Step 12 — Environment Variables](#step-12--environment-variables)
- [Step 13 — Data Directories](#step-13--data-directories)
- [First Run](#first-run)
- [Verify Computer Use](#verify-computer-use)
- [Verify Scheduler](#verify-scheduler)
- [Verify Self-Improvement (v1.4.0)](#verify-self-improvement-v140)
- [Verify Vision (v1.3.0)](#verify-vision-v130)
- [Verify HTTP API](#verify-http-api)
- [Troubleshooting](#troubleshooting)
- [Directory Reference](#directory-reference)

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 22.04 LTS | **Ubuntu 24.04 LTS** |
| CPU | Any x64, 4 cores | AMD Ryzen 7 or better |
| RAM | 8GB | 16GB (32GB for training) |
| GPU | NVIDIA with 4GB VRAM | RTX 3050 or better |
| Storage | 30GB free | 60GB free |
| CUDA Compute | 7.5+ | 8.6 (RTX 30 series) |
| Python | 3.10+ | 3.12 |
| Display | X11 or Wayland | X11 (broader tool support) |

Cardinal does **not** run on AMD or Intel GPUs. NVIDIA CUDA is required.
**Windows is not supported.**

---

## Quick Start

```bash
# 1. System dependencies
sudo apt update
sudo apt install -y build-essential cmake libsqlite3-dev libssl-dev swi-prolog \
    python3 python3-pip python3-venv \
    scrot imagemagick xdotool wmctrl xprop \
    ydotool wtype grim \
    pulseaudio-utils brightnessctl network-manager bluez

# 2. Vendor dependencies
cd ~/cardinal && mkdir -p vendor && cd vendor
git clone https://github.com/ggerganov/llama.cpp.git
git clone https://github.com/nlohmann/json.git nlohmann_json
git clone https://github.com/yhirose/cpp-httplib.git
git clone https://github.com/beltoforion/muparser.git
git clone https://github.com/mlc-ai/tokenizers-cpp

# 3. Build llama.cpp
cd llama.cpp && mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && cd ~/cardinal

# 4. Build Cardinal
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 5. Browser venv (required for browser tool)
python3 -m venv ~/cardinal/cardinal-browser-venv
source ~/cardinal/cardinal-browser-venv/bin/activate
pip install playwright && playwright install chromium && playwright install-deps chromium
deactivate

# 6. Run
cd ~/cardinal && ./build/bin/cardinal
```

---

## Step 1 — System Dependencies

### Base build tools

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
    python3-pip \
    python3-venv
```

### v1.5.0 — Computer Use tools

```bash
sudo apt install -y \
    # Screenshots
    scrot \
    imagemagick \
    # X11 input + window management
    xdotool \
    wmctrl \
    xprop \
    # Wayland input
    ydotool \
    wtype \
    # Wayland screenshots
    grim \
    # System control
    pulseaudio-utils \
    brightnessctl \
    network-manager \
    bluez
```

Verify:

```bash
cmake --version          # 3.28 or higher
swipl --version          # SWI-Prolog 9.2.x
python3 --version        # 3.10+
scrot --version          # any
xdotool version          # any
```

If CMake is older than 3.28:

```bash
sudo apt remove cmake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt update && sudo apt install cmake
```

---

## Step 2 — CUDA Toolkit

Install the NVIDIA driver first:

```bash
ubuntu-drivers autoinstall
sudo reboot
```

After reboot, install CUDA 12.6:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6
```

Add to PATH:

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

## Step 3 — Vendor Dependencies

Cardinal has no git submodules. Clone each dependency manually:

```bash
cd ~/cardinal
mkdir -p vendor && cd vendor

git clone https://github.com/ggerganov/llama.cpp.git
git clone https://github.com/nlohmann/json.git nlohmann_json
git clone https://github.com/yhirose/cpp-httplib.git
git clone https://github.com/beltoforion/muparser.git
git clone https://github.com/mlc-ai/tokenizers-cpp
```

| Dependency | Repository | Clone path |
|:-----------|:-----------|:-----------|
| llama.cpp | https://github.com/ggerganov/llama.cpp | `vendor/llama.cpp` |
| nlohmann/json | https://github.com/nlohmann/json | `vendor/nlohmann_json` |
| cpp-httplib | https://github.com/yhirose/cpp-httplib | `vendor/cpp-httplib` |
| muparser | https://github.com/beltoforion/muparser | `vendor/muparser` |
| muparser | https://github.com/mlc-ai/tokenizers-cpp | `vendor/tokenizers-cpp` |

---

## Step 4 — Build llama.cpp

```bash
cd ~/cardinal/vendor/llama.cpp
mkdir -p build && cd build
cmake .. \
    -DGGML_CUDA=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Verify `mtmd` was built (required for vision):

```bash
ls ~/cardinal/vendor/llama.cpp/tools/mtmd/
# Should contain mtmd.h, mtmd-helper.h
```

Add libraries to `LD_LIBRARY_PATH`:

```bash
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 5 — Download Models

### Primary LLM (required)

- **Qwen3.5 4B Q4_K_M**
- From: [bartowski/Qwen_Qwen3.5-4B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF)
- File: `Qwen_Qwen3.5-4B-Q4_K_M.gguf` (~2.5GB)
- Place at: `~/cardinal/models/Qwen_Qwen3.5-4B-Q4_K_M.gguf`

### Vision encoder (v1.3.0, optional but recommended)

```bash
mkdir -p ~/cardinal/models/vision
cd ~/cardinal/models/vision
wget https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-text-model-f16.gguf
wget https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-mmproj-f16.gguf
```

Quantized alternatives at: https://huggingface.co/salivosa/moondream2-gguf

### HuggingFace weights for training (Layer 3, optional)

Required only if you want LoRA fine-tuning to actually execute locally:

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-4B', local_dir='models/qwen3.5-4b-hf')
"
```

This downloads ~8GB. Cardinal works fully without it — Layers 1 and 2 have no Python or HF dependency.

---

## Step 6 — Build Cardinal

```bash
cd ~/cardinal
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

A successful build prints:

```
-- =============================================================
-- Cardinal v1.5.0 Configuration
-- =============================================================
-- Build type:         Release
-- CUDA arch:          75
-- TensorRT backend:   OFF
-- Features:           tools + agent + explainability + vision
--                     + self-improvement (layers 1-3)
--                     + scheduler + computer use + watch
-- Output dir:         .../build/bin
-- =============================================================
```

### Optional: TensorRT backend

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCARDINAL_ENABLE_TENSORRT=ON \
    -DTRT_LLM_INCLUDE_DIR=/path/to/TensorRT-LLM/include \
    -DTRT_LLM_LIB_DIR=/path/to/TensorRT-LLM/lib
make -j$(nproc)
```

---

## Step 7 — Browser Venv (Playwright)

Required for the `browser` tool. One-time setup:

```bash
python3 -m venv ~/cardinal/cardinal-browser-venv
source ~/cardinal/cardinal-browser-venv/bin/activate

pip install playwright
playwright install chromium
playwright install-deps chromium

deactivate
```

Verify:

```bash
~/cardinal/cardinal-browser-venv/bin/python -c \
    "from playwright.sync_api import sync_playwright; print('Playwright OK')"
```

The venv path must match `computer_use.browser.venv_path` in `config.json` (default: `~/cardinal/cardinal-browser-venv`).

---

## Step 8 — Training Venv (Layer 3 LoRA, optional)

Only needed for local LoRA fine-tuning. Skip if using TensorRT backend (script-export mode) or if you only want Layers 1 and 2.

```bash
python3 -m venv ~/cardinal/cardinal-train-venv
source ~/cardinal/cardinal-train-venv/bin/activate
pip install peft transformers torch accelerate
deactivate
```

Verify the paths in `config.json`:

```json
"training": {
    "python_venv": "~/cardinal/cardinal-train-venv",
    "hf_model_path": "models/qwen3.5-4b-hf",
    "convert_lora_script": "vendor/llama.cpp/convert_lora_to_gguf.py"
}
```

---

## Step 9 — Email Setup

### Option A — IMAP/SMTP (any email provider)

Set your password as an environment variable (never hardcoded in config):

```bash
export CARDINAL_EMAIL_PASS="your_app_password"
# Add to ~/.bashrc to persist across reboots:
echo 'export CARDINAL_EMAIL_PASS="your_app_password"' >> ~/.bashrc
```

Update `config.json`:

```json
"email": {
    "enabled": true,
    "mode": "imap_smtp",
    "imap_host": "imap.gmail.com",
    "imap_port": 993,
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "address": "you@gmail.com"
}
```

> **Gmail note:** Gmail requires an App Password, not your account password.
> Google Account → Security → 2-Step Verification → App Passwords → generate one for "Mail".
> Use that value as `CARDINAL_EMAIL_PASS`.

### Option B — Gmail REST API

Install dependencies in the browser venv:

```bash
source ~/cardinal/cardinal-browser-venv/bin/activate
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
deactivate
```

Set up OAuth credentials:
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project → Enable the **Gmail API**
3. Create OAuth 2.0 credentials → Desktop app
4. Download `credentials.json` → place at `data/gmail_credentials.json`

The first run will open a browser for OAuth consent. The token is saved automatically for future runs.

Update `config.json`:

```json
"email": {
    "enabled": true,
    "mode": "gmail_api",
    "gmail_api_enabled": true,
    "gmail_credentials_path": "data/gmail_credentials.json",
    "address": "you@gmail.com"
}
```

---

## Step 10 — Wayland Input (ydotool)

On Wayland, `ydotool` requires a running daemon with the right device permissions:

```bash
# Add your user to the input group
sudo usermod -aG input $USER
# Log out and back in for group change to take effect

# Start the daemon
sudo ydotoold &
```

Or run as a persistent systemd user service:

```bash
systemctl --user enable --now ydotool
```

Verify:

```bash
ydotool type "hello"
# Should type "hello" into whatever window is focused
```

On X11, `xdotool` needs no special setup — it works out of the box.

---

## Step 11 — config.json

Your existing `config.json` already has the complete v1.5.0 `scheduler` and `computer_use` blocks. Key fields to review:

```json
"scheduler": {
    "enabled": true,
    "db_path": "data/scheduler/scheduler.db",
    "check_interval_seconds": 30,
    "max_concurrent_tasks": 1,
    "idle_threshold_minutes": 5,
    "task_session_prefix": "scheduler_",
    "run_history_max_entries": 1000,
    "max_task_duration_seconds": 300
},
"computer_use": {
    "enabled": true,
    "safety": {
        "allowed_apps": ["google-chrome", "firefox", "nautilus", "gnome-terminal"],
        "allowed_paths": ["~/Documents", "~/Downloads", "~/Desktop", "data/"],
        "blocked_commands": ["rm -rf /", "rm -rf ~", "mkfs", "dd if=", ":(){:|:&};:"],
        "confirmation_required": true,
        "full_autonomy": false,
        "allow_file_write": false
    },
    "browser": {
        "venv_path": "~/cardinal/cardinal-browser-venv",
        "headless": false
    },
    "shell": {
        "enabled": true,
        "timeout_seconds": 30,
        "working_directory": "~"
    },
    "email": {
        "enabled": false
    }
}
```

To enable email, set `"enabled": true` and fill in the host/credentials (see Step 9).
To allow unsupervised shell commands, set `"confirmation_required": false` in the safety block (not recommended).

---

## Step 12 — Environment Variables

```bash
# Email password (required if email.enabled=true and mode=imap_smtp)
export CARDINAL_EMAIL_PASS="your_app_password"

# Persist across reboots
echo 'export CARDINAL_EMAIL_PASS="your_app_password"' >> ~/.bashrc

# llama.cpp libraries
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# CUDA
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

source ~/.bashrc
```

---

## Step 13 — Data Directories

CMake creates these automatically on first build. To create them manually:

```bash
mkdir -p ~/cardinal/data/{memory,self_model,scheduler,training/{adapters,datasets,scripts},explainability/exports,vision_cache,browser_profile,screenshots}
mkdir -p ~/cardinal/logs
```

---

## First Run

Always run from the Cardinal root directory so relative paths in `config.json` resolve:

```bash
cd ~/cardinal
./build/bin/cardinal
```

Expected startup output (v1.5.0):

```
[INFO ] SelfModel: opened database at data/self_model/self_model.db
[INFO ] SelfImprovementLoop: started
[INFO ] VisionEncoder: ready (CPU, 4 threads)
[INFO ] LlamaCppBackend ready - vocab: 151936, ctx: 8192
[INFO ] ToolRegistry: 21 tools registered
[INFO ] SchedulerEngine started
[INFO ] Computer use initialised (x11)
  +===========================================+
  |        C A R D I N A L  v1.5.0           |
  |   Neurosymbolic AGI Architecture          |
  +===========================================+
```

---

## Verify Computer Use

### Screenshots

```
You: take a screenshot
```

Cardinal should capture the screen, save it to `data/screenshots/`, and (if vision is configured) describe what it sees.

Via HTTP:

```bash
curl -s -X POST http://127.0.0.1:8080/api/computer/screenshot \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"analyze": true}' | python3 -m json.tool
```

### Status

```bash
curl -s http://127.0.0.1:8080/api/computer/status \
  -H "Authorization: Bearer secret_api_key"
# Returns: {"width":1920,"height":1080,"server":"x11","display_var":":0"}
```

Or from the CLI:

```
/computer
```

### Open an app

```
You: open the calculator
```

### Click something

```
You: click on the Submit button
You: click at coordinates 500, 300
```

### Browser

Requires the Playwright venv (Step 7):

```
You: open google.com in the browser
You: search for latest news
```

---

## Verify Scheduler

### From chat

```
You: remind me to check the weather every day at 8am
You: search for Python tutorials every Sunday at 10am and save to a file
You: list my scheduled tasks
```

### CLI

```
/scheduler
/tasks
```

### HTTP

```bash
# List tasks
curl -s http://127.0.0.1:8080/api/scheduler/tasks \
  -H "Authorization: Bearer secret_api_key" | python3 -m json.tool

# Create a task
curl -s -X POST http://127.0.0.1:8080/api/scheduler/tasks \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"description": "search for AI news every morning at 7am"}' | python3 -m json.tool

# Scheduler status
curl -s http://127.0.0.1:8080/api/scheduler/status \
  -H "Authorization: Bearer secret_api_key"
```

---

## Verify Self-Improvement (v1.4.0)

### Layer 1 — Self-Model

```bash
curl -s http://127.0.0.1:8080/api/self_model \
  -H "Authorization: Bearer secret_api_key" | python3 -m json.tool
```

### Layer 2 — Meta-Cognition

```bash
curl -s -X POST http://127.0.0.1:8080/api/reflect \
  -H "Authorization: Bearer secret_api_key" | python3 -m json.tool
```

### Layer 3 — Training

```bash
curl -s -X POST http://127.0.0.1:8080/api/train \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"domain_hint": ""}' | python3 -m json.tool
```

---

## Verify Vision (v1.3.0)

Place a test image in `data/` and ask:

```
You: describe data/test.jpg
```

---

## Verify HTTP API

```bash
# Health — no auth
curl http://127.0.0.1:8080/api/health

# Chat
curl -X POST http://127.0.0.1:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer secret_api_key" \
  -d '{"session_id":"test","message":"Hello!"}'
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `scrot: command not found` | `sudo apt install scrot` |
| `xdotool: command not found` | `sudo apt install xdotool` |
| `wmctrl: command not found` | `sudo apt install wmctrl` |
| `grim: command not found` (Wayland) | `sudo apt install grim` |
| `ydotool: /dev/uinput permission denied` | `sudo usermod -aG input $USER` then re-login |
| `ydotool: daemon not running` | `sudo ydotoold &` or `systemctl --user enable --now ydotool` |
| `browser: Playwright not found` | Run Step 7 (browser venv setup) |
| `email: authentication failed` | Use Gmail App Password, not account password |
| `email: CARDINAL_EMAIL_PASS not set` | `export CARDINAL_EMAIL_PASS="..."` |
| `open_app: failed to launch` | Check app is installed; try full executable name |
| Scheduler tasks not firing | Verify `check_interval_seconds` in config; check task is enabled |
| Display not detected | Set `DISPLAY=:0` (X11) or `WAYLAND_DISPLAY=wayland-0` |
| `mtmd.h not found` during CMake | Rebuild llama.cpp with `-DLLAMA_BUILD_EXAMPLES=ON` |
| `libmtmd.so not found` at runtime | Add llama.cpp/build/lib to `LD_LIBRARY_PATH` |
| `CUDA out of memory` | Reduce `gpu_layers` in `config.json` |
| Self-model DB not created | Ensure `self_improvement.enabled=true`; DB created on first chat |
| Reflection returns `ran=false` | Lower `min_failures_to_reflect` to 1 for testing |
| Training returns `accepted=false` | Check `training.enabled=true`; check cycle isn't already running |

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
"api": { "auth_enabled": false }
```

---

## Directory Reference (v1.5.0)

```
~/cardinal/
    build/
        bin/
            cardinal
    data/
        memory/
            rules.json, knowledge.json, episodes.db
        self_model/
            self_model.db              ← Layer 1 SQLite
        training/
            adapters/                  ← trained GGUF adapters
            datasets/                  ← curated JSONL datasets
            scripts/                   ← TensorRT training scripts
        scheduler/
            scheduler.db               ← tasks, runs, action_logs  ← NEW v1.5.0
        explainability/
            audit.db, exports/
        vision_cache/
        browser_profile/               ← Playwright browser profile ← NEW v1.5.0
        screenshots/                   ← computer use screenshots   ← NEW v1.5.0
    logs/
        cardinal.log, episodic.log
    models/
        Qwen_Qwen3.5-4B-Q4_K_M.gguf
        qwen3.5-4b-hf/                 ← HF weights for Layer 3
        vision/
            moondream2-text-model-f16.gguf
            moondream2-mmproj-f16.gguf
    src/
        api/
        agent/
        computer/                      ← NEW v1.5.0
        scheduler/                     ← NEW v1.5.0
        watch/                         ← NEW v1.5.0
        tools/builtin/computer/        ← NEW v1.5.0
        self_model/, training/
        core/, memory/, verifier/, vision/
        tools/, explainability/, learning/
    vendor/
        llama.cpp/
        nlohmann_json/
        cpp-httplib/
        muparser/
        tokenizers-cpp/
    cardinal-browser-venv/             ← Playwright venv   ← NEW v1.5.0
    cardinal-train-venv/               ← PEFT training venv
    config.json
    CMakeLists.txt
    README.md, INSTALL.md, DOCUMENTATION.md
```

---

*If something in this guide is wrong or out of date, the source of truth is always `CMakeLists.txt`, `config.json`, and the source code.*
