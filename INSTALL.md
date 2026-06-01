# Cardinal v2.0.0 — Installation Guide

**Linux only. Ubuntu 24.04 LTS recommended. No git submodules — all vendor dependencies cloned manually.**

This guide consolidates all installation steps from v1.0.0 through v1.6.0. Windows support was removed in v1.1.0 and is not coming back. If you are migrating from an older install, every section is marked with the version that introduced it.

---

## Table of Contents

- [Version History Summary](#version-history-summary)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Step 1 — System Dependencies](#step-1--system-dependencies)
- [Step 2 — CUDA Toolkit 12.6](#step-2--cuda-toolkit-126)
- [Step 3 — Vendor Dependencies](#step-3--vendor-dependencies)
- [Step 4 — Build llama.cpp](#step-4--build-llamacpp)
- [Step 5 — Build Voice Vendor Libs](#step-5--build-voice-vendor-libs)
- [Step 6 — Download Models](#step-6--download-models)
- [Step 7 — Build Cardinal](#step-7--build-cardinal)
- [Step 8 — Browser Venv (Playwright)](#step-8--browser-venv-playwright)
- [Step 9 — Training Venv (Layer 3 LoRA, optional)](#step-9--training-venv-layer-3-lora-optional)
- [Step 10 — Email Setup](#step-10--email-setup)
- [Step 11 — Wayland Input (ydotool)](#step-11--wayland-input-ydotool)
- [Step 12 — config.json](#step-12--configjson)
- [Step 13 — Environment Variables](#step-13--environment-variables)
- [Step 14 — Data Directories](#step-14--data-directories)
- [First Run](#first-run)
- [Verify Voice (v1.6.0)](#verify-voice-v160)
- [Verify Computer Use (v1.5.0)](#verify-computer-use-v150)
- [Verify Scheduler (v1.5.0)](#verify-scheduler-v150)
- [Verify Self-Improvement (v1.4.0)](#verify-self-improvement-v140)
- [Verify Vision (v1.3.0)](#verify-vision-v130)
- [Verify the HTTP API](#verify-the-http-api)
- [Troubleshooting](#troubleshooting)
- [Changing the API Key](#changing-the-api-key)
- [Directory Reference](#directory-reference)
- [Next Steps](#next-steps)

---

## Version History Summary

| Version | What was added |
|:--------|:---------------|
| v1.0.0 | Initial release. Windows + Linux. llama.cpp + SWI-Prolog + SQLite. HTTP API. |
| v1.1.0 | Windows support removed. Ubuntu 24.04 LTS recommended. Clean vendor folder. TensorRT backend (optional). |
| v1.2.0 | Agentic loop (`max_iterations`). Python sandbox (`run_python` tool). Docker sandbox mode. Explainability system. |
| v1.3.0 | Native vision encoding via moondream2 / llama.cpp `mtmd`. `analyze_image` tool. Vision cache. |
| v1.4.0 | SEAL self-improvement system: Layer 1 (self-model), Layer 2 (meta-cognition), Layer 3 (LoRA fine-tuning). `muparser` vendor dep added. Training Python venv. |
| v1.5.0 | Scheduler, Computer Use, and Watch subsystems. Browser tool (Playwright). Email tool. Wayland input (ydotool). |
| v1.6.0 | Voice subsystem: Whisper STT, Piper TTS, PortAudio, PocketSphinx wake word. Four new vendor deps. ALSA dev headers required. |

---

## System Requirements

| Component | Minimum | Recommended |
|:----------|:--------|:------------|
| OS | Ubuntu 22.04 LTS | **Ubuntu 24.04 LTS** |
| CPU | Any x64, 4 cores | AMD Ryzen 7 or better |
| RAM | 8GB | 16GB (32GB for training) |
| GPU | NVIDIA with 4GB VRAM | RTX 3050 or better |
| Storage | 35GB free | 70GB free |
| CUDA Compute | 7.5+ | 8.6 (RTX 30 series) |
| Python | 3.10+ | 3.12 |
| Display | X11 or Wayland | X11 (broader tool support) |
| Audio | ALSA-compatible sound card | Any — PortAudio auto-detects |
| Microphone | Any ALSA/PulseAudio mic | USB or 3.5mm |

Cardinal does **not** run on AMD or Intel GPUs. NVIDIA CUDA is required.
**Windows is not supported.**

---

## Quick Start

For the impatient. Each step is detailed in the sections below.

```bash
# 1. System dependencies
sudo apt update
sudo apt install -y build-essential cmake libsqlite3-dev libssl-dev swi-prolog \
    python3 python3-pip python3-venv \
    libasound2-dev \
    scrot imagemagick xdotool wmctrl xprop x11-utils \
    ydotool wtype grim \
    pulseaudio-utils brightnessctl network-manager bluez

# 2. Clone Cardinal
git clone https://github.com/satwiksinghchauhan/cardinal ~/cardinal
cd ~/cardinal

# 3. Vendor dependencies
mkdir -p vendor && cd vendor
git clone https://github.com/ggerganov/llama.cpp.git
git clone https://github.com/nlohmann/json.git nlohmann_json
git clone https://github.com/yhirose/cpp-httplib.git
git clone https://github.com/beltoforion/muparser.git
git clone https://github.com/mlc-ai/tokenizers-cpp
git clone https://github.com/ggerganov/whisper.cpp.git
git clone https://github.com/rhasspy/piper.git
git clone https://github.com/PortAudio/portaudio.git
git clone https://github.com/cmusphinx/pocketsphinx.git
cd ~/cardinal

# ONNX Runtime for Piper (pre-built binary)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-gpu-1.17.3.tgz
tar xzf onnxruntime-linux-x64-gpu-1.17.3.tgz
mv onnxruntime-linux-x64-gpu-1.17.3 onnxruntime
rm onnxruntime-linux-x64-gpu-1.17.3.tgz

# 4. Build llama.cpp
cd vendor/llama.cpp && mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && cd ~/cardinal

# 5. Pre-build piper
cd vendor/piper
cmake -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install -j$(nproc)
cd ~/cardinal

# 6. Pre-build pocketsphinx
cd vendor/pocketsphinx
cmake -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install -j$(nproc)
cd ~/cardinal

# 7. Build Cardinal
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
cd ~/cardinal

# 8. Browser venv
python3 -m venv ~/cardinal/cardinal-browser-venv
source ~/cardinal/cardinal-browser-venv/bin/activate
pip install playwright && playwright install chromium && playwright install-deps chromium
deactivate

# 9. Run (text mode)
cd ~/cardinal && ./build/bin/cardinal

# 9b. Run with voice
./build/bin/cardinal --voice
```

---

## Step 1 — System Dependencies

### Base build tools (since v1.0.0)

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

Verify:

```bash
cmake --version    # must be 3.28 or higher
swipl --version    # should print SWI-Prolog 9.2.9 or newer
openssl version    # should print 3.0.x or newer
python3 --version  # 3.10 or newer
```

If your distro ships CMake older than 3.28, install it from Kitware:

```bash
sudo apt remove cmake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt update && sudo apt install cmake
```

---

### Computer Use tools (added v1.5.0)

```bash
sudo apt install -y \
    scrot \
    imagemagick \
    xdotool \
    wmctrl \
    x11-utils \
    ydotool \
    wtype \
    grim \
    pulseaudio-utils \
    brightnessctl \
    network-manager \
    bluez
```

---

### Voice dependencies (added v1.6.0)

```bash
sudo apt install -y \
    libasound2-dev
```

That is the only new system package for voice. `onnxruntime`, `espeak-ng`, `fmt`, and `spdlog` are downloaded and built automatically by piper's CMake during Step 5.

---

## Step 2 — CUDA Toolkit 12.6

Install the NVIDIA driver first if not already installed:

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

Add CUDA to your PATH:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
nvcc --version    # should print: release 12.6
nvidia-smi        # should show your GPU
```

---

## Step 3 — Vendor Dependencies

Cardinal has no git submodules. Clone all dependencies manually into `vendor/`:

```bash
cd ~/cardinal
mkdir -p vendor && cd vendor

# Core deps (since v1.1.0)
git clone https://github.com/ggerganov/llama.cpp.git
git clone https://github.com/nlohmann/json.git nlohmann_json
git clone https://github.com/yhirose/cpp-httplib.git
git clone https://github.com/mlc-ai/tokenizers-cpp

# muparser (added v1.4.0)
git clone https://github.com/beltoforion/muparser.git

# Voice deps (added v1.6.0)
git clone https://github.com/ggerganov/whisper.cpp.git
git clone https://github.com/rhasspy/piper.git
git clone https://github.com/PortAudio/portaudio.git
git clone https://github.com/cmusphinx/pocketsphinx.git
```

| Dependency | Repository | Clone path |
|:-----------|:-----------|:-----------|
| llama.cpp | https://github.com/ggerganov/llama.cpp | `vendor/llama.cpp` |
| nlohmann/json | https://github.com/nlohmann/json | `vendor/nlohmann_json` |
| cpp-httplib | https://github.com/yhirose/cpp-httplib | `vendor/cpp-httplib` |
| tokenizers-cpp | https://github.com/mlc-ai/tokenizers-cpp | `vendor/tokenizers-cpp` |
| muparser *(v1.4.0)* | https://github.com/beltoforion/muparser | `vendor/muparser` |
| whisper.cpp *(v1.6.0)* | https://github.com/ggerganov/whisper.cpp | `vendor/whisper.cpp` |
| piper *(v1.6.0)* | https://github.com/rhasspy/piper | `vendor/piper` |
| portaudio *(v1.6.0)* | https://github.com/PortAudio/portaudio | `vendor/portaudio` |
| pocketsphinx *(v1.6.0)* | https://github.com/cmusphinx/pocketsphinx | `vendor/pocketsphinx` |

Alternatively, use the provided script if it exists in your repo:

```bash
cd ~/cardinal
chmod +x scripts/populate_vendor.sh
./scripts/populate_vendor.sh
```

---

## Step 4 — Build llama.cpp

Build with CUDA and the `mtmd` multimodal subsystem (required for vision, added v1.3.0):

```bash
cd ~/cardinal/vendor/llama.cpp
mkdir -p build && cd build
cmake .. \
    -DGGML_CUDA=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This will take 10–20 minutes.

Verify `mtmd` was built (required for vision):

```bash
ls ~/cardinal/vendor/llama.cpp/tools/mtmd/
# Should contain mtmd.h, mtmd-helper.h
```

Add llama.cpp libraries to `LD_LIBRARY_PATH`:

```bash
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 5 — Build Voice Vendor Libs

*(Added v1.6.0. Skip this step if you do not need voice.)*

`whisper.cpp` is built automatically by Cardinal's CMake via `add_subdirectory`. Piper and PocketSphinx must be **pre-built** into local install prefixes before Cardinal's CMake runs, because their CMake systems use `ExternalProject_Add` and are not designed for `add_subdirectory`. This is a one-time step.

### 5a — Build piper

Piper's build automatically downloads its own bundled versions of `onnxruntime`, `espeak-ng`, `fmt`, and `spdlog`. This requires an internet connection and takes 5–10 minutes.

```bash
cd ~/cardinal/vendor/piper
cmake -B build \
      -DCMAKE_INSTALL_PREFIX=install \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install -j$(nproc)
```

After this, `vendor/piper/install/` will contain:

```
vendor/piper/install/
    piper                           ← the piper binary (not used directly by Cardinal)
    libpiper_phonemize.so.1.2.0
    libonnxruntime.so.1.14.1
    libespeak-ng.so.1.52.0.1
    espeak-ng-data/                 ← language data for espeak-ng
    include/                        ← piper-phonemize headers
    pkgconfig/
```

Add piper's libs to `LD_LIBRARY_PATH`:

```bash
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/piper/install:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 5b — Build pocketsphinx

```bash
cd ~/cardinal/vendor/pocketsphinx
cmake -B build \
      -DCMAKE_INSTALL_PREFIX=install \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install -j$(nproc)
```

After this, `vendor/pocketsphinx/install/` will contain:

```
vendor/pocketsphinx/install/
    lib/
        libpocketsphinx.so
    include/
        pocketsphinx/
            pocketsphinx.h
```

Add pocketsphinx's lib dir to `LD_LIBRARY_PATH`:

```bash
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/pocketsphinx/install/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 5c ONNX Runtime for Piper (pre-built binary)

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-gpu-1.17.3.tgz
tar xzf onnxruntime-linux-x64-gpu-1.17.3.tgz
mv onnxruntime-linux-x64-gpu-1.17.3 onnxruntime
rm onnxruntime-linux-x64-gpu-1.17.3.tgz
```

---

## Step 6 — Download Models

### Primary LLM (required, since v1.0.0)

- **Qwen3.5 4B Q4_K_M**
- From: [bartowski/Qwen_Qwen3.5-4B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF)
- File: `Qwen_Qwen3.5-4B-Q4_K_M.gguf` (~2.5GB)
- Place at: `~/cardinal/models/Qwen_Qwen3.5-4B-Q4_K_M.gguf`

### Neural verifier model (optional, since v1.0.0)

- **Llama 3.2 1B Instruct Q4_K_M** (~700MB)
- From: [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)
- Place at: `~/cardinal/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`

If you skip the neural verifier, set `verifier.mode` to `"symbolic"` in `config.json` and leave `neural_model_path` empty.

### Vision encoder (optional but recommended, added v1.3.0)

```bash
mkdir -p ~/cardinal/models/vision
cd ~/cardinal/models/vision

# Text model (~1.1GB)
wget https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-text-model-f16.gguf

# Vision projector (~400MB)
wget https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-mmproj-f16.gguf
```

If those links do not work, try: https://huggingface.co/salivosa/moondream2-gguf  
If using the alternative source, update the model paths in `config.json` accordingly.

If you skip the vision models, set `vision.enabled = false` in `config.json`.

### Voice models (added v1.6.0)

```bash
mkdir -p ~/cardinal/models/voice
cd ~/cardinal/models/voice
```

#### Whisper STT model

```bash
# Medium English model — best accuracy/speed balance on RTX 3050
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin
```

Smaller alternatives (less accurate, faster):

```bash
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
```

Update `config.json` `voice.stt.model_path` if you use a model other than `ggml-medium.en.bin`.

#### Piper TTS voice

```bash
# Default voice: en_US-lessac-medium (warm American English, ~65MB)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Other voices are available at: https://huggingface.co/rhasspy/piper-voices  
Any voice `.onnx` + `.onnx.json` pair can be used — set `voice.tts.model_path` and `voice.tts.config_path` accordingly.

#### PocketSphinx acoustic model and dictionary (wake-word mode only)

Required only if using `wake_word` input mode:

```bash
mkdir -p ~/cardinal/models/voice/pocketsphinx

# Download en-US acoustic model
wget -O en-us.tar.gz \
  https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/US%20English/cmusphinx-en-us-5.2.tar.gz/download
tar xf en-us.tar.gz
mv cmusphinx-en-us-5.2 ~/cardinal/models/voice/pocketsphinx/en-us

# Download CMUdict
wget -O ~/cardinal/models/voice/pocketsphinx/cmudict-en-us.dict \
  https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
```

### HuggingFace weights for LoRA training (Layer 3, optional, added v1.4.0)

Required only if you want Layer 3 fine-tuning to execute locally:

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-4B', local_dir='models/qwen3.5-4b-hf')
"
```

This downloads ~8GB. Cardinal works fully without it — Layers 1 and 2 have no Python or HF dependency.

---

## Step 7 — Build Cardinal

```bash
cd ~/cardinal
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

A successful build prints:

```
-- =============================================================
-- Cardinal v2.0.0 Configuration
-- =============================================================
-- Build type:         Release
-- CUDA arch:          75
-- TensorRT backend:   OFF
-- Voice subsystem:    ON
-- Piper install:      /home/user/cardinal/vendor/piper/install
-- PocketSphinx:       /home/user/cardinal/vendor/pocketsphinx/install
-- Features:           tools + agent + explainability + vision
--                     + self-improvement (layers 1-3)
--                     + scheduler + computer use + watch + voice
-- Output dir:         .../build/bin
-- =============================================================
```

### Optional: TensorRT backend

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCARDINAL_ENABLE_TENSORRT=ON \
    -DTRT_LLM_INCLUDE_DIR=/path/to/TensorRT-LLM/include \
    -DTRT_LLM_LIB_DIR=/path/to/TensorRT-LLM/lib
cmake --build . -j$(nproc)
```

With TensorRT, Layer 3 runs in script-export mode: instead of launching a training subprocess, Cardinal writes a ready-to-run shell script to `data/training/scripts/` for your cluster to execute. Omit the `-DCARDINAL_ENABLE_TENSORRT=ON` flag to switch back to llama.cpp.

### Optional: Disable voice subsystem

On machines without a microphone or audio hardware:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCARDINAL_ENABLE_VOICE=OFF
```

### If CMake cannot find SWI-Prolog headers

```bash
swipl --dump-runtime-variables | grep PLBASE
# Example output: PLBASE='/usr/lib/swi-prolog'
```

Then pass the include path explicitly:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DSWIPL_INCLUDE_DIR=/usr/lib/swi-prolog/include
```

---

## Step 8 — Browser Venv (Playwright)

*(Added v1.5.0. Required for the `browser` computer use tool.)*

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

Set `computer_use.browser.venv_path` in `config.json` to `~/cardinal/cardinal-browser-venv`.

For Gmail REST API support, also install inside this venv:

```bash
source ~/cardinal/cardinal-browser-venv/bin/activate
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
deactivate
```

---

## Step 9 — Training Venv (Layer 3 LoRA, optional)

*(Added v1.4.0. Only needed for local LoRA fine-tuning. Skip if using TensorRT backend script-export mode or if you only want Layers 1 and 2.)*

```bash
python3 -m venv ~/cardinal/cardinal-train-venv
source ~/cardinal/cardinal-train-venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft datasets accelerate bitsandbytes

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

## Step 10 — Email Setup

*(Added v1.5.0. Required only if `computer_use.email.enabled = true`.)*

### Option A — IMAP/SMTP (any provider)

Set your password as an environment variable (never hardcode in config):

```bash
export CARDINAL_EMAIL_PASS="your_app_password"
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
> Use that value as `CARDINAL_EMAIL_PASS`. Also enable IMAP in Gmail settings.

### Option B — Gmail REST API

Install dependencies (see Step 8 for the browser venv), then set up OAuth credentials:

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

## Step 11 — Wayland Input (ydotool)

*(Added v1.5.0. Required only if running Wayland. On X11, `xdotool` works out of the box with no special setup.)*

`ydotool` requires the `uinput` kernel module and a running daemon:

```bash
sudo modprobe uinput
sudo usermod -aG input $USER
# Log out and back in for the group change to take effect

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

---

## Step 12 — config.json

The config file lives at `~/cardinal/config.json`. The full reference is in `DOCUMENTATION.md`. Key blocks to review:

### Scheduler block (added v1.5.0)

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
}
```

### Computer Use block (added v1.5.0)

```json
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

To allow unsupervised shell commands, set `"confirmation_required": false` in the safety block (not recommended).

### Voice block (added v1.6.0)

```json
"voice": {
    "enabled": false,
    "input_mode": "vad",
    "tts_streaming": "sentence",
    "session_id": "voice_session",
    "stt": {
        "model_path": "models/voice/ggml-medium.en.bin",
        "language": "en",
        "gpu_layers": 8,
        "threads": 4,
        "beam_size": 5,
        "initial_prompt": ""
    },
    "tts": {
        "model_path": "models/voice/en_US-lessac-medium.onnx",
        "config_path": "models/voice/en_US-lessac-medium.onnx.json",
        "speaker_id": 0,
        "length_scale": 1.0,
        "noise_scale": 0.667,
        "noise_w": 0.8,
        "sample_rate": 22050
    },
    "vad": {
        "energy_threshold": 0.02,
        "pre_speech_ms": 300,
        "post_speech_ms": 800,
        "min_speech_ms": 200,
        "max_speech_ms": 30000
    },
    "push_to_talk": {
        "key": "space"
    },
    "wake_word": {
        "phrase": "hey cardinal",
        "acoustic_model": "models/voice/pocketsphinx/en-us",
        "dictionary": "models/voice/pocketsphinx/cmudict-en-us.dict",
        "sensitivity": 1e-20
    },
    "audio": {
        "input_device": -1,
        "output_device": -1,
        "sample_rate": 16000,
        "channels": 1,
        "frames_per_buffer": 512
    }
}
```

---

## Step 13 — Environment Variables

Add all of these to `~/.bashrc` so they persist across reboots:

```bash
# Email password (required if email.enabled=true and mode=imap_smtp)
export CARDINAL_EMAIL_PASS="your_app_password"
echo 'export CARDINAL_EMAIL_PASS="your_app_password"' >> ~/.bashrc

# CUDA
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# llama.cpp libraries
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Piper runtime libs (flat install dir — no lib/ subdirectory)
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/piper/install:$LD_LIBRARY_PATH' >> ~/.bashrc

# PocketSphinx
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/pocketsphinx/install/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

source ~/.bashrc
```

---

## Step 14 — Data Directories

CMake creates these automatically on first build. To create them manually:

```bash
mkdir -p ~/cardinal/data/{memory,self_model,scheduler,training/{adapters,datasets,scripts},explainability/exports,vision_cache,browser_profile,screenshots}
mkdir -p ~/cardinal/models/voice/pocketsphinx
mkdir -p ~/cardinal/logs
```

---

## First Run

Always run from the Cardinal root directory so relative paths in `config.json` resolve correctly:

```bash
cd ~/cardinal

# Text mode
./build/bin/cardinal

# Voice mode — VAD (voice activity detection, from config)
./build/bin/cardinal --voice

# Voice mode — push-to-talk (hold spacebar to speak)
./build/bin/cardinal --voice=ptt

# Voice mode — wake word ("hey cardinal")
./build/bin/cardinal --voice=wake
```

Expected startup output (all features enabled):

```
[INFO ] SelfModel: opened database at data/self_model/self_model.db
[INFO ] MetaCognition: initialised (enabled=true, trigger_every=20)
[INFO ] LlamaCppTrainer: initialised (venv=~/cardinal/cardinal-train-venv)
[INFO ] SelfImprovementLoop: started
[INFO ] VisionEncoder: ready (CPU, 4 threads)
[INFO ] LlamaCppBackend ready - vocab: 151936, ctx: 8192
[INFO ] ToolRegistry: 22 tools registered
[INFO ] SchedulerEngine started
[INFO ] Computer use initialised (x11)
[INFO ] AudioDevice init: input=0 output=1 out_rate=44100
[INFO ] STTEngine ready — model: models/voice/ggml-medium.en.bin lang: en
[INFO ] TTSEngine ready — model: models/voice/en_US-lessac-medium.onnx
[INFO ] VoiceLoop started — mode=vad
  +===========================================+
  |        C A R D I N A L  v2.0.0           |
  |   Neurosymbolic AGI Architecture          |
  +===========================================+
  Voice mode active
```

If vision is configured, you will also see:

```
[INFO ] VisionCache: initialized at data/vision_cache (TTL=24h)
[INFO ] VisionEncoder: loading text model: models/vision/moondream2-text-model-f16.gguf
[INFO ] VisionEncoder: loading mmproj:     models/vision/moondream2-mmproj-f16.gguf
[INFO ] Vision encoder ready (moondream2, CPU)
```

---

## Verify Voice (v1.6.0)

### Voice status from CLI

```
/voice status
```

Expected output:

```
  -- Voice Subsystem --
  Active:      yes
  Mode:        vad
  State:       listening
  STT ready:   yes
  TTS ready:   yes
  Wake ready:  no
  Transcripts: 0
  Utterances:  0
```

### TTS test

```
/voice speak Hello, I am Cardinal.
```

You should hear the voice through your speakers immediately.

### Microphone test (VAD mode)

Speak a question. Cardinal should:
1. Detect speech onset (VAD)
2. Record until you stop talking
3. Transcribe with Whisper
4. Print the transcript
5. Run inference
6. Speak the response via Piper

### Via HTTP

```bash
# Status
curl -s http://127.0.0.1:8080/api/voice/status \
  -H "Authorization: Bearer secret_api_key" | python3 -m json.tool

# Enable voice via API
curl -s -X POST http://127.0.0.1:8080/api/voice/enable \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"input_mode": "vad"}' | python3 -m json.tool

# TTS via API
curl -s -X POST http://127.0.0.1:8080/api/voice/speak \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Cardinal."}' | python3 -m json.tool

# Transcribe raw PCM (16-bit signed, 16kHz mono)
curl -s -X POST http://127.0.0.1:8080/api/voice/transcribe \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/octet-stream" \
  -H "X-Sample-Rate: 16000" \
  --data-binary @audio.raw | python3 -m json.tool
```

### Push-to-talk test

```bash
./build/bin/cardinal --voice=ptt
```

Hold spacebar to speak, release to transcribe.

### Wake word test

```bash
./build/bin/cardinal --voice=wake
```

Say "hey cardinal" — Cardinal should transition from `passive_listening` to `listening` and then record your follow-up.

---

## Verify Computer Use (v1.5.0)

### Status

```bash
curl -s http://127.0.0.1:8080/api/computer/status \
  -H "Authorization: Bearer secret_api_key"
# Returns: {"width":1920,"height":1080,"server":"x11","display_var":":0"}
```

Or from the CLI: `/computer`

### Screenshot

```bash
curl -s -X POST http://127.0.0.1:8080/api/computer/screenshot \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"analyze": true}' | python3 -m json.tool
```

Or from the CLI:

```
You: take a screenshot
```

Cardinal will capture the screen, save it to `data/screenshots/`, and (if vision is configured) describe what it sees.

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

Requires the Playwright venv (Step 8):

```
You: open google.com in the browser
You: search for latest news
```

---

## Verify Scheduler (v1.5.0)

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

### Via HTTP

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

Send a few messages, then query the self-model endpoint:

```bash
curl -s http://127.0.0.1:8080/api/self_model \
  -H "Authorization: Bearer secret_api_key" | python3 -m json.tool
```

Expected response includes `weakest_domain`, `strongest_domain`, `total_domain_stats`.

Inspect the SQLite database directly:

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

Place a test image (e.g., `test.jpg`) in `data/` and ask:

```
You: describe the image at data/test.jpg
```

Cardinal will call the `analyze_image` tool and return a description.

Via HTTP:

```bash
curl -X POST http://127.0.0.1:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer secret_api_key" \
  -d '{"session_id":"test","message":"describe the image at data/test.jpg"}'
```

If vision models are missing at startup, you will see:

```
[WARN ] VisionEncoder: built without mtmd support — vision disabled
[WARN ]   Add vendor/llama.cpp/tools/mtmd to include path in CMakeLists
```

In that case, check that `llama.cpp` was built with `-DLLAMA_BUILD_EXAMPLES=ON` and the model files are present at the paths in `config.json`.

---

## Verify the HTTP API

```bash
# Health check — no auth required
curl http://127.0.0.1:8080/api/health

# Chat
curl -X POST http://127.0.0.1:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer secret_api_key" \
  -d '{"session_id":"test","message":"What is entropy?"}'

# Agentic request (with tools)
curl -X POST http://127.0.0.1:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer secret_api_key" \
  -d '{
    "session_id": "agent-session",
    "message": "Search the web for DRDO latest news and save to ~/Downloads/drdo_news.txt",
    "max_iterations": 5
  }'

# Self-model status
curl http://127.0.0.1:8080/api/self_model \
  -H "Authorization: Bearer secret_api_key"

# Stats
curl http://127.0.0.1:8080/api/stats \
  -H "Authorization: Bearer secret_api_key"
```

---

## Troubleshooting

### Voice

| Problem | Fix |
|:--------|:----|
| `libasound2-dev not found` | `sudo apt install libasound2-dev` |
| `AudioDevice init failed` | Check microphone is connected; try `aplay -l` and `arecord -l` |
| `STTEngine: failed to load model` | Verify `models/voice/ggml-medium.en.bin` exists |
| `TTSEngine: failed to load voice` | Verify both `.onnx` and `.onnx.json` files exist |
| No audio output | Check output device with `aplay -l`; set `audio.output_device` to correct index |
| Voice not transcribing | Lower `vad.energy_threshold` (try `0.01`); check microphone input level |
| STT slow | Reduce `stt.beam_size` to 1 or 2; or use `ggml-base.en.bin` instead of medium |
| `libpiper_phonemize.so not found` | Add `vendor/piper/install` to `LD_LIBRARY_PATH` |
| `libpocketsphinx.so not found` | Add `vendor/pocketsphinx/install/lib` to `LD_LIBRARY_PATH` |
| Wake word never triggers | Check acoustic model path; lower `sensitivity` (e.g. `1e-30`) |
| Push-to-talk not responding | Terminal must have focus; space key captures stdin in raw mode |
| Piper build fails — onnxruntime download error | Check internet access during `cmake --build build --target install` |
| Pocketsphinx build: `BISON/FLEX not found` | `sudo apt install bison flex` then rebuild |

### Vision

| Problem | Fix |
|:--------|:----|
| `mtmd.h not found` during CMake | Rebuild llama.cpp with `-DLLAMA_BUILD_EXAMPLES=ON` |
| `libmtmd.so not found` at runtime | `export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/bin:$LD_LIBRARY_PATH` — make permanent in `~/.bashrc` |
| Vision encoder says "built without mtmd support" | Check `CARDINAL_MTMD_AVAILABLE` in build summary; re-run CMake and ensure detection prints `mtmd: found at ...`; if not, manually set `-DCARDINAL_MTMD_AVAILABLE=ON` |
| Vision models not loading | Verify paths in `config.json` point to existing files; re-download if corrupted (text model ~1.1GB, mmproj ~400MB) |

### Computer Use

| Problem | Fix |
|:--------|:----|
| `scrot: command not found` | `sudo apt install scrot` |
| `xdotool: command not found` | `sudo apt install xdotool` |
| `wmctrl: command not found` | `sudo apt install wmctrl` |
| `grim: command not found` (Wayland) | `sudo apt install grim` |
| `ydotool: /dev/uinput permission denied` | `sudo usermod -aG input $USER` then re-login |
| `ydotool: daemon not running` | `sudo ydotoold &` or `systemctl --user enable --now ydotool` |
| `browser: Playwright not found` | Run Step 8 (browser venv setup) |
| `email: authentication failed` | Use Gmail App Password, not account password |
| `email: CARDINAL_EMAIL_PASS not set` | `export CARDINAL_EMAIL_PASS="..."` |
| `open_app: failed to launch` | Check app is installed; try full executable name |
| Scheduler tasks not firing | Verify `check_interval_seconds` in config; check task is enabled |
| Display not detected | Set `DISPLAY=:0` (X11) or `WAYLAND_DISPLAY=wayland-0` |

### Self-Improvement

| Problem | Fix |
|:--------|:----|
| Self-model DB not created | Ensure `self_improvement.enabled=true` and `self_improvement.self_model.enabled=true`; DB is created on first chat, not on startup |
| Reflection returns `ran=false` | Lower `min_failures_to_reflect` to 1 for testing; failure episodes are those where `contradiction=true` or `uncertainty=true` |
| Training returns `accepted=false` | Either training is disabled in config or a cycle is already running; check log for `training_in_progress` |
| `cardinal_train` module not found | Activate the training venv and re-run `pip install peft transformers torch accelerate` |
| Training fails: `hf_model_path not found` | Download HF weights (see Step 6) or disable Layer 3: `"training": {"enabled": false}` |

### Build and Runtime

| Problem | Fix |
|:--------|:----|
| `libllama.so not found` at runtime | `export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/lib:$LD_LIBRARY_PATH` — make permanent in `~/.bashrc` |
| `SWI-Prolog not found` during CMake | `find /usr -name "SWI-Prolog.h" 2>/dev/null` then pass `cmake .. -DSWIPL_INCLUDE_DIR=<path>` |
| `CUDA not found` during CMake | `export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` then re-run CMake |
| `Permission denied` when running cardinal | `chmod +x ~/cardinal/build/bin/cardinal` |
| `CUDA out of memory` on startup | Reduce `gpu_layers` in `config.json`; vision and training run on CPU |
| `n_ubatch` error in llama.cpp | In `src/core/backends/llama_cpp_backend.cpp` in `create_context()`, adjust `ctx_params.n_ubatch` (default `512`) up or down to match your GPU |
| HTTP server not responding | Check port: `ss -tlnp \| grep 8080`; if taken, set `"port": 8181` in `config.json` |
| Cardinal starts but immediately exits | Check `logs/cardinal.log`; common causes: missing model file, wrong grammar path in `config.json`, SQLite permission error on `data/memory/` |
| Docker sandbox permission denied | `sudo usermod -aG docker $USER && newgrp docker` then restart Cardinal |
| Agentic loop crashes with `max_iterations` | Lower `max_iterations` in `config.json`: `"agent": { "max_iterations": 5, "max_iterations_hard_cap": 20 }` |

---

## Changing the API Key

The default API key `cardinal-dev-key-change-in-production` is intentionally obvious. Change it before any networked deployment:

```json
"api": {
    "auth_enabled": true,
    "api_key": "your-strong-random-key-here"
}
```

To disable auth entirely for local development:

```json
"api": {
    "auth_enabled": false
}
```

---

## Directory Reference

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
            agent_working_memory        (SQLite, created at runtime)
        self_model/                     (added v1.4.0)
            self_model.db               ← Layer 1 SQLite accumulator
        training/                       (added v1.4.0)
            adapters/                   ← trained GGUF adapters
            datasets/                   ← curated JSONL datasets
            scripts/                    ← TensorRT training scripts
        scheduler/                      (added v1.5.0)
            scheduler.db                ← tasks, runs, action_logs
        explainability/                 (added v1.2.0)
            audit.db
            cardinal_private.pem        ← Ed25519 private key, auto-generated
            cardinal_public.pem
            exports/
        vision_cache/                   (added v1.3.0)
        browser_profile/                (added v1.5.0)
        screenshots/                    (added v1.5.0)
        training_export.jsonl           (generated on first export)
    logs/
        cardinal.log
        episodic.log
    models/
        Qwen_Qwen3.5-4B-Q4_K_M.gguf
        Llama-3.2-1B-Instruct-Q4_K_M.gguf    (optional)
        qwen3.5-4b-hf/                  (added v1.4.0) ← HF weights for Layer 3 training
        vision/                         (added v1.3.0)
            moondream2-text-model-f16.gguf
            moondream2-mmproj-f16.gguf
        voice/                          (added v1.6.0)
            ggml-medium.en.bin          ← Whisper STT model
            en_US-lessac-medium.onnx    ← Piper TTS model
            en_US-lessac-medium.onnx.json
            pocketsphinx/               ← Wake word (optional)
                en-us/                  ← acoustic model
                cmudict-en-us.dict
    scripts/
        populate_vendor.sh              (optional helper script)
    src/
        api/
        agent/
        computer/                       (added v1.5.0)
        scheduler/                      (added v1.5.0)
        watch/                          (added v1.5.0)
        voice/                          (added v1.6.0)
            audio_device.h/.cpp
            vad_detector.h/.cpp
            stt_engine.h/.cpp
            tts_engine.h/.cpp
            wake_word_detector.h/.cpp
            voice_loop.h/.cpp
        self_model/                     (added v1.4.0)
            self_model_types.h
            self_model.h/.cpp           ← Layer 1
            meta_cognition.h/.cpp       ← Layer 2
        training/                       (added v1.4.0)
            i_training_backend.h
            llama_cpp_trainer.h/.cpp
            tensorrt_trainer.h/.cpp
            training_factory.h/.cpp
            curriculum_builder.h/.cpp
            dataset_curator.h/.cpp
            adapter_evaluator.h/.cpp
            self_improvement_loop.h/.cpp
        vision/                         (added v1.3.0)
            vision_types.h
            vision_encoder.h/.cpp
            vision_cache.h/.cpp
        tools/
            builtin/
                analyze_image.h/.cpp    (added v1.3.0)
                computer/               (added v1.5.0)
                voice/                  (added v1.6.0)
                    tool_voice_control.h/.cpp
        core/
        memory/
        verifier/
        explainability/
        learning/
        utils/
    vendor/
        llama.cpp/
            build/
                lib/
                    libllama.so
                    libggml.so
                    libggml-cuda.so
                    libmtmd.so          (required for vision)
        nlohmann_json/
        cpp-httplib/
        tokenizers-cpp/
        muparser/                       (added v1.4.0)
        whisper.cpp/                    (added v1.6.0)
        onnxruntime/                    (added v1.6.0)
        piper/                          (added v1.6.0)
            install/                    ← pre-built libs + espeak-ng-data
        portaudio/                      (added v1.6.0)
        pocketsphinx/                   (added v1.6.0)
            install/                    ← pre-built lib + headers
    cardinal-browser-venv/              (added v1.5.0) ← Playwright venv
    cardinal-train-venv/                (added v1.4.0) ← PEFT training venv
    config.json
    CMakeLists.txt
    README.md
    INSTALL.md
    DOCUMENTATION.md
```

---

## Next Steps

Once Cardinal is running:

- Read `README.md` for the full architecture overview.
- Read `DOCUMENTATION.md` for the complete API, agentic pipeline, explainability system, vision and voice subsystem internals, and configuration reference.
- Use `/stats` to see memory and verifier state.
- Use `/rules` to watch the rule base grow over time.
- Query past episodes: `GET /api/episodes?keyword=your+query`
- Export training data: `/export` or `POST /api/export`
- Export signed explainability traces: `POST /api/explainability/export`
- Monitor self-knowledge: `GET /api/self_model`
- Trigger on-demand reflection: `POST /api/reflect`
- Trigger a LoRA training cycle: `POST /api/train`

---

*If something in this guide is wrong or out of date, the source of truth is always `CMakeLists.txt`, `config.json`, and the source code in the repository.*
