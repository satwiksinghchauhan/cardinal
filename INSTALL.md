# Cardinal v1.6.0 — Installation Guide

**Linux only. Ubuntu 24.04 LTS recommended. No git submodules — all vendor dependencies cloned manually.**

v1.6.0 adds the Voice subsystem. This requires four new vendored libraries (whisper.cpp, piper, portaudio, pocketsphinx), two pre-build steps (piper and pocketsphinx must be built into local install prefixes before Cardinal's cmake), voice model files, and ALSA dev headers.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Step 1 — System Dependencies](#step-1--system-dependencies)
- [Step 2 — CUDA Toolkit](#step-2--cuda-toolkit)
- [Step 3 — Vendor Dependencies](#step-3--vendor-dependencies)
- [Step 4 — Build llama.cpp](#step-4--build-llamacpp)
- [Step 5 — Build Voice Vendor Libs](#step-5--build-voice-vendor-libs)
- [Step 6 — Download Models](#step-6--download-models)
- [Step 7 — Build Cardinal](#step-7--build-cardinal)
- [Step 8 — Browser Venv (Playwright)](#step-8--browser-venv-playwright)
- [Step 9 — Training Venv (Layer 3 LoRA)](#step-9--training-venv-layer-3-lora-optional)
- [Step 10 — Email Setup](#step-10--email-setup)
- [Step 11 — Wayland Input (ydotool)](#step-11--wayland-input-ydotool)
- [Step 12 — config.json](#step-12--configjson)
- [Step 13 — Environment Variables](#step-13--environment-variables)
- [Step 14 — Data Directories](#step-14--data-directories)
- [First Run](#first-run)
- [Verify Voice](#verify-voice)
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

```bash
# 1. System dependencies (including voice deps)
sudo apt update
sudo apt install -y build-essential cmake libsqlite3-dev libssl-dev swi-prolog \
    python3 python3-pip python3-venv \
    libasound2-dev \
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
git clone https://github.com/ggerganov/whisper.cpp.git
git clone https://github.com/rhasspy/piper.git
git clone https://github.com/PortAudio/portaudio.git
git clone https://github.com/cmusphinx/pocketsphinx.git

# 3. Build llama.cpp
cd ~/cardinal/vendor/llama.cpp && mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && cd ~/cardinal

# 4. Pre-build piper (downloads onnxruntime, espeak-ng, fmt, spdlog automatically)
cd vendor/piper
cmake -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install -j$(nproc)
cd ~/cardinal

# 5. Pre-build pocketsphinx
cd vendor/pocketsphinx
cmake -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install -j$(nproc)
cd ~/cardinal

# 6. Build Cardinal
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# 7. Browser venv
python3 -m venv ~/cardinal/cardinal-browser-venv
source ~/cardinal/cardinal-browser-venv/bin/activate
pip install playwright && playwright install chromium && playwright install-deps chromium
deactivate

# 8. Run with voice
cd ~/cardinal && ./build/bin/cardinal --voice
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

### v1.6.0 — Voice dependencies

```bash
sudo apt install -y \
    libasound2-dev       # ALSA — required by PortAudio on Linux
    pocketsphinx
    pocketsphinx-en-us
    portaudio19-dev
    libportaudio2
```

And the ONNX Runtime for Piper (pre-built binary, don't clone):

```bash
cd ~/cardinal/vendor
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-gpu-1.17.3.tgz
tar xzf onnxruntime-linux-x64-gpu-1.17.3.tgz
mv onnxruntime-linux-x64-gpu-1.17.3 onnxruntime
rm onnxruntime-linux-x64-gpu-1.17.3.tgz
```


That is the only new system package for voice. `onnxruntime`, `espeak-ng`, `fmt`, and `spdlog` are all downloaded and built automatically by piper's CMake during Step 5.

### v1.5.0 — Computer Use tools

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

Verify key tools:

```bash
cmake --version          # 3.28 or higher
swipl --version          # SWI-Prolog 9.x
python3 --version        # 3.10+
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

```bash
cd ~/cardinal
mkdir -p vendor && cd vendor

# Original deps
git clone https://github.com/ggerganov/llama.cpp.git
git clone https://github.com/nlohmann/json.git nlohmann_json
git clone https://github.com/yhirose/cpp-httplib.git
git clone https://github.com/beltoforion/muparser.git
git clone https://github.com/mlc-ai/tokenizers-cpp

# v1.6.0 voice deps
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
| muparser | https://github.com/beltoforion/muparser | `vendor/muparser` |
| tokenizers-cpp | https://github.com/mlc-ai/tokenizers-cpp | `vendor/tokenizers-cpp` |
| whisper.cpp | https://github.com/ggerganov/whisper.cpp | `vendor/whisper.cpp` |
| piper | https://github.com/rhasspy/piper | `vendor/piper` |
| portaudio | https://github.com/PortAudio/portaudio | `vendor/portaudio` |
| pocketsphinx | https://github.com/cmusphinx/pocketsphinx | `vendor/pocketsphinx` |

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

## Step 5 — Build Voice Vendor Libs

Whisper.cpp is built automatically by Cardinal's CMake (`add_subdirectory`). Piper and PocketSphinx must be **pre-built** into local install prefixes because their CMake systems use `ExternalProject_Add` and are not designed for `add_subdirectory`. This is a one-time step.

### 5a — Build piper

Piper's build downloads its own bundled versions of `onnxruntime`, `espeak-ng`, `fmt`, and `spdlog` automatically. This step requires an internet connection and takes 5–10 minutes.

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
    piper                         ← the piper binary (not used directly by Cardinal)
    libpiper_phonemize.so.1.2.0
    libonnxruntime.so.1.14.1
    libespeak-ng.so.1.52.0.1
    espeak-ng-data/               ← language data for espeak-ng
    include/                      ← piper-phonemize headers
    pkgconfig/
```

Add piper's libs to `LD_LIBRARY_PATH` so the Cardinal binary can find them at runtime:

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

---

## Step 6 — Download Models

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

### Voice models (v1.6.0)

```bash
mkdir -p ~/cardinal/models/voice
cd ~/cardinal/models/voice
```

#### Whisper STT model

```bash
# Medium English model — best accuracy/speed on RTX 3050
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin
```

Smaller alternatives (less accurate, faster):

```bash
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
```

Update `config.json` `voice.stt.model_path` if you use a different model.

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

### HuggingFace weights for training (Layer 3, optional)

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
-- Cardinal v1.6.0 Configuration
-- =============================================================
-- Build type:         Release
-- CUDA arch:          75
-- TensorRT backend:   OFF
-- Voice subsystem:    ON
-- Piper install:      /home/doctor/cardinal/vendor/piper/install
-- PocketSphinx:       /home/doctor/cardinal/vendor/pocketsphinx/install
-- Output dir:         /home/doctor/cardinal/build/bin
-- =============================================================
```

**To disable the voice subsystem** (e.g. on a machine without a microphone):

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCARDINAL_ENABLE_VOICE=OFF
```

---

## Step 8 — Browser Venv (Playwright)

Required for the `browser` computer use tool:

```bash
python3 -m venv ~/cardinal/cardinal-browser-venv
source ~/cardinal/cardinal-browser-venv/bin/activate
pip install playwright
playwright install chromium
playwright install-deps chromium
deactivate
```

Set `computer_use.browser.venv_path` in `config.json` to `~/cardinal/cardinal-browser-venv`.

---

## Step 9 — Training Venv (Layer 3 LoRA, optional)

Required only if you want local LoRA fine-tuning:

```bash
python3 -m venv ~/cardinal/cardinal-train-venv
source ~/cardinal/cardinal-train-venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft datasets accelerate bitsandbytes
deactivate
```

---

## Step 10 — Email Setup

Required only if `computer_use.email.enabled=true`:

```bash
export CARDINAL_EMAIL_PASS="your_app_password"
echo 'export CARDINAL_EMAIL_PASS="your_app_password"' >> ~/.bashrc
```

For Gmail: use an **App Password** (Google Account → Security → App Passwords). Enable IMAP in Gmail settings.

---

## Step 11 — Wayland Input (ydotool)

Required only if running Wayland (not X11). `ydotool` requires the `uinput` kernel module and daemon:

```bash
sudo modprobe uinput
sudo usermod -aG input $USER
# Re-login, then:
sudo ydotoold &
# Or for persistence:
systemctl --user enable --now ydotool
```

---

## Step 12 — config.json

The config file lives at `~/cardinal/config.json`. The full reference is in `DOCUMENTATION.md`.

### v1.6.0 voice block (add to config.json)

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

```bash
# Email password (required if email.enabled=true and mode=imap_smtp)
export CARDINAL_EMAIL_PASS="your_app_password"
echo 'export CARDINAL_EMAIL_PASS="your_app_password"' >> ~/.bashrc

# llama.cpp libraries
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Piper runtime libs (flat install dir — no lib/ subdirectory)
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/piper/install:$LD_LIBRARY_PATH' >> ~/.bashrc

# PocketSphinx
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/pocketsphinx/install/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# CUDA
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

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

Always run from the Cardinal root directory:

```bash
cd ~/cardinal

# Text mode (unchanged)
./build/bin/cardinal

# Voice mode — VAD (from config)
./build/bin/cardinal --voice

# Voice mode — push-to-talk
./build/bin/cardinal --voice=ptt

# Voice mode — wake word ("hey cardinal")
./build/bin/cardinal --voice=wake
```

Expected startup output (v1.6.0 with voice):

```
[INFO ] SelfModel: opened database at data/self_model/self_model.db
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
  |        C A R D I N A L  v1.6.0           |
  |   Neurosymbolic AGI Architecture          |
  +===========================================+
  Voice mode active
```

---

## Verify Voice

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

## Verify Computer Use

```bash
# Status
curl -s http://127.0.0.1:8080/api/computer/status \
  -H "Authorization: Bearer secret_api_key"
# Returns: {"width":1920,"height":1080,"server":"x11","display_var":":0"}

# Screenshot
curl -s -X POST http://127.0.0.1:8080/api/computer/screenshot \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"analyze": true}' | python3 -m json.tool
```

Or from CLI: `/computer`

---

## Verify Scheduler

```bash
# List tasks
curl -s http://127.0.0.1:8080/api/scheduler/tasks \
  -H "Authorization: Bearer secret_api_key" | python3 -m json.tool

# Create a task
curl -s -X POST http://127.0.0.1:8080/api/scheduler/tasks \
  -H "Authorization: Bearer secret_api_key" \
  -H "Content-Type: application/json" \
  -d '{"description": "search for AI news every morning at 7am"}' | python3 -m json.tool
```

Or from CLI: `/scheduler` and `/tasks`

---

## Verify Self-Improvement (v1.4.0)

```bash
curl -s http://127.0.0.1:8080/api/self_model -H "Authorization: Bearer secret_api_key" | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8080/api/reflect -H "Authorization: Bearer secret_api_key" | python3 -m json.tool
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
curl http://127.0.0.1:8080/api/health

curl -X POST http://127.0.0.1:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer secret_api_key" \
  -d '{"session_id":"test","message":"Hello!"}'
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
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
| `scrot: command not found` | `sudo apt install scrot` |
| `xdotool: command not found` | `sudo apt install xdotool` |
| `ydotool: /dev/uinput permission denied` | `sudo usermod -aG input $USER` then re-login |
| `browser: Playwright not found` | Run Step 8 (browser venv setup) |
| `mtmd.h not found` during CMake | Rebuild llama.cpp with `-DLLAMA_BUILD_EXAMPLES=ON` |
| `CUDA out of memory` | Reduce `gpu_layers` in `config.json` and/or `voice.stt.gpu_layers` |
| Reflection returns `ran=false` | Lower `min_failures_to_reflect` to 1 for testing |

---

## Changing the API Key

```json
"api": {
    "auth_enabled": true,
    "api_key": "your-strong-random-key-here"
}
```

---

## Directory Reference (v1.6.0)

```
~/cardinal/
    build/
        bin/
            cardinal
    data/
        memory/
            rules.json, knowledge.json, episodes.db
        self_model/
            self_model.db
        training/
            adapters/, datasets/, scripts/
        scheduler/
            scheduler.db
        explainability/
            audit.db, exports/
        vision_cache/
        browser_profile/
        screenshots/
    logs/
        cardinal.log, episodic.log
    models/
        Qwen_Qwen3.5-4B-Q4_K_M.gguf
        qwen3.5-4b-hf/
        vision/
            moondream2-text-model-f16.gguf
            moondream2-mmproj-f16.gguf
        voice/                                   ← NEW v1.6.0
            ggml-medium.en.bin                   ← Whisper STT model
            en_US-lessac-medium.onnx             ← Piper TTS model
            en_US-lessac-medium.onnx.json        ← Piper voice config
            pocketsphinx/                        ← Wake word (optional)
                en-us/                           ← Acoustic model
                cmudict-en-us.dict               ← Pronunciation dictionary
    src/
        api/
        agent/
        computer/
        scheduler/
        watch/
        voice/                                   ← NEW v1.6.0
            audio_device.h/.cpp
            vad_detector.h/.cpp
            stt_engine.h/.cpp
            tts_engine.h/.cpp
            wake_word_detector.h/.cpp
            voice_loop.h/.cpp
        tools/
            builtin/
                computer/
                voice/                           ← NEW v1.6.0
                    tool_voice_control.h/.cpp
        self_model/, training/
        core/, memory/, verifier/, vision/
        explainability/, learning/
        utils/
    vendor/
        llama.cpp/
        nlohmann_json/
        cpp-httplib/
        muparser/
        tokenizers-cpp/
        onnxruntime/                             ← NEW v1.6.0
        whisper.cpp/                             ← NEW v1.6.0
        piper/                                   ← NEW v1.6.0
            install/                             ← pre-built libs + espeak-ng-data
        portaudio/                               ← NEW v1.6.0
        pocketsphinx/                            ← NEW v1.6.0
            install/                             ← pre-built lib + headers
    cardinal-browser-venv/
    cardinal-train-venv/
    config.json
    CMakeLists.txt
    README.md, INSTALL.md, DOCUMENTATION.md
```

---

*If something in this guide is wrong or out of date, the source of truth is always `CMakeLists.txt`, `config.json`, and the source code.*
