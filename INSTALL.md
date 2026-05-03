# Cardinal v1.1.0 — Installation Guide

**Linux only. Offline builds. Ubuntu 24.04 LTS recommended.**

Cardinal v1.1.0 has removed Windows support. For development and deployment, use Ubuntu 24.04 LTS (or newer LTS). The build is fully offline once dependencies are vendored.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start (5 minutes)](#quick-start-5-minutes)
- [Detailed Installation](#detailed-installation)
  - [System Dependencies](#system-dependencies)
  - [CUDA Toolkit](#cuda-toolkit)
  - [Vendored Dependencies](#vendored-dependencies)
  - [Download Models](#download-models)
  - [Build Cardinal](#build-cardinal)
- [First Run](#first-run)
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
| RAM | 8GB | 16GB |
| GPU | NVIDIA with 4GB VRAM | RTX 3050 or better |
| Storage | 20GB free | 50GB free |
| CUDA Compute | 7.5+ | 8.6 (RTX 30 series) |

Cardinal does not run on AMD or Intel GPUs. NVIDIA CUDA is required.

**Windows is no longer supported as of v1.1.0.**

---

## Quick Start (5 minutes)

For the impatient:

```bash
# Install system dependencies
sudo apt update
sudo apt install -y build-essential cmake libsqlite3-dev libssl-dev swi-prolog

# Clone Cardinal
git clone https://github.com/satwiksinghchauhan/cardinal ~/cardinal
cd ~/cardinal

# Populate vendor dependencies (see section below for manual clone)
./scripts/populate_vendor.sh   # if provided, or clone manually

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

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
    swi-prolog
```

Verify:

```bash
cmake --version    # must be 3.28 or higher
swipl --version    # should print SWI-Prolog 9.2.9 or newer
openssl version    # should print 3.0.x or newer
```

If your distro ships CMake older than 3.28, install it manually:

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
nvcc --version    # should print: release 12.6
nvidia-smi        # should show your GPU
```

---

### Step 3 — Vendored Dependencies

Cardinal uses a **clean vendor folder** for offline builds. You populate `vendor/` with these repositories:

| Dependency | Upstream URL | Clone as |
| :--- | :--- | :--- |
| llama.cpp | https://github.com/ggerganov/llama.cpp | vendor/llama.cpp |
| nlohmann/json | https://github.com/nlohmann/json | vendor/nlohmann_json |
| cpp-httplib | https://github.com/yhirose/cpp-httplib | vendor/cpp-httplib |
| tokenizers-cpp | https://github.com/mlc-ai/tokenizers-cpp | vendor/tokenizers-cpp |

**Clone them manually:**

```bash
cd ~/cardinal
mkdir -p vendor
cd vendor

git clone https://github.com/ggerganov/llama.cpp.git
git clone https://github.com/nlohmann/json.git nlohmann_json
git clone https://github.com/yhirose/cpp-httplib.git
git clone https://github.com/mlc-ai/tokenizers-cpp.git
```

**Or use the provided script:**

```bash
cd ~/cardinal
chmod +x scripts/populate_vendor.sh
./scripts/populate_vendor.sh
```

---

### Step 4 — Download Models

Cardinal requires a primary GGUF model. The neural verifier is optional.

**Primary model (required):**
- Qwen3.5 4B Q4_K_M
- Download from: https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF
- File: `Qwen_Qwen3.5-4B-Q4_K_M.gguf` (~2.5GB)
- Place at: `~/cardinal/models/Qwen_Qwen3.5-4B-Q4_K_M.gguf`

**Neural verifier model (optional):**
- Llama 3.2 1B Instruct Q4_K_M
- Download from: https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF
- File: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (~700MB)
- Place at: `~/cardinal/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`

If you skip the neural verifier, set `verifier.mode` to `"symbolic"` in `config.json` and leave `neural_model_path` empty.

---

### Step 5 — Build Cardinal

```bash
cd ~/cardinal
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
### Optional: Build with TensorRT Backend

Cardinal v1.1.0 supports NVIDIA TensorRT for optimized inference on supported GPUs.

**Prerequisites:**
- NVIDIA TensorRT installed (typically `/usr/lib/x86_64-linux-gnu/` or `/opt/TensorRT-xxx/`)
- TRT-LLM (TensorRT LLM) for LLM inference

**Build with TensorRT:**

```bash
cd ~/cardinal
mkdir -p build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCARDINAL_ENABLE_TENSORRT=ON \
    -DTRT_LLM_INCLUDE_DIR=/path/to/TensorRT-LLM/include \
    -DTRT_LLM_LIB_DIR=/path/to/TensorRT-LLM/lib

make -j$(nproc)
```

**Note:** If `TRT_LLM_INCLUDE_DIR` and `TRT_LLM_LIB_DIR` are not specified, CMake will attempt to find TensorRT in standard system paths.

**Switching back to llama.cpp:** Omit the `-DCARDINAL_ENABLE_TENSORRT=ON` flag.

A successful build produces `cardinal` in `~/cardinal/build/bin/`.

**If CMake cannot find SWI-Prolog headers:**

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

## First Run

Run from the cardinal root directory so relative paths in `config.json` resolve:

```bash
cd ~/cardinal
./build/bin/cardinal
```

You should see:

```
  +===========================================+
  |         C A R D I N A L  v1.1.0           |
  |    Neurosymbolic AGI Architecture         |
  +===========================================+

  HTTP API: http://127.0.0.1:8080
  TypeScript bridge: ready

  Commands:
    /exit        -- quit
    /reset       -- clear conversation history
    /rules       -- show active rule store
    /stats       -- show memory and verifier stats
    /export      -- export training data to JSONL
    /scan        -- run full contradiction scan
    /http start  -- start HTTP server
    /http stop   -- stop HTTP server

  Cardinal is ready. Type your message.

You:
```

---

## Verify the HTTP API

Open a new terminal while Cardinal is running:

```bash
# Health check — no auth required
curl http://127.0.0.1:8080/api/health

# Chat
curl -X POST http://127.0.0.1:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer cardinal-dev-key-change-in-production" \
  -d '{"session_id":"test","message":"What is entropy?"}'
```

---

## Troubleshooting

### `libllama.so not found` at runtime

The RPATH is set in CMakeLists, but sometimes the loader still can't find the library:

```bash
export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/lib:$LD_LIBRARY_PATH
./build/bin/cardinal
```

To make it permanent:

```bash
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

### `SWI-Prolog not found` during CMake

Find where SWI-Prolog installed its headers:

```bash
find /usr -name "SWI-Prolog.h" 2>/dev/null
```

Pass the path explicitly:

```bash
cmake .. -DSWIPL_INCLUDE_DIR=/usr/lib/swi-prolog/include
```

---

### `CUDA not found` during CMake

Make sure CUDA is on your PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Then re-run CMake.

---

### `Permission denied` when running cardinal

```bash
chmod +x ~/cardinal/build/bin/cardinal
```

---

### `CUDA out of memory` on startup

Reduce `gpu_layers` in `config.json`:

```json
"gpu_layers": 20
```

The remaining layers run on CPU. Performance decreases but Cardinal works.

---

### HTTP server not responding

Check the port is not in use:

```bash
ss -tlnp | grep 8080
```

If the port is taken, change it in `config.json`:

```json
"port": 8181
```

---

### Cardinal starts but immediately exits

Check the log file:

```bash
cat ~/cardinal/logs/cardinal.log
```

Common causes:
- Missing model file
- Wrong grammar path in `config.json`
- SQLite permission error on `data/memory/` directory

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
        training_export.jsonl   (generated on first export)
    logs/
        cardinal.log
        episodic.log
    models/
        Qwen_Qwen3.5-4B-Q4_K_M.gguf
        Llama-3.2-1B-Instruct-Q4_K_M.gguf
    scripts/
        populate_vendor.sh      (optional)
    src/
        ...
    vendor/
        llama.cpp/
            build/
                lib/
                    libllama.so
                    libggml.so
                    libggml-cuda.so
        nlohmann_json/
        cpp-httplib/
        tokenizers-cpp/
    config.json
    CMakeLists.txt
    README.md
    INSTALL.md
    DOCUMENTATION.md
```

---

## Next Steps

Once Cardinal is running:

- Read `README.md` for a full architecture overview
- Use `/stats` to see memory and verifier state
- Use `/rules` to watch the rule base grow over time
- Query past episodes: `GET /api/episodes?keyword=your+query`
- Export training data: `/export` or `POST /api/export`

---

*If something in this guide is wrong or out of date, the source of truth is always the `CMakeLists.txt` and `config.json` in the repository.*

