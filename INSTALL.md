# Cardinal v1.3.0 — Installation Guide

**Linux only. Offline builds. Ubuntu 24.04 LTS recommended.**

Cardinal v1.3.0 adds native vision encoding (moondream2 via llama.cpp's `mtmd` subsystem). The build remains fully offline once dependencies are vendored.

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
- [Verify Vision (v1.3.0)](#verify-vision-v130)
- [Verify the HTTP API](#verify-the-http-api)
- [Python Sandbox Setup (Optional)](#python-sandbox-setup-optional)
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
| GPU | NVIDIA with 4GB VRAM | RTX 3050 or better (vision adds ~1.5GB) |
| Storage | 25GB free | 50GB free |
| CUDA Compute | 7.5+ | 8.6 (RTX 30 series) |

Cardinal does **not** run on AMD or Intel GPUs. NVIDIA CUDA is required.

**Windows is no longer supported as of v1.1.0.**

---

## Quick Start (5 minutes)

For the impatient:

```bash
# Install system dependencies
sudo apt update
sudo apt install -y build-essential cmake libsqlite3-dev libssl-dev swi-prolog python3

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
    swi-prolog \
    python3
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
| llama.cpp | https://github.com/ggerganov/llama.cpp | `vendor/llama.cpp` |
| nlohmann/json | https://github.com/nlohmann/json | `vendor/nlohmann_json` |
| cpp-httplib | https://github.com/yhirose/cpp-httplib | `vendor/cpp-httplib` |
| tokenizers-cpp | https://github.com/mlc-ai/tokenizers-cpp | `vendor/tokenizers-cpp` |

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

Cardinal requires a primary GGUF model and the vision encoder models.

#### Primary LLM (required)

- **Qwen3.5 4B Q4_K_M**
- Download from: [bartowski/Qwen_Qwen3.5-4B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF)
- File: `Qwen_Qwen3.5-4B-Q4_K_M.gguf` (~2.5GB)
- Place at: `~/cardinal/models/Qwen_Qwen3.5-4B-Q4_K_M.gguf`

#### Vision encoder (v1.3.0, optional but recommended)

```bash
mkdir -p ~/cardinal/models/vision
cd ~/cardinal/models/vision

# Text model (~1.1GB)
wget https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-text-model-f16.gguf

# Vision projector (~400MB)
wget https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-mmproj-f16.gguf
```
If those links dont work you can manually install from https://huggingface.co/salivosa/moondream2-gguf, just make sure to change the model path in config.json

If you skip the vision models, set `vision.enabled = false` in `config.json`.

#### Neural verifier (optional)

- Llama 3.2 1B Instruct Q4_K_M (~700MB)
- Download from: [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)
- Place at: `~/cardinal/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`

If you skip the neural verifier, set `verifier.mode = "symbolic"` in `config.json`.

---

### Step 5 — Build llama.cpp with CUDA and mtmd

**Important:** v1.3.0 uses the new `mtmd` (multimodal) subsystem, not the old `llava` example.  
Enable `LLAMA_BUILD_EXAMPLES` (which includes `mtmd`) and ensure CUDA is on.

```bash
cd ~/cardinal/vendor/llama.cpp
git checkout b8660   # or whatever commit works for you
mkdir -p build && cd build
cmake .. \
    -DGGML_CUDA=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**What this builds:**
- `libllama.so`, `libggml.so`, `libggml-cuda.so`
- `libmtmd.so` (multimodal wrapper, required for vision)
- `llama-mtmd-cli` (test utility, optional)

Verify:
```bash
ls build/bin/
# Should contain: libllama.so, libggml.so, libmtmd.so, ...
```

---

### Step 6 — Build Cardinal

```bash
cd ~/cardinal
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**CMake will automatically detect `mtmd`** from `vendor/llama.cpp/tools/mtmd`.  
If you see:

```
-- mtmd: found at .../vendor/llama.cpp/tools/mtmd — vision will be enabled
```

then vision is enabled. If not, check that `llama.cpp` was built with `-DLLAMA_BUILD_EXAMPLES=ON`.

#### Optional: Build with TensorRT Backend

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCARDINAL_ENABLE_TENSORRT=ON \
    -DTRT_LLM_INCLUDE_DIR=/path/to/TensorRT-LLM/include \
    -DTRT_LLM_LIB_DIR=/path/to/TensorRT-LLM/lib
make -j$(nproc)
```

**Switching back to llama.cpp:** Omit the `-DCARDINAL_ENABLE_TENSORRT=ON` flag.

A successful build produces `cardinal` in `~/cardinal/build/bin/`.

---

## First Run

Run from the cardinal root directory so relative paths in `config.json` resolve:

```bash
cd ~/cardinal
./build/bin/cardinal
```

If vision models are correctly configured and `mtmd` was found, you will see:

```
[INFO ] VisionCache: initialized at data/vision_cache (TTL=24h)
[INFO ] VisionEncoder: loading text model: models/vision/moondream2-text-model-f16.gguf
[INFO ] VisionEncoder: loading mmproj:     models/vision/moondream2-mmproj-f16.gguf
[INFO ] VisionEncoder: ready (CPU, 4 threads)
[INFO ] Vision encoder ready (moondream2, CPU)
```

If vision models are missing, you will see:

```
[WARN ] VisionEncoder: built without mtmd support — vision disabled
[WARN ]   Add vendor/llama.cpp/tools/mtmd to include path in CMakeLists
```

In that case, check that:
- `llama.cpp` was built with `-DLLAMA_BUILD_EXAMPLES=ON`
- The headers in `vendor/llama.cpp/tools/mtmd/` exist
- The models are present at the paths specified in `config.json`

Then you should see the Cardinal prompt:

```
  +===========================================+
  |         C A R D I N A L  v1.3.0           |
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

## Verify Vision (v1.3.0)

Place a test image (e.g., `test.jpg`) in `data/` and ask:

```
You: describe the image at data/test.jpg
```

Cardinal will call the `analyze_image` tool and return a description.

Example output:

```
Cardinal: The image shows a portrait of a young man with glasses and a nose ring. He looks directly at the camera with a friendly and approachable demeanor.
```

---

## Verify the HTTP API

Open a new terminal while Cardinal is running:

```bash
# Health check — no auth required
curl http://127.0.0.1:8080/api/health

# Chat with vision
curl -X POST http://127.0.0.1:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer cardinal-dev-key-change-in-production" \
  -d '{"session_id":"test","message":"describe the image at data/test.jpg"}'

# Agentic request (with vision tool)
curl -X POST http://127.0.0.1:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer cardinal-dev-key-change-in-production" \
  -d '{
    "session_id": "agent-session",
    "message": "what is in https://example.com/image.jpg",
    "max_iterations": 5
  }'
```

---

## Troubleshooting

### `mtmd.h not found` during CMake

Ensure `llama.cpp` was built with `-DLLAMA_BUILD_EXAMPLES=ON`.  
If you already built it, you can still add the include path manually:

```bash
cd ~/cardinal/build
cmake .. -DCMAKE_CXX_FLAGS="-I/path/to/llama.cpp/tools/mtmd"
```

But this should not be necessary if the CMake detection works.

### `libmtmd.so not found` at runtime

```bash
export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/bin:$LD_LIBRARY_PATH
./build/bin/cardinal
```

Make it permanent:

```bash
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Vision encoder says "built without mtmd support"

- Check that `CARDINAL_MTMD_AVAILABLE` is defined in the build summary.
- Re‑run CMake and ensure the detection block prints `mtmd: found at ...`
- If not, manually set `-DCARDINAL_MTMD_AVAILABLE=ON` during CMake.

### `CUDA out of memory` on startup

Reduce `gpu_layers` in `config.json`. Vision encoding runs on CPU; you only need VRAM for the primary LLM.

### Vision models not loading

- Verify the paths in `config.json` point to existing files.
- Ensure the files are not corrupted (re‑download if necessary).
- The text model should be ~1.1GB, the mmproj ~400MB.

### Docker sandbox permission denied (run_python tool)

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Then restart Cardinal.

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

## Directory Reference (v1.3.0)

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
            agent_working_memory   (SQLite, created at runtime)
        explainability/
            audit.db
            cardinal_private.pem
            cardinal_public.pem
            exports/
        vision_cache/              ← (new in v1.3.0) cached downloaded images
    logs/
        cardinal.log
        episodic.log
    models/
        Qwen_Qwen3.5-4B-Q4_K_M.gguf
        Llama-3.2-1B-Instruct-Q4_K_M.gguf   (optional)
        vision/
            moondream2-text-model-f16.gguf  ← (v1.3.0)
            moondream2-mmproj-f16.gguf      ← (v1.3.0)
    src/
        vision/                    ← (v1.3.0)
            vision_types.h
            vision_encoder.h/.cpp
            vision_cache.h/.cpp
        tools/builtin/
            analyze_image.h/.cpp   ← (v1.3.0)
        ... (rest of source)
    vendor/
        llama.cpp/
            build/
                bin/
                    libmtmd.so     ← (required for vision)
                    ...
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

Once Cardinal is running with vision:

- Read `README.md` for the full architecture overview.
- Read `DOCUMENTATION.md` for the complete API, agentic pipeline, explainability system, and vision subsystem internals.
- Use `/stats` to see memory and verifier state.
- Export training data: `/export` or `POST /api/export`.
- Export signed explainability traces: `POST /api/explainability/export`.

---

*If something in this guide is wrong or out of date, the source of truth is always the `CMakeLists.txt`, `config.json`, and source code in the repository.*
