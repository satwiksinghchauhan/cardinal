# Cardinal -- Installation Guide

This guide covers installation on both Windows and Linux.
Follow the section for your platform. Do not skip steps.

---

## Table of Contents

- [Windows Installation](#windows-installation)
- [Linux Installation](#linux-installation)
- [Platform-Independent Steps](#platform-independent-steps-both-platforms)
- [Troubleshooting](#troubleshooting)
- [Changing the API Key](#changing-the-api-key)
- [Directory Reference](#directory-reference)

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 64-bit or Ubuntu 20.04 | Windows 11 or Ubuntu 22.04 |
| CPU | Any x64, 4 cores | AMD Ryzen 7 or better |
| RAM | 8GB | 16GB |
| GPU | NVIDIA with 4GB VRAM | RTX 3050 or better |
| Storage | 20GB free | 50GB free |
| CUDA Compute | 7.5+ | 8.6 (RTX 30 series) |

Cardinal does not run on AMD or Intel GPUs. NVIDIA CUDA is required on both platforms.

---

## Windows Installation

### Step 1 -- Visual Studio 2022

Cardinal requires MSVC C++20. The free Community edition works.

1. Download Visual Studio 2022 Community from https://visualstudio.microsoft.com/
2. Run the installer
3. In the workload selector, check:
   - **Desktop development with C++**
   - **CUDA development** (under Individual Components if not shown)
4. Install and restart if prompted

Verify:
```powershell
cl
# Should print: Microsoft (R) C/C++ Optimizing Compiler
```

### Step 2 -- CUDA Toolkit 12.6

1. Go to https://developer.nvidia.com/cuda-toolkit-archive
2. Download CUDA Toolkit 12.6.3 for Windows x86_64
3. Run the installer -- choose Express installation
4. Restart your machine after installation

Verify:
```powershell
nvcc --version
# Should print: release 12.6

nvidia-smi
# Should show your GPU and driver version
```

### Step 3 -- CMake

1. Download CMake 3.28 or later from https://cmake.org/download/
2. Choose the Windows x64 installer
3. During install, select **Add CMake to the system PATH for all users**

Verify:
```powershell
cmake --version
# Should print: cmake version 3.28 or higher
```

### Step 4 -- Git

1. Download Git from https://git-scm.com/download/win
2. Install with default options

Verify:
```powershell
git --version
```

### Step 5 -- Clone Cardinal

```powershell
cd D:\
git clone https://github.com/satwiksinghchauhan/cardinal cardinal
cd cardinal
```

If you do not have the repository as a git repo, place the Cardinal source folder at `D:\cardinal\`.

### Step 6 -- SWI-Prolog 9.2.9

Cardinal's symbolic verifier requires SWI-Prolog. The version must be 9.2.9.

1. Go to https://www.swi-prolog.org/download/stable
2. Download the Windows 64-bit installer for version 9.2.9
3. Run the installer
4. Install to the default path: `C:\Program Files\swipl`
5. During install, check **Add SWI-Prolog to PATH**
6. Manually set the path just in case using "setx SWI_HOME_DIR "c:\program files\swipl" /M"

Verify:
```powershell
swipl --version
# Should print: SWI-Prolog version 9.2.9
```

If SWI-Prolog installed to a different path, update `SWIPL_ROOT` in `CMakeLists.txt`:
```cmake
set(SWIPL_ROOT "C:/Program Files/swipl")
```

### Step 7 -- vcpkg

vcpkg manages the SQLite and OpenSSL dependencies.

```powershell
cd D:\cardinal\vendor
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
```

Install required packages:
```powershell
.\vcpkg.exe install sqlite3:x64-windows
.\vcpkg.exe install openssl:x64-windows
```

This will take several minutes. Wait for both to complete.

Verify:
```powershell
.\vcpkg.exe list
# Should show sqlite3 and openssl in the list
```

### Step 8 -- Build llama.cpp with CUDA

```powershell
cd D:\cardinal\vendor
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git checkout b8660
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8
```

This will take 10-20 minutes.

Verify:
```powershell
ls build\bin\Release\
# Should show: llama.dll, ggml.dll, ggml-cuda.dll, ggml-base.dll, ggml-cpu.dll
```

### Step 9 -- Download Models

See [Platform-Independent Steps](#platform-independent-steps-both-platforms).

### Step 10 -- Configure

Create required directories:
```powershell
mkdir D:\cardinal\data\memory
mkdir D:\cardinal\logs
```

Open `config.json` and verify all paths are correct for your machine. Since all paths in `config.json` are relative, they should work as-is if you run Cardinal from its root directory.

### Step 11 -- Build Cardinal

Open a **Developer PowerShell for VS 2022** (not a regular PowerShell):

Start menu -> Visual Studio 2022 -> Developer PowerShell for VS 2022

Then:
```powershell
cd D:\cardinal
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j8
```

A successful build produces `cardinal.exe` in `D:\cardinal\build\bin\Release\`.

If CMake cannot find vcpkg, pass the toolchain path explicitly:
```powershell
cmake .. -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

### Step 12 -- First Run (Windows)

```powershell
cd D:\cardinal\build\bin\Release
.\cardinal.exe
```

### Step 13 -- Verify the HTTP API (Windows)

Open a new PowerShell window while Cardinal is running:

```powershell
# Health check -- no auth required
curl http://127.0.0.1:8080/api/health

# Chat
curl -X POST http://127.0.0.1:8080/api/chat `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer cardinal-dev-key-change-in-production" `
  -d "{\"session_id\":\"test\",\"message\":\"What is entropy?\"}"
```

---

## Linux Installation

Tested on Ubuntu 22.04. Other Debian-based distros should work identically.
For other distributions, substitute the package manager commands as needed.

### Step 1 -- System Dependencies

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
swipl --version    # should print SWI-Prolog version
openssl version
```

If your distro ships CMake older than 3.28, install it manually:
```bash
# Remove old cmake
sudo apt remove cmake

# Install from Kitware repo
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt update
sudo apt install cmake
```

### Step 2 -- CUDA Toolkit 12.6

Install the NVIDIA driver first if not already installed:
```bash
ubuntu-drivers autoinstall
sudo reboot
```

Then install CUDA:
1. Go to https://developer.nvidia.com/cuda-toolkit-archive
2. Select CUDA 12.6, Linux, x86_64, Ubuntu, your version, deb (network)
3. Follow the instructions on the download page -- they look like:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
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

### Step 3 -- Clone Cardinal

```bash
git clone https://github.com/satwiksinghchauhan/cardinal ~/cardinal
cd ~/cardinal
```

Or place the Cardinal source at `~/cardinal/` directly.

### Step 4 -- Build llama.cpp with CUDA

```bash
cd ~/cardinal/vendor
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git checkout b8660
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

This will take 10-20 minutes.

Verify:
```bash
ls build/lib/
# Should show: libllama.so, libggml.so, libggml-cuda.so (or similar)
# Exact names vary by llama.cpp version -- .a static libs are also acceptable
```

### Step 5 -- Download Models

See [Platform-Independent Steps](#platform-independent-steps-both-platforms).

### Step 6 -- Configure

Create required directories:
```bash
mkdir -p ~/cardinal/data/memory
mkdir -p ~/cardinal/logs
```

All paths in `config.json` are relative, so no path changes are needed as long as you run Cardinal from its root directory. Verify the config looks correct:

```bash
cat ~/cardinal/config.json
```

The paths should be relative like `"models/Qwen_Qwen3.5-4B-Q4_K_M.gguf"` not absolute. If you still have absolute Windows paths (`D:/cardinal/...`), replace them with relative paths.

### Step 7 -- Build Cardinal

```bash
cd ~/cardinal
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

A successful build produces `cardinal` (no extension) in `~/cardinal/build/bin/`.

If CMake cannot find SWI-Prolog automatically, check where it installed:
```bash
swipl --dump-runtime-variables | grep PLBASE
# Example output: PLBASE='/usr/lib/swi-prolog'
```

Then pass the include path explicitly:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DSWIPL_INCLUDE_DIR=/usr/lib/swi-prolog/include
```

If you installed dependencies via vcpkg instead of apt, pass the toolchain:
```bash
export VCPKG_ROOT=~/vcpkg
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

### Step 8 -- First Run (Linux)

Run from the cardinal root directory so relative paths in config resolve correctly:

```bash
cd ~/cardinal
./build/bin/cardinal
```

You should see:

```
  +===========================================+
  |         C A R D I N A L  v1.0.0           |
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

### Step 9 -- Verify the HTTP API (Linux)

Open a new terminal while Cardinal is running:

```bash
# Health check -- no auth required
curl http://127.0.0.1:8080/api/health

# Chat
curl -X POST http://127.0.0.1:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer cardinal-dev-key-change-in-production" \
  -d '{"session_id":"test","message":"What is entropy?"}'
```

---

## Platform-Independent Steps (Both Platforms)

### Download Models

Cardinal requires two models in GGUF format.

**Primary model (required):**
- Qwen3.5 4B Q4_K_M
- Download from: https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF
- File: `Qwen_Qwen3.5-4B-Q4_K_M.gguf` (~2.5GB)
- Place at: `models/Qwen_Qwen3.5-4B-Q4_K_M.gguf` inside your cardinal directory

**Neural verifier model (optional):**
- Llama 3.2 1B Instruct Q4_K_M
- Download from: https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF
- File: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (~700MB)
- Place at: `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` inside your cardinal directory

If you skip the neural verifier model, set `verifier.mode` to `"symbolic"` in `config.json` and leave `neural_model_path` empty. Cardinal will run fine without it.

---

## Troubleshooting

### Windows

**`llama.dll not found` on startup**

The llama.cpp DLLs are not in the build output directory. Run the build again:
```powershell
cmake --build . --config Release
```

Or copy them manually:
```powershell
cp D:\cardinal\vendor\llama.cpp\build\bin\Release\*.dll D:\cardinal\build\bin\Release\
```

---

**`SWI-Prolog not found` during CMake**

Update `SWIPL_ROOT` in `CMakeLists.txt`:
```cmake
set(SWIPL_ROOT "C:/Program Files/swipl")
```

---

**Build fails with `error C2065` or similar**

You are not in a Developer PowerShell. Open:
Start menu -> Visual Studio 2022 -> Developer PowerShell for VS 2022

---

**`Parse error` on first inference**

The grammar file path is wrong. Verify `config.json`:
```json
"grammar_path": "src/prompts/feeling_schema.gbnf"
```

The file must exist at that path relative to your working directory.

---

### Linux

**`libllama.so not found` at runtime**

The RPATH is set in the CMakeLists but sometimes the loader still can't find the library. Fix with:
```bash
export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/lib:$LD_LIBRARY_PATH
./build/bin/cardinal
```

To make this permanent:
```bash
echo 'export LD_LIBRARY_PATH=~/cardinal/vendor/llama.cpp/build/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

**`SWI-Prolog not found` during CMake**

Find where SWI-Prolog installed its headers:
```bash
find /usr -name "SWI-Prolog.h" 2>/dev/null
```

Pass the path explicitly to CMake:
```bash
cmake .. -DSWIPL_INCLUDE_DIR=/usr/lib/swi-prolog/include
```

---

**`CUDA not found` during CMake**

Make sure CUDA is on your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Then re-run CMake.

---

**`Permission denied` when running cardinal**

Make the binary executable:
```bash
chmod +x ~/cardinal/build/bin/cardinal
```

---

**llama.cpp build fails on Linux**

Make sure you have the CUDA compiler visible:
```bash
which nvcc   # must return a path
nvcc --version
```

If nvcc is missing, your CUDA installation is incomplete. Reinstall following Step 2 of the Linux section.

---

### Both Platforms

**`CUDA out of memory` on startup**

Reduce `gpu_layers` in `config.json`:
```json
"gpu_layers": 20
```

The remaining layers run on CPU. Performance decreases but Cardinal works.

---

**HTTP server not responding**

Check the port is not in use.

Windows:
```powershell
netstat -ano | findstr :8080
```

Linux:
```bash
ss -tlnp | grep 8080
```

If the port is taken, change it in `config.json`:
```json
"port": 8181
```

---

**Cardinal starts but immediately exits**

Check the log file for the error:

Windows: `build\bin\Release\` directory or `logs\cardinal.log`

Linux: `logs/cardinal.log`

The most common causes are a missing model file, a wrong grammar path, or a SQLite permission error on the data directory.

---

## Changing the API Key

The default API key `cardinal-dev-key-change-in-production` is intentionally obvious.
Change it before any networked deployment:

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

### Windows

```
D:\cardinal\
    build\
        bin\
            Release\
                cardinal.exe
                llama.dll
                ggml.dll
                ggml-cuda.dll
                libswipl.dll
    data\
        memory\
            rules.json
            knowledge.json
            episodes.db
        training_export.jsonl   (generated on first export)
    logs\
        cardinal.log
        episodic.log
    models\
        Qwen_Qwen3.5-4B-Q4_K_M.gguf
        Llama-3.2-1B-Instruct-Q4_K_M.gguf
    src\
        ...
    vendor\
        llama.cpp\
            build\
                bin\
                    Release\
                        llama.dll
                        ggml.dll
                        ...
    config.json
    CMakeLists.txt
    README.md
    INSTALL.md
    DOCUMENTATION.md
```

### Linux

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
    src/
        ...
    vendor/
        llama.cpp/
            build/
                lib/
                    libllama.so
                    libggml.so
                    libggml-cuda.so
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
- Query past reasoning: `GET /api/episodes?keyword=your+query`
- Export training data: `/export` or `POST /api/export`

---

*If something in this guide is wrong or out of date, the source of truth is always the CMakeLists.txt and config.json in the repository.*
