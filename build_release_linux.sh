#!/bin/bash
# build_release_linux.sh
# Run this on Ubuntu/Debian/Fedora

set -e

VERSION=${1:-"v1.0.0"}

echo "========================================"
echo "Building Cardinal Release Package"
echo "========================================"

# Install dependencies
echo ""
echo "[1/5] Installing dependencies..."
if command -v apt &> /dev/null; then
    sudo apt update
    sudo apt install -y build-essential cmake git libssl-dev libsqlite3-dev swi-prolog
elif command -v dnf &> /dev/null; then
    sudo dnf install -y gcc-c++ cmake git openssl-devel sqlite-devel swipl
else
    echo "Unsupported package manager. Install dependencies manually."
    exit 1
fi

# Create directories
mkdir -p vendor models logs data/memory

# Build llama.cpp
echo ""
echo "[2/5] Building llama.cpp..."
cd vendor
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp
git checkout b8660
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cd ../..

# Build Cardinal
echo ""
echo "[3/5] Building Cardinal..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

# Create release package
echo ""
echo "[4/5] Creating release package..."
RELEASE_NAME="Cardinal-${VERSION}-linux-x64"
RELEASE_DIR="release-${RELEASE_NAME}"

rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR/bin"
mkdir -p "$RELEASE_DIR/data/memory"
mkdir -p "$RELEASE_DIR/logs"
mkdir -p "$RELEASE_DIR/models"
mkdir -p "$RELEASE_DIR/src/prompts"
mkdir -p "$RELEASE_DIR/src/verifier"

# Create models placeholder
cat > "$RELEASE_DIR/models/PLACE_MODEL_HERE.txt" << 'EOF'
Place your GGUF model file here.

Example:
  models/Qwen_Qwen3.5-4B-Q4_K_M.gguf

Then update config.json to point to: models/your-model.gguf
EOF

cp build/bin/cardinal "$RELEASE_DIR/bin/"
cp config.json "$RELEASE_DIR/"
cp README.md "$RELEASE_DIR/"
cp DOCUMENTATION.md "$RELEASE_DIR/"
cp LICENSE "$RELEASE_DIR/"
cp src/verifier/cardinal_kb.pl "$RELEASE_DIR/src/verifier/"
cp src/prompts/feeling_schema.gbnf "$RELEASE_DIR/src/prompts/"

# Copy llama.cpp libraries
if [ -d "vendor/llama.cpp/build/lib" ]; then
    cp vendor/llama.cpp/build/lib/*.so* "$RELEASE_DIR/bin/" 2>/dev/null || true
fi

# Create run.sh for Linux users
echo ""
echo "[5/5] Creating run.sh..."
cat > "$RELEASE_DIR/run.sh" << 'EOF'
#!/bin/bash
# Cardinal AGI Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/bin:$LD_LIBRARY_PATH"

echo "========================================"
echo "Cardinal AGI - Starting..."
echo "========================================"
echo ""
echo "API Server: http://127.0.0.1:8080"
echo "Dashboard: http://127.0.0.1:8080"
echo ""
echo "Press Ctrl+C to stop"
echo "========================================"
echo ""

"$SCRIPT_DIR/bin/cardinal" --serve
EOF

chmod +x "$RELEASE_DIR/run.sh"

# Create README for the release package
cat > "$RELEASE_DIR/README_RELEASE.txt" << 'EOF'
# Cardinal AGI - Quick Start

## Linux

1. Place your model file in the `models/` folder
2. Edit `config.json` to set the model path
3. Run `./run.sh` to start

## Default API Key

The default API key is in `config.json`. Change it before exposing to network.

## Requirements

- Linux (Ubuntu 20.04+, Fedora 37+)
- NVIDIA GPU recommended (runs on CPU without one)
- SWI-Prolog, OpenSSL, SQLite3 (installed automatically by build script)

## Need Help?

See README.md for full documentation.
EOF

# Create tar.gz
tar -czf "${RELEASE_NAME}.tar.gz" -C "$RELEASE_DIR" .

rm -rf "$RELEASE_DIR"

echo ""
echo "========================================"
echo "Release created: $RELEASE_NAME.tar.gz"
echo "========================================"
echo ""
echo "The tar.gz contains:"
echo "  - bin/cardinal"
echo "  - run.sh (run this to start)"
echo "  - models/ (place your .gguf here)"
echo "  - config.json (edit settings)"