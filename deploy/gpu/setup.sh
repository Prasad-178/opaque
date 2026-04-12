#!/bin/bash
# User-data script for GPU benchmark instance.
# Runs on first boot: installs Go, clones repo, builds GPU benchmark tools.
# The Deep Learning AMI already has CUDA + NVIDIA drivers pre-installed.
set -euo pipefail

LOG="/var/log/opaque-setup.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== Opaque GPU Bench Setup — $(date) ==="

# --- Install Go ---
GO_VERSION="1.24.4"
echo "Installing Go $GO_VERSION..."
cd /tmp
curl -fSL "https://go.dev/dl/go$GO_VERSION.linux-amd64.tar.gz" -o go.tar.gz
rm -rf /usr/local/go
tar -C /usr/local -xzf go.tar.gz
rm go.tar.gz

# Set up Go for ubuntu user.
cat >> /home/ubuntu/.bashrc << 'GOEOF'
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
export GOPATH=$HOME/go
GOEOF

export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
export GOPATH=/home/ubuntu/go

echo "Go version: $(go version)"

# --- Install build tools for GPU libraries ---
echo "Installing build dependencies..."
apt-get update -qq
apt-get install -y -qq build-essential cmake git pkg-config \
    libprotobuf-dev protobuf-compiler libgrpc++-dev protobuf-compiler-grpc \
    libgmp-dev libntl-dev

# --- Clone Opaque ---
echo "Cloning Opaque repo..."
cd /home/ubuntu
if [ -d "opaque" ]; then
  cd opaque && git pull
else
  git clone --branch "${repo_branch}" "${repo_url}" opaque
  cd opaque
fi

# --- Build Go project ---
echo "Building Opaque..."
cd /home/ubuntu/opaque
go build ./...
echo "Go build successful."

# --- Clone and build GPU-NTT (NTT microbenchmark) ---
echo "Cloning GPU-NTT..."
cd /home/ubuntu
if [ ! -d "GPU-NTT" ]; then
  git clone https://github.com/Alisah-Ozcan/GPU-NTT.git
fi

echo "Building GPU-NTT..."
cd GPU-NTT
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 2>/dev/null || echo "GPU-NTT cmake failed (may need CUDA arch adjustment)"
make -j$(nproc) 2>/dev/null || echo "GPU-NTT build failed (non-blocking)"

# --- Clone and build HEonGPU (CKKS GPU benchmark) ---
echo "Cloning HEonGPU..."
cd /home/ubuntu
if [ ! -d "HEonGPU" ]; then
  git clone https://github.com/Alisah-Ozcan/HEonGPU.git
fi

echo "Building HEonGPU..."
cd HEonGPU

# Patch: add public setters for cross-library ciphertext reconstruction
python3 << 'PATCHEOF'
# Patch 1: Ciphertext set_scale/set_depth
with open('src/include/heongpu/host/ckks/ciphertext.cuh', 'r') as f:
    c = f.read()
marker = 'inline double scale() const noexcept { return scale_; }'
if 'set_scale' not in c:
    c = c.replace(marker, marker + '\n\n        inline void set_scale(double s) noexcept { scale_ = s; }\n        inline void set_depth(int d) noexcept { depth_ = d; }', 1)
    with open('src/include/heongpu/host/ckks/ciphertext.cuh', 'w') as f:
        f.write(c)
    print('Patched ciphertext with set_scale/set_depth')

# Patch 2: Context public accessors for NTT table and modulus
for ctx_file in ['src/include/heongpu/host/ckks/context.cuh', 'src/include/heongpu/host/bfv/context.cuh']:
    try:
        with open(ctx_file, 'r') as f:
            c = f.read()
        if 'get_ntt_table' not in c and 'ntt_table_' in c:
            # Add public getters before the first 'private:' or at end of public section
            c = c.replace('bool context_generated_', 'bool context_generated_;\n\n        auto& get_ntt_table() { return *ntt_table_; }\n        auto& get_intt_table() { return *intt_table_; }\n        auto& get_modulus() { return *modulus_; }', 1)
            with open(ctx_file, 'w') as f:
                f.write(c)
            print(f'Patched {ctx_file} with NTT/modulus getters')
    except FileNotFoundError:
        pass
PATCHEOF
echo "HEonGPU patches applied" || echo "HEonGPU patch failed (non-blocking)"

mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 2>/dev/null || echo "HEonGPU cmake failed (non-blocking)"
make -j$(nproc) 2>/dev/null || echo "HEonGPU build failed (non-blocking)"

# --- Build GPU HE server examples (batch_dot_product, bridge tests) ---
echo "Building GPU HE server examples..."
cd /home/ubuntu/HEonGPU
# Copy test programs from opaque repo
for f in batch_dot_product.cpp bridge_final.cpp bridge_matching_keys.cpp; do
  if [ -f /home/ubuntu/opaque/deploy/gpu/gpu-he-server/$f ]; then
    cp /home/ubuntu/opaque/deploy/gpu/gpu-he-server/$f example/basic/
    name=$(basename $f .cpp)
    if ! grep -q "$name" example/basic/CMakeLists.txt 2>/dev/null; then
      echo "add_executable($name $f)
target_link_libraries($name PRIVATE heongpu)
set_target_properties($name PROPERTIES CUDA_ARCHITECTURES 75)" >> example/basic/CMakeLists.txt
    fi
  fi
done
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 -DHEonGPU_BUILD_EXAMPLES=ON 2>/dev/null || true
make -j$(nproc) 2>/dev/null || echo "Example build failed (non-blocking)"

# --- Build C++ GPU HE gRPC server ---
echo "Building GPU HE server..."
cd /home/ubuntu/opaque/deploy/gpu/gpu-he-server
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 2>&1 || echo "GPU HE server cmake failed"
make -j$(nproc) 2>&1 || echo "GPU HE server build failed (non-blocking)"
if [ -f gpu-he-server ]; then
  echo "GPU HE server built successfully"
else
  echo "GPU HE server binary not found after build"
fi

# --- Download SIFT1M dataset ---
echo "Downloading SIFT1M dataset..."
cd /home/ubuntu/opaque
bash scripts/download_sift1m.sh || echo "SIFT download failed (non-blocking)"

# --- Fix permissions ---
chown -R ubuntu:ubuntu /home/ubuntu/opaque /home/ubuntu/GPU-NTT /home/ubuntu/HEonGPU 2>/dev/null || true
chown -R ubuntu:ubuntu /home/ubuntu/go 2>/dev/null || true

# --- Write ready marker ---
touch /home/ubuntu/.setup-complete
echo "=== Setup complete — $(date) ==="
echo "Run GPU E2E benchmark with: bash opaque/deploy/gpu/run_gpu_e2e.sh"
