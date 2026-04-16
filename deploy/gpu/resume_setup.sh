#!/bin/bash
# resume_setup.sh — continue from an extracted opaque bundle.
# Run on T4 instance as ubuntu (needs sudo for some steps).

set -euo pipefail
export HOME=/home/ubuntu

echo "=== Extracting opaque bundle ==="
rm -rf /home/ubuntu/opaque
mkdir -p /home/ubuntu/opaque
tar -xzf /tmp/opaque-bundle.tar.gz -C /home/ubuntu/opaque
ls /home/ubuntu/opaque | head

echo "=== Building Go project (skip, not needed for encode_compare) ==="
# go mod download — may need, but encode_compare is pure C++
# Skip to save time; will only matter for Go dump which we already have.

echo "=== Clone + build GPU-NTT ==="
if [ ! -d /home/ubuntu/GPU-NTT ]; then
  git clone https://github.com/Alisah-Ozcan/GPU-NTT.git /home/ubuntu/GPU-NTT
fi
cd /home/ubuntu/GPU-NTT
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 > /dev/null
make -j$(nproc) > /dev/null 2>&1 || echo "GPU-NTT build warned"

echo "=== Clone HEonGPU ==="
if [ ! -d /home/ubuntu/HEonGPU ]; then
  git clone https://github.com/Alisah-Ozcan/HEonGPU.git /home/ubuntu/HEonGPU
fi

echo "=== Apply HEonGPU patches ==="
cd /home/ubuntu/HEonGPU
python3 <<'PATCHEOF'
# Patch 1: Add set_scale/set_depth setters to Ciphertext
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
            c = c.replace('bool context_generated_', 'bool context_generated_;\n\n        auto& get_ntt_table() { return *ntt_table_; }\n        auto& get_intt_table() { return *intt_table_; }\n        auto& get_modulus() { return *modulus_; }', 1)
            with open(ctx_file, 'w') as f:
                f.write(c)
            print(f'Patched {ctx_file} with NTT/modulus getters')
    except FileNotFoundError:
        pass
PATCHEOF

echo "=== Build HEonGPU ==="
cd /home/ubuntu/HEonGPU
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 > /tmp/heongpu_cmake.log 2>&1
make -j$(nproc) > /tmp/heongpu_make.log 2>&1
echo "HEonGPU libs:"
ls src/*.a 2>/dev/null || ls src/ | head

echo "=== Build encode_compare via gpu-he-server CMakeLists ==="
cd /home/ubuntu/opaque/deploy/gpu/gpu-he-server
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 > /tmp/enc_cmake.log 2>&1 || true
tail -5 /tmp/enc_cmake.log
make encode_compare -j$(nproc) > /tmp/enc_make.log 2>&1 || true
tail -10 /tmp/enc_make.log
ls -la encode_compare 2>&1 || echo "encode_compare NOT BUILT"
