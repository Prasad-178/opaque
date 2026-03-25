# GPU Acceleration Analysis

## Executive Summary

Profiling reveals that **Galois rotation (key-switching)** — not NTT — is the dominant bottleneck in HE operations, consuming 71-84% of total HE time. GPU acceleration of key-switching could yield **3.2-5.1x speedup** on HE operations, with the impact scaling with vector dimensionality.

**Critical finding:** At 960-dim (GIST), HE takes 311ms per query (86.8% is batch computation). GPU could reduce this to ~61ms. At 128-dim (SIFT), HE takes 65ms (74.4% is batch computation). GPU could reduce this to ~20ms.

## Profiling Results

All measurements on Apple M4 Pro (10 CPUs), Lattigo v5.0.7, CKKS LogN=14.

### SIFT 128-dim (64 centroids, 1 CKKS pack)

| Phase | Avg Time | % of HE | GPU Target |
|-------|----------|---------|------------|
| Query packing | 31µs | 0.0% | No |
| HE encrypt | 7.1ms | 11.0% | Maybe |
| **HE batch total** | **48.2ms** | **74.4%** | **YES** |
| ├─ Multiply (NTT) | 0.7ms | 1.1% | YES (20-50x) |
| ├─ Rescale | 1.2ms | 1.8% | YES (5-10x) |
| **├─ Rotate (Galois)** | **45.9ms** | **70.8%** | **YES (10-20x)** |
| └─ Add | 0.5ms | 0.7% | Marginal |
| HE decrypt | 9.4ms | 14.5% | Maybe |
| **TOTAL HE** | **64.7ms** | **100%** | |

**GPU projection:** 64.7ms → **20.0ms (3.2x speedup)**

### GIST 960-dim (32 centroids, 4 CKKS packs)

| Phase | Avg Time | % of HE | GPU Target |
|-------|----------|---------|------------|
| Query packing | 6µs | 0.0% | No |
| HE encrypt | 7.8ms | 2.5% | Maybe |
| **HE batch (4 packs)** | **270.0ms** | **86.8%** | **YES** |
| ├─ Multiply (NTT) | 1.4ms | 0.5% | YES (20-50x) |
| ├─ Rescale | 4.8ms | 1.6% | YES (5-10x) |
| **├─ Rotate (Galois)** | **260.9ms** | **83.8%** | **YES (10-20x)** |
| └─ Add | 2.9ms | 0.9% | Marginal |
| HE decrypt | 33.3ms | 10.7% | Maybe |
| **TOTAL HE** | **311.1ms** | **100%** | |

**GPU projection:** 311.1ms → **60.6ms (5.1x speedup)**

## Why Galois Rotation Dominates (Not NTT)

The initial assumption was that NTT (Number Theoretic Transform) would be the bottleneck, as it is for raw polynomial multiplication. However, profiling shows:

1. **Multiply (NTT)** is only 0.5-1.1% of HE time. Lattigo's optimized NTT on ARM64/NEON is already fast (~0.7ms for degree-16384 polynomials).

2. **Galois rotation** is 71-84% of HE time. Each rotation involves a **key-switching** operation:
   - Decompose ciphertext into digits (gadget decomposition)
   - Multiply each digit by a Galois key element (large matrix-vector product in the ring)
   - Apply the automorphism (permutation)
   - For 128-dim: 7 rotations × ~6.5ms each = ~45ms
   - For 960-dim: 10 rotations × 4 packs × ~6.5ms each = ~260ms

3. **Key-switching is the GPU-acceleratable bottleneck.** It involves multiple polynomial multiplications (each using NTT internally) per rotation. Published GPU key-switching benchmarks show **10-20x speedup** (see references below).

## GPU Acceleration Strategy

### Recommended Approach: Hybrid Client-Server

```
Client (Go/Lattigo, your Mac):
  → Key generation, encryption, decryption
  → Unchanged code

Server (C++/CUDA, cloud GPU):
  → HE centroid scoring (GPU-accelerated)
  → Key-switching, NTT, rescale all on GPU
  → Communicates via existing gRPC layer

Protocol: Serialize ciphertext → gRPC → GPU compute → gRPC → decrypt locally
```

### Implementation Phases

#### Phase 1: GPU HE Service (CUDA, 4-6 weeks)

Build a lightweight gRPC service that wraps a GPU-HE library for server-side CKKS computation.

**Candidate GPU libraries (ranked):**

| Library | Key-Switching GPU | CKKS Support | Interop | Effort |
|---------|-------------------|--------------|---------|--------|
| **HEonGPU** | YES, 380x over SEAL | Full CKKS | Own format | Medium |
| **FIDESlib** | YES, 70-228x over OpenFHE | Full CKKS | OpenFHE compatible | Medium |
| **ICICLE** | NTT only (no KS yet) | Partial | Go bindings exist | Lower |

**Recommended: HEonGPU** — broadest operation coverage, all computation stays on GPU (no CPU-GPU transfers), proven CKKS performance.

**Required work:**
1. C++ wrapper with `extern "C"` facade for key-switching + batch dot product
2. Go client via cgo OR gRPC service binary
3. Parameter alignment: ensure Lattigo and HEonGPU use compatible CKKS parameters (ring degree, modulus chain, scale)
4. Ciphertext serialization bridge between Lattigo and GPU library formats

#### Phase 2: Apple Metal Path (M-series native, 3-4 weeks)

For development on Apple Silicon without a CUDA GPU:

1. Write Metal compute shaders for key-switching (the dominant operation)
2. Call from Go via cgo + Objective-C bridge
3. M4 unified memory = zero-copy buffer sharing
4. This would be novel (no existing Metal HE implementations)

**Note:** This is research-grade work and would be publishable.

#### Phase 3: Integration & Benchmarking (2 weeks)

1. Add `GPUHEProvider` implementing the existing `HEProvider` interface
2. Benchmark real end-to-end speedup on SIFT 100K and GIST 100K
3. Measure GPU memory usage, warm-up time, throughput under concurrent queries

### Projected End-to-End Impact

**SIFT 100K (128-dim, probe-16, current: ~160ms):**

| Phase | CPU (current) | With GPU HE | Savings |
|-------|---------------|-------------|---------|
| HE scoring | ~48ms | ~5ms | -43ms |
| HE encrypt+decrypt | ~16ms | ~16ms | 0 |
| AES + local scoring | ~96ms | ~96ms | 0 |
| **Total** | **~160ms** | **~117ms** | **1.4x** |

**GIST 100K (960-dim, probe-8, current: ~2.6s):**

| Phase | CPU (current) | With GPU HE | Savings |
|-------|---------------|-------------|---------|
| HE scoring (4 packs) | ~270ms × 4 | ~20ms × 4 | -1000ms |
| HE encrypt+decrypt | ~41ms | ~41ms | 0 |
| AES + local scoring | ~1500ms | ~1500ms | 0 |
| **Total** | **~2.6s** | **~1.6s** | **1.6x** |

**GIST 100K (960-dim, probe-8, with GPU HE + PQ):**

| Phase | CPU + PQ | With GPU HE + PQ | Savings |
|-------|----------|-------------------|---------|
| HE scoring (4 packs) | ~270ms × 4 | ~20ms × 4 | -1000ms |
| HE encrypt+decrypt | ~41ms | ~41ms | 0 |
| PQ scoring + re-rank | ~300ms | ~300ms | 0 |
| **Total** | **~1.4s** | **~420ms** | **3.3x** |

**Projected insight:** GPU HE + PQ combined would be the strongest optimization for high-dimensional vectors. GPU reduces HE time, PQ reduces local scoring time. Together they could cut GIST 960-dim from 2.6s to ~420ms. These are projections — actual end-to-end GPU pipeline has not been built yet.

## Hardware Requirements

| Target | GPU | Memory | Cost |
|--------|-----|--------|------|
| Development | Apple M4 GPU (Metal) | Unified 32GB+ | $0 (existing) |
| Benchmarking | RTX 4090 (CUDA) | 24GB VRAM | ~$1,600 |
| Production | A100/H100 (CUDA) | 40-80GB HBM | Cloud instances |
| Budget option | RTX 3060 (CUDA) | 12GB VRAM | ~$300 |

CKKS with LogN=14 requires ~200MB GPU memory for parameters + keys + workspace. Any modern GPU has sufficient memory.

## Real GPU Benchmark Results (Tesla T4, g4dn.xlarge)

Benchmarked HEonGPU (CKKS) on AWS g4dn.xlarge (Tesla T4, 16GB VRAM, CUDA 12.9) against Lattigo CPU on Apple M4 Pro. Both use `poly_modulus_degree=16384` (LogN=14).

### Per-Operation Comparison

| Operation | HEonGPU (T4 GPU) | Lattigo (M4 CPU) | Speedup |
|-----------|-------------------|-------------------|---------|
| Encrypt | 1.47ms | 7.14ms | **4.9x** |
| Plain Multiply | 0.027ms | 0.70ms | **26x** |
| Rescale | 0.107ms | 1.18ms | **11x** |
| **Rotate (Galois)** | **0.761ms** | **6.55ms** | **8.6x** |
| Decrypt | 0.030ms | 9.38ms | **313x** |
| Add | 0.020ms | 0.45ms | **22x** |

### Search Pipeline Projection

**SIFT 128-dim (1 pack, 7 rotations):**

| Component | CPU (Lattigo) | GPU (HEonGPU) | Speedup |
|-----------|---------------|---------------|---------|
| Batch HE scoring | 48.2ms | ~6.4ms | **7.5x** |
| + Encrypt + Decrypt | 16.5ms | 1.5ms | 11x |
| **Total HE** | **64.7ms** | **~8ms** | **8x** |

**GIST 960-dim (4 packs, 10 rotations each):**

| Component | CPU (Lattigo) | GPU (HEonGPU) | Speedup |
|-----------|---------------|---------------|---------|
| Batch HE scoring | 270ms | ~32ms | **8.5x** |
| + Encrypt + Decrypt | 41ms | ~1.5ms | 27x |
| **Total HE** | **311ms** | **~33ms** | **9.4x** |

### End-to-End Impact (Projected, Not Measured)

> **Note:** The numbers below are projections based on combining independently measured components (Lattigo CPU profiling + HEonGPU GPU benchmarks + PQ synthetic benchmarks). No end-to-end GPU pipeline has been built yet — this requires Go ↔ HEonGPU integration. Actual results may differ due to data transfer overhead, memory layout differences, and parameter compatibility.

| Dataset | Current (CPU) | With GPU HE | With GPU HE + PQ |
|---------|---------------|-------------|-------------------|
| SIFT 100K probe-16 | 160ms | ~120ms (1.3x) | ~100ms (1.6x) |
| **GIST 100K probe-8** | **2.6s** | **~700ms (3.7x)** | **~350ms (7.4x)** |
| GIST 100K probe-16 | 6.2s | ~1.5s (4.1x) | ~800ms (7.8x) |

**Projected insight:** GPU HE alone could give ~3.7x on high-dimensional GIST. Combined with PQ, the projected improvement reaches ~7.4x — potentially cutting GIST 100K from 2.6s to ~350ms. These are estimates pending actual GPU integration.

## References

- [HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU) — 380x over SEAL, CKKS+BFV+TFHE on GPU
- [FIDESlib](https://github.com/CAPS-UMU/FIDESlib) — 70-228x over OpenFHE, first open-source GPU CKKS bootstrapping
- [GPU Key-Switching (2025)](https://eprint.iacr.org/2025/124) — Up to 181x speedup on key-switching
- [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT) — State-of-the-art NTT on multi-GPU
- [MoMA NTT (CGO 2025)](http://users.ece.cmu.edu/~franzf/papers/2025_CGO_MoMA.pdf) — Near-ASIC NTT on consumer GPUs
- [ICICLE v4](https://github.com/ingonyama-zk/icicle) — Lattice crypto primitives with Go bindings, Apple Silicon planned
- [Go + Metal GPU](https://towardsdatascience.com/programming-apple-gpus-through-go-and-metal-shading-language-a0e7a60a3dba/) — Calling Metal compute shaders from Go

## GPU Benchmark Infrastructure

Ephemeral Terraform infrastructure for running HE benchmarks on real CUDA hardware. See `deploy/gpu/` for full source.

**Quick start:**

```bash
cd deploy/gpu
terraform init
terraform apply -var="enabled=true"    # Spot g4dn.xlarge, ~$0.16/hr
bash run_benchmarks.sh                  # SSH in, run all benchmarks, save results
terraform destroy -var="enabled=true"   # Tear down, $0 when off
```

**What it provisions:**
- g4dn.xlarge spot instance (NVIDIA T4, 16GB VRAM)
- AWS Deep Learning AMI (CUDA + NVIDIA drivers pre-installed)
- Auto-installs Go, clones Opaque, builds GPU-NTT + HEonGPU
- SSH key auto-generated, results saved to `deploy/gpu/results/`

**Toggle:** `enabled=false` (default) means no resources exist. Set `enabled=true` only when running benchmarks. All resources destroyed on `terraform destroy`.

**Cost:** ~$0.16/hr spot ($0.53/hr on-demand). Typical benchmark run (30 min) costs ~$0.08.

**AWS profile:** Uses `--profile personal` to avoid touching production accounts.

## Reproducing

```bash
# Local profiling (no GPU needed):

# SIFT 128-dim HE sub-phase profiling (~30s)
go test -tags sift1m -v -run TestGPU_ProfilingSIFT100K ./test/ -timeout 30m

# GIST 960-dim HE sub-phase profiling (~12s)
go test -tags gist -v -run TestGPU_ProfilingGIST ./test/ -timeout 30m

# GPU benchmarks (requires AWS):

cd deploy/gpu
terraform init && terraform apply -var="enabled=true"
bash run_benchmarks.sh
terraform destroy -var="enabled=true"
```
