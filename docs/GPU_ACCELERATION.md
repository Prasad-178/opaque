# GPU Acceleration Analysis

## Executive Summary

We attempted to GPU-accelerate CKKS homomorphic encryption for privacy-preserving vector search by bridging Lattigo (Go, CPU) with HEonGPU (C++/CUDA, GPU) via gRPC. The effort produced valuable infrastructure and identified 5 critical bugs, but **the end-to-end recall is broken** due to an unresolved NTT domain conversion issue between the two libraries.

**Bottom line:** GPU path achieves **1.7x latency improvement** (144ms vs 244ms on SIFT 100K) but recall drops from 100% to ~14%, making it unusable for production. The recommended path forward is either CPU scale-out (simple, 1-2 weeks) or OpenFHE + FIDESlib (eliminates the bridge problem entirely, 4-6 weeks).

### Status at a Glance

| Component | Status | Notes |
|-----------|--------|-------|
| Profiling (bottleneck identification) | DONE | Galois rotation = 71-84% of HE time |
| HEonGPU per-operation benchmarks | DONE | 8-313x faster per op on Tesla T4 |
| Eval key bridge (format + NTT conversion) | DONE | Verified correct: rotation [10,20,30,40] -> [20,30,40,0] |
| GPU batch dot product (isolated) | DONE | 0.54ms vs 48ms CPU = 89x speedup, ZERO error |
| End-to-end pipeline (SIFT 100K) | DONE | 1.7x faster but recall broken (~14%) |
| NTT domain conversion debugging | BLOCKED | Go round-trip perfect, GPU computation wrong |

## End-to-End GPU Benchmark Results (SIFT 100K, Tesla T4)

These are measured end-to-end results on AWS g4dn.xlarge (Tesla T4, 4 vCPUs) running the full Opaque search pipeline through the GPU gRPC bridge.

### CPU Baseline (Lattigo, g4dn.xlarge 4 vCPUs)

| Config | Recall@1 | Recall@10 | Avg Latency |
|--------|----------|-----------|-------------|
| cpu-strict8 | 100% | 98.3% | 244ms |
| cpu-probe16 | 100% | 100% | 260ms |

### GPU Path (coefficient-domain approach, latest)

| Config | Recall@1 | Recall@10 | Avg Latency | Speedup |
|--------|----------|-----------|-------------|---------|
| gpu-strict8 | 13.3% | 14.3% | 144ms | **1.7x** |
| gpu-probe16 | 26.7% | 24.7% | 176ms | **1.5x** |

**Key finding:** The GPU path is 1.5-1.7x faster for latency, but recall is broken (13-27% vs 98-100%). The recall degradation is caused by NTT domain conversion issues between Lattigo and HEonGPU that corrupt ciphertext computation results on the GPU side.

## Bugs Found and Fixed

Over the course of this effort, we found and fixed 5 distinct bugs in the Lattigo-to-HEonGPU bridge. Each one was a subtle interop issue that produced silently wrong results.

| # | Bug | Symptom | Root Cause | Fix |
|---|-----|---------|------------|-----|
| 1 | Montgomery removal on non-Montgomery data | Results off by factor of R=2^64 mod Q | Lattigo v5 ciphertexts have `IsMontgomery=false`. We were dividing by R when data was already in standard form. | Check `IsMontgomery` flag; only remove Montgomery when true |
| 2 | Missing LogDimensions and IsBatched metadata | Decoder treats 8192-slot ciphertext as single value | Reconstructed ciphertexts had `Cols:0`, `IsBatched:false` | Set `LogDimensions={LogN-1, 1}` and `IsBatched=true` on reconstructed ciphertexts |
| 3 | HEonGPU ciphertext `scale_=0` | `multiply_plain` computes `output.scale = 0 * pt_scale = 0` | HEonGPU default constructor sets `scale_=0`; after `cudaMemcpy`, scale remains zero | Explicitly set `scale_` after data loading |
| 4 | Plaintext all zeros | GPU multiply produces zero output | Using `rlwe.NewPlaintext` (scale=0) instead of `hefloat.NewPlaintext` (sets scale from params) | Use `hefloat.NewPlaintext` which properly initializes scale |
| 5 | Proto Depth field semantics mismatch | Fresh ciphertexts treated as nearly-exhausted | Go sends Lattigo `Level()` (7) as Depth, but HEonGPU expects `depth=0` for fresh ciphertext | Send `depth=0` for fresh ciphertexts; HEonGPU depth = max_level - lattigo_level |

## Remaining Unsolved Issue: NTT Domain Conversion

The NTT domain conversion between Lattigo and HEonGPU has an unresolved mismatch that causes recall to drop to ~14% in end-to-end search.

### What works

- **Go-side round-trip is PERFECT**: Lattigo coefficients -> INTT_Lattigo -> NTT_HEonGPU -> INTT_HEonGPU -> NTT_Lattigo produces zero error across all 8192 slots.
- **HEonGPU native operations work**: `cudaMemcpy` bridge with HEonGPU-native data produces correct results.
- **Eval key bridge works**: Rotation of `[10,20,30,40]` -> `[20,30,40,0]` verified correct on GPU.
- **Isolated batch dot product works**: 0.54ms, ZERO error against expected values.

### What doesn't work

When Lattigo-converted ciphertext data goes through HEonGPU's `multiply_plain` + `rescale` + `rotate` pipeline in the full search path, results are incorrect. The recall drops to ~14%, which is approximately random performance.

### What we tried

| Approach | Result |
|----------|--------|
| Go NTT conversion (INTT_Lattigo -> NTT_HEonGPU) | Round-trip perfect in Go, but GPU produces wrong results |
| Bit-reverse permutation | No effect (both libraries use same ordering) |
| Coefficient-domain approach (skip Go NTT, let GPU apply HEonGPU's NTT) | GPU is faster (1.7x) but recall still ~14% |
| Native plaintext encoding (send float64 values, HEonGPU encodes) | Works for plaintexts but doesn't fix ciphertext issue |

### Hypothesis

There is likely a normalization factor difference between Lattigo's INTT output and HEonGPU's expected coefficient-domain format. Specifically, the N^{-1} factor that INTT applies may differ, or there may be a missing/extra scaling in one library's NTT convention.

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

#### Phase 3: Integration & Benchmarking — PARTIALLY DONE

**Completed:**
- `GPUHEProvider` implementing `HEProvider` interface (`pkg/crypto/gpu_provider.go`)
- GPU HE gRPC proto definition (`api/proto/gpuhe.proto`) with RegisterEvalKeys, BatchDotProduct, HealthCheck
- GPU server stub with Lattigo CPU backend (`cmd/gpu-server/main.go`)
- Config wiring: `GPUServerAddress` in opaque.Config, auto-creates GPUHEProvider
- Comprehensive tests: encrypt/decrypt, batch dot product, matches DirectHEProvider, pool mechanics

**Key breakthrough — eval key decomposition match:**
- `NewParametersGPU()` with `LogP=[61]` produces decomposition `d=8`, matching HEonGPU exactly
- Galois key sizes match: 2,359,296 uint64 = 18.0 MB per key (both libraries)
- Raw uint64 coefficient transfer for eval keys is now feasible
- See `docs/GPU_FORMAT_BRIDGE.md` for full analysis

**Eval key bridge: COMPLETE (verified on GPU)**
- NTT domain conversion working: Lattigo INTT → Montgomery removal → HEonGPU NTT
- Rotation verified correct on Tesla T4: `[10,20,30,40]` → `[20,30,40,0]` ✓
- GPU rotation speed: 66-630 us per rotation
- All conversion done in Go — no GPU-side NTT needed

**GPU batch dot product: VERIFIED (0.54ms on T4 vs 48ms CPU = 89x speedup)**
- Full HE pipeline on GPU: multiply + rescale + 7 rotations in 0.54ms
- All dot products verified with ZERO error against expected values
- C++ server BatchDotProduct handler fully implemented with cudaMemcpy data loading

**End-to-end pipeline: COMPLETED but recall broken**
- Full SIFT 100K pipeline runs through GPU gRPC bridge
- GPU is 1.7x faster for latency (144ms vs 244ms)
- Recall drops from 100% to ~14% due to NTT domain conversion issue
- See "End-to-End GPU Benchmark Results" and "Remaining Unsolved Issue" sections above

### Projected End-to-End Impact (Pre-Integration Estimates)

> **Note:** These projections were made before end-to-end integration. Actual measured results (see top of this document) show 1.7x latency improvement but with broken recall (~14%). The projections below assumed correct computation, which was not achieved.

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

PQ-M32 probe-8 is **measured** at 497ms on CPU (see BENCHMARKS.md). The breakdown:
- HE scoring (4 packs): ~270ms (measured, CPU)
- HE encrypt+decrypt: ~41ms (measured, CPU)
- PQ ADC + re-rank: ~186ms (measured, CPU — the PQ savings)

| Phase | CPU + PQ (measured) | With GPU HE + PQ (projected) | Savings |
|-------|---------------------|------------------------------|---------|
| HE scoring (4 packs) | ~270ms | ~20ms × 4 = ~80ms | -190ms |
| HE encrypt+decrypt | ~41ms | ~41ms | 0 |
| PQ ADC + re-rank | ~186ms | ~186ms | 0 |
| **Total** | **497ms (measured)** | **~307ms (projected)** | **1.6x** |

**Key finding:** PQ alone already delivers the massive win (19.2x vs standard). GPU HE would provide an additional ~1.6x on top of PQ by reducing the HE scoring phase. The combined projection is ~307ms — but PQ at 497ms is already sub-second and may be sufficient without GPU complexity.

## If Picking Up Again -- Start Here

If someone (including future us) wants to resume the GPU acceleration work, here is the shortest path to progress.

### 1. The coefficient-domain approach is closest to working

The GPU is already 1.7x faster for latency. The only problem is recall. All infrastructure is built and working.

### 2. Write a C++ diagnostic for NTT normalization

Create a test in `deploy/gpu/gpu-he-server/` that:
1. Takes `[1, 0, 0, ..., 0]` in coefficient domain
2. Applies HEonGPU's forward NTT
3. Applies HEonGPU's inverse NTT
4. Checks if you get `[1, 0, 0, ..., 0]` back

This verifies whether HEonGPU's NTT includes the N^{-1} normalization factor in INTT (standard convention) or distributes it differently.

### 3. Compare Lattigo's INTT output with HEonGPU's expectations

For a known NTT input (e.g., the NTT of `[1, 0, ..., 0]`):
- Compute Lattigo's INTT output
- Compute HEonGPU's INTT output
- If they differ by a factor of N (or N^{-1} mod Q), that confirms the normalization mismatch

### 4. Key files

| File | Purpose |
|------|---------|
| `go/pkg/crypto/ntt_convert.go` | Go-side NTT domain conversion (Lattigo <-> HEonGPU) |
| `go/pkg/crypto/ciphertext_convert.go` | Ciphertext format conversion for the gRPC bridge |
| `deploy/gpu/gpu-he-server/main.cpp` | C++ GPU server (gRPC handlers, HEonGPU integration) |
| `go/pkg/crypto/bridge_diagnostic_test.go` | Go diagnostic tests (12 tests, all pass) |
| `deploy/gpu/gpu-he-server/bridge_diagnostic.cpp` | C++ diagnostic tests (all pass) |

### 5. Key tests

- **Go diagnostics**: `go test -v -run TestBridge ./pkg/crypto/` -- 12 tests covering NTT round-trip, Montgomery handling, coefficient ordering, metadata reconstruction. All pass.
- **C++ diagnostics**: `bridge_diagnostic.cpp` -- verifies HEonGPU-side data loading, NTT operations, key reconstruction. All pass.
- **End-to-end GPU**: requires running the GPU server on a CUDA instance and the Go client pointing at it via gRPC.

## Future Alternatives

### Option A: OpenFHE + FIDESlib (Recommended for GPU path)

| Attribute | Value |
|-----------|-------|
| Feasibility | 6/10 |
| Effort | 4-6 weeks |
| Key advantage | Eliminates NTT bridge entirely |

[FIDESlib](https://github.com/CAPS-UMU/FIDESlib) is a purpose-built GPU backend for OpenFHE with full CKKS support, achieving 70-228x speedup over CPU OpenFHE. Since both client and server would use OpenFHE's internal representation, there is no NTT domain conversion problem -- the exact issue that blocked our HEonGPU bridge.

**What it requires:**
- Rewrite the Go crypto layer (`pkg/crypto/`) to call OpenFHE via C bindings or a gRPC C++ service
- Client-side encryption/decryption moves to C++/OpenFHE (could be wrapped as a Go library via cgo)
- Server-side computation uses FIDESlib's GPU backend
- Privacy model is identical: same client/server split, secret key stays on client

**Why it solves our problem:** The NTT domain mismatch exists because Lattigo and HEonGPU are independent implementations with different NTT conventions. Using a single library (OpenFHE) on both sides eliminates this class of bug entirely.

### Option B: CPU Scale-Out (Recommended for production)

| Attribute | Value |
|-----------|-------|
| Feasibility | 9/10 |
| Effort | 1-2 weeks |
| Key advantage | No new code, just configuration |

Deploy the existing Go/Lattigo gRPC server on a high-core-count instance (e.g., AWS c7i.48xlarge with 96 vCPUs). The gRPC server already supports parallel engine pools -- scaling to 96 cores gives ~10x throughput improvement with zero code changes.

**Combined with PQ (19x on GIST)**, this may be sufficient for production:
- GIST 960-dim: 497ms with PQ on current hardware, potentially ~50ms with 10x parallel throughput
- SIFT 128-dim: 127ms with PQ-probe16, already fast enough for most use cases

**Cost:** ~$3/hr spot for c7i.48xlarge. No GPU complexity, no format bridges, no CUDA dependencies.

### Option C: Continue HEonGPU Bridge (If GPU latency is critical)

| Attribute | Value |
|-----------|-------|
| Feasibility | 4/10 |
| Effort | 1-3 weeks (if NTT issue is solved) |
| Key advantage | Infrastructure already built |

The coefficient-domain approach already achieves 1.7x latency improvement. One remaining normalization issue needs to be resolved. All diagnostic infrastructure exists (`bridge_diagnostic_test.go`, `bridge_diagnostic.cpp`), and the "If Picking Up Again" section above describes the exact next steps.

**Risk:** The NTT normalization issue may be deeper than expected (e.g., HEonGPU may use a non-standard NTT convention internally that isn't exposed in its API).

### Comparison

| Criterion | OpenFHE + FIDESlib | CPU Scale-Out | Continue HEonGPU |
|-----------|-------------------|---------------|-------------------|
| Latency improvement | 70-228x on HE ops | ~10x throughput | 1.7x measured |
| Recall risk | None (same library) | None (proven path) | High (unresolved bug) |
| New code required | Significant (crypto rewrite) | None | Minimal |
| Production readiness | Months | Days | Weeks (if bug fixed) |
| Cost | GPU instance | High-CPU instance (~$3/hr) | GPU instance |

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

### Per-Operation Comparison (poly_modulus_degree=16384, confirmed across 2 runs)

| Operation | HEonGPU (T4 GPU) | Lattigo (M4 CPU) | Speedup |
|-----------|-------------------|-------------------|---------|
| Encrypt | 1.46ms | 7.14ms | **4.9x** |
| Plain Multiply | 0.026ms | 0.70ms | **27x** |
| Rescale | 0.107ms | 1.18ms | **11x** |
| **Rotate (Galois)** | **0.759ms** | **6.55ms** | **8.6x** |
| Decrypt | 0.030ms | 9.38ms | **313x** |
| Add | 0.021ms | 0.45ms | **21x** |
| Relinearize | 0.669ms | N/A (included in MulNew) | — |

**Cross-degree scaling (HEonGPU, all measured on T4):**

| Degree | Rotate | Multiply | Rescale | Encrypt | Notes |
|--------|--------|----------|---------|---------|-------|
| 4096 | 0.085ms | 0.009ms | 0.043ms | 1.61ms | Small params |
| 8192 | 0.142ms | 0.009ms | 0.045ms | 1.27ms | |
| **16384** | **0.759ms** | **0.041ms** | **0.107ms** | **1.46ms** | **Our params (LogN=14)** |
| 32768 | 4.27ms | 0.138ms | 0.453ms | 2.48ms | Large params |

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

> **Note:** The numbers below were projections made before end-to-end integration was completed. The end-to-end pipeline was subsequently built and tested (see "End-to-End GPU Benchmark Results" at the top). Actual results showed 1.7x latency improvement but with broken recall (~14%) due to NTT domain conversion issues. The per-operation speedups are real, but the full pipeline does not yet produce correct search results.

| Dataset | Standard CPU | PQ CPU (measured) | GPU HE + PQ (projected) |
|---------|-------------|-------------------|-------------------------|
| SIFT 100K probe-16 | 160ms | 127ms (1.26x) | ~100ms (1.6x projected) |
| **GIST 100K probe-8** | **9.53s** | **497ms (19.2x measured)** | **~307ms (projected)** |
| GIST 100K probe-16 | 13.0s | 855ms (15.2x measured) | ~665ms (projected) |

**Key insight:** PQ alone delivers transformative speedups on high-dimensional data — **19.2x measured** on GIST 960-dim. GPU HE would add an additional ~1.6x on top. The earlier 350ms projection was optimistic; real PQ measurement at 497ms is close and already sub-second. GPU integration would primarily help if sub-300ms latency is required.

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
