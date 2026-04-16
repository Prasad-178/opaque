# GPU Acceleration Analysis

## Executive Summary

We attempted to GPU-accelerate CKKS homomorphic encryption for privacy-preserving vector search by bridging Lattigo (Go, CPU) with HEonGPU (C++/CUDA, GPU) via gRPC. Three rounds of deep debugging isolated every bridge bug we originally suspected, including a real NTT-batch-shape kernel bug. **The remaining failure is a gRPC-server statefulness issue inside HEonGPU**, not the bridge itself.

**Bottom line:** Bridge math is **provably correct** in-process (20/20 layered GPU tests pass with error on the CKKS noise floor, ~1e-8). The first gRPC call after a cold server decrypts correctly. Every subsequent call returns ~10^87 garbage — deterministic state leak between requests. Recall on SIFT 100K still ~10–20% through the gRPC path. Recommended path forward: CPU scale-out (simple, 1–2 weeks) or OpenFHE + FIDESlib (replaces the entire GPU library, 4–6 weeks).

### Status at a Glance

| Component | Status | Notes |
|-----------|--------|-------|
| Profiling (bottleneck identification) | DONE | Galois rotation = 71-84% of HE time |
| HEonGPU per-operation benchmarks | DONE | 8-313x faster per op on Tesla T4 |
| Encoder / plaintext bridge | DONE | Coefficient polynomials byte-identical between Lattigo and HEonGPU; canonical embedding verified compatible. |
| Eval key bridge (format + NTT conversion) | DONE | Verified correct: rotation [10,20,30,40] -> [20,30,40,0] |
| Ciphertext bridge (NTT batch shape) | FIXED | Forward NTT over bridged ct must use `(cipher_size × Q_size, Q_size)`, not `(cipher_size, Q_size)`. Previous shape silently skipped primes Q[1..7]. Same pattern needed for result INTT. |
| GPU batch dot product (isolated) | DONE | 0.54ms vs 48ms CPU = 89x speedup, ZERO error |
| Layered bridge_suite (10 tests in-process) | DONE | 20/20 PASS, max_err ≈ 8.5e-9. Covers ct-bridge-only, pt-bridge-only, multiply combos, +rescale, +rotate, full dot-product loop, and INTT + host round-trip. |
| gRPC server (multi-request) | BLOCKED | First call after server cold-start passes; every subsequent call returns 10^87 garbage. State leak in HEonGPU context / RMM pool / galois-key device memory. |
| End-to-end pipeline (SIFT 100K) | BLOCKED on above | Recall@10 ~14–22 % through gRPC. |

## End-to-End GPU Benchmark Results (SIFT 100K, Tesla T4)

These are measured end-to-end results on AWS g4dn.xlarge (Tesla T4, 4 vCPUs) running the full Opaque search pipeline through the GPU gRPC bridge.

### CPU Baseline (Lattigo, g4dn.xlarge 4 vCPUs)

| Config | Recall@1 | Recall@10 | Avg Latency |
|--------|----------|-----------|-------------|
| cpu-strict8 | 100% | 99.3% | 245ms |
| cpu-probe16 | 100% | 100% | 262ms |

### GPU Path (gRPC + multi-request, after NTT-batch fix + sync patches)

| Config | Recall@1 | Recall@10 | Avg Latency | Speedup |
|--------|----------|-----------|-------------|---------|
| gpu-strict8 | 10.0% | 14.7% | 155ms | **1.58x** |
| gpu-probe16 | 20.0% | 22.3% | 188ms | **1.39x** |

**Key finding:** The GPU compute pipeline is provably correct (see `bridge_suite` below: 20/20 layered in-process tests pass with 10^-8 error), but the production gRPC server leaks state between requests. The **first** call after a cold server decrypts correctly (~1e-8 error in `bridge_grpc_test`). Every subsequent call returns ~10^87 garbage. This collapses end-to-end recall to near-random.

### Layered in-process GPU tests (`bridge_suite.cpp`, Tesla T4)

| # | Layer | Result |
|---|------|--------|
| T0 | HEonGPU native encrypt + decrypt with loaded Lattigo sk | PASS (5e-10) |
| T1 | Lattigo ct coef → cudaMemcpy → NTT → decrypt | PASS (6e-10) |
| T2 | Lattigo pt coef → cudaMemcpy → NTT → decode | PASS (4e-12) |
| T3 | Native ct × bridged pt → decrypt | PASS (6e-11) |
| T4 | Bridged ct × native pt → decrypt | PASS (8e-11) |
| T5 | Bridged ct × bridged pt → decrypt | PASS (8e-11) |
| T6 | T5 + rescale_inplace | PASS (4e-10) |
| T7 | T6 + rotate_rows(1) | PASS (2e-8) |
| T8 | Full dot-product loop (rotate 1,2,4,…,64 + add) | PASS (9e-9) |
| T9 | Result INTT → host round-trip → NTT → decrypt | PASS (9e-9) |

All 20/20 tests pass across two independent test cases. The bridge math, the Galois key conversion, the cudaMemcpy plumbing and the full dot-product + result-INTT round-trip are each correct when invoked in a single process.

### gRPC round-trip test (`test/bridge_grpc_test.go`)

Same compute, but driven through the production gRPC `GPUHEProvider` with Lattigo-side decrypt:

| Run | Case 0 | Case 1 | Case 2 | Case 3 | Case 4 |
|-----|--------|--------|--------|--------|--------|
| err | 1e-8 (**PASS**) | 10^87 (FAIL) | 10^87 (FAIL) | 10^87 (FAIL) | 10^87 (FAIL) |

Pattern is deterministic across restarts: exactly the first request after a cold server produces correct results. Identical inputs on subsequent requests still fail.

## Bugs Found and Fixed

Over the course of this effort we found and fixed **6** distinct bugs in the Lattigo-to-HEonGPU bridge. Each was a subtle interop issue that produced silently wrong results.

| # | Bug | Symptom | Root Cause | Fix |
|---|-----|---------|------------|-----|
| 1 | Montgomery removal on non-Montgomery data | Results off by factor of R=2^64 mod Q | Lattigo v5 ciphertexts have `IsMontgomery=false`. We were dividing by R when data was already in standard form. | Check `IsMontgomery` flag; only remove Montgomery when true |
| 2 | Missing LogDimensions and IsBatched metadata | Decoder treats 8192-slot ciphertext as single value | Reconstructed ciphertexts had `Cols:0`, `IsBatched:false` | Set `LogDimensions={LogN-1, 1}` and `IsBatched=true` on reconstructed ciphertexts |
| 3 | HEonGPU ciphertext `scale_=0` | `multiply_plain` computes `output.scale = 0 * pt_scale = 0` | HEonGPU default constructor sets `scale_=0`; after `cudaMemcpy`, scale remains zero | Explicitly set `scale_` after data loading |
| 4 | Plaintext all zeros | GPU multiply produces zero output | Using `rlwe.NewPlaintext` (scale=0) instead of `hefloat.NewPlaintext` (sets scale from params) | Use `hefloat.NewPlaintext` which properly initializes scale |
| 5 | Proto Depth field semantics mismatch | Fresh ciphertexts treated as nearly-exhausted | Go sends Lattigo `Level()` (7) as Depth, but HEonGPU expects `depth=0` for fresh ciphertext | Send `depth=0` for fresh ciphertexts; HEonGPU depth = max_level - lattigo_level |
| **6** | **HEonGPU RNS NTT batch shape** | **Forward NTT on bridged ct silently skipped primes Q[1..7]; recall collapsed to 14 %.** | **Used `GPU_NTT_Inplace(batch=cipher_size, mod_count=Q_size)`. HEonGPU's own `operator.cu` uses `batch_size = cipher_size × decomp_count, mod_count = decomp_count` for every ciphertext-level RNS NTT call. The kernel in `PerPolynomial` layout treats `batch_size` as the total number of polynomial-NTTs to issue, not the cipher-dimension. The smaller batch value we had made the kernel only launch work for mod 0.** | **Forward: `(num_polys × Q_size, Q_size)`. Result INTT: `(cipher_size × res_levels, res_levels)`, switch to the dedicated `GPU_INTT_Inplace` entry with `mod_inverse` set explicitly.** |

## Remaining Unsolved Issue: gRPC Server State Leak

After bug #6 was fixed, every layer of the bridge works in-process (20/20 tests in `deploy/gpu/gpu-he-server/bridge_suite.cpp`). **But the production gRPC server still fails on every request after the first.**

### What we proved PASSES

- **Canonical embedding is compatible.** `encode_compare` dumps the coefficient polynomial of the same input vector from both libraries; Q[0..7] match byte-for-byte. Previous hypothesis that Lattigo and HEonGPU encoded vectors differently is disproved.
- **Ciphertext bridge is correct.** Lattigo ct → INTT → coefficients → `cudaMemcpy` → HEonGPU NTT → decrypt (with loaded Lattigo sk) recovers the original vector with CKKS noise floor error. `bridge_suite` T1 proves it.
- **Full GPU pipeline is correct.** Bridged ct × bridged pt → rescale → 7 rotations (powers of 2 up to 64) + add → slot 0 of each segment decrypts to the expected dot product. `bridge_suite` T8 passes with 9e-9 error.
- **Return path is correct.** After the rotation loop, INTT on result + host round-trip + NTT + decrypt reproduces the expected values. `bridge_suite` T9 passes.
- **First gRPC request is correct.** `test/bridge_grpc_test.go` with `nCases=1` passes with 5e-9 error — through the real `GPUHEProvider`, real gRPC transport, and real Lattigo-side decrypt.

### What FAILS

Any BatchDotProduct request AFTER the first on a live gRPC server returns 10^87-order garbage, deterministically, with identical inputs. The failure is independent of:

- Query / centroid data (reproduced with fixed seed across cases)
- NTT batch shape (already the fixed one)
- Scale / depth metadata (server logs `depth=1 numLevels=7 scale=2^45` on every response)
- Session-level state (session map keeps the same context, encoder, ops, galois_key across calls)

Patches that did NOT fix the second-call bug:
- Fresh `HEEncoder` / `HEArithmeticOperator` constructed inside each gRPC handler
- `cudaSetDevice(0)` at handler entry
- `cudaDeviceSynchronize` between every op, including around the Ciphertext copy constructor that `cudaMemcpyAsync`s the accumulator
- `cudaMemset` zero-fill of `ct_query.data()` before the cudaMemcpy of bridged coefficients

### Hypothesis

The leak is below the HEonGPU op layer. Most likely suspects:

1. **RMM memory pool.** HEonGPU's `DeviceVector<T>` extends `rmm::device_uvector<T>`. After the first request frees its temporaries, the pool reuses the same device addresses for subsequent allocations. Some scratch buffer (e.g. `new_prime_locations_` inside `HEArithmeticOperator`, or context-level `rescaled_*`) may be read while being overwritten by the concurrent request.
2. **HEContext scratch buffers.** Fields like `rescaled_half_`, `rescaled_half_mod_`, `rescaled_last_q_modinv_` live on the context and are reused by every rescale. If they require some initialization that only happens during the first rescale or the first NTT forward pass, the second call reads stale values.
3. **Galois key device memory corruption.** `apply_galois_ckks_method_I` performs an in-place GPU_NTT_Modulus_Ordered_Inplace on a temp buffer and reads `galois_key.device_location_[galoiselt]`. If any in-place NTT clobbers its own input after the first run, subsequent rotations see corrupted keys. Inspection of the kernel code suggests read-only access, but the effect matches.

None of these have been confirmed yet, but the stateful failure pattern — **cold server → first call works, every later call returns garbage of magnitude ~10^87** — is a fingerprint of an in-place op on a buffer that's aliased by a subsequent allocation.

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

If someone (including future us) wants to resume the GPU acceleration work, the compute path is solved — focus entirely on the gRPC server state leak.

### 1. Reproduce the failure quickly

```
cd deploy/gpu
terraform apply -var="enabled=true" -var="aws_profile=personal" -var="use_spot=false"
# wait for setup, then:
bash deploy/gpu/run_encode_compare.sh   # encoder match: expected PASS
# bridge_suite in-process:
ssh ... 'cd opaque/deploy/gpu/gpu-he-server/build && ./bridge_suite \
  /tmp/bridge_sk.bin /tmp/bridge_galois.bin /tmp/bridge_test.bin'    # 20/20 PASS
# gRPC round-trip:
GPU_HE_SERVER=localhost:50052 go test -tags sift1m -count=1 -v \
  -run TestBridge_GPURoundTrip ./test/ -timeout 5m                   # case 0 PASS, rest FAIL
```

### 2. First investigation — RMM pool behaviour

HEonGPU uses RMM. Our best guess is its pool reuses device memory between gRPC calls in a way that aliases an in-flight scratch buffer. Concrete next step: call
`heongpu::MemoryPool::use_memory_pool(false)` (defined in `src/lib/util/memorypool.cu`) during `InitContext` and re-run `TestBridge_GPURoundTrip` with 5 cases. If they all pass, the pool is the culprit and the fix is either disabling the pool globally or isolating per-request allocations.

### 3. Second investigation — serialise BatchDotProduct

Set gRPC server's `thread_pool` size to 1 via `ServerBuilder::SetSyncServerOption(NUM_CQS, 1); SetSyncServerOption(MIN_POLLERS, 1); SetSyncServerOption(MAX_POLLERS, 1);`. If passing on all 5 cases, the bug is thread-bound CUDA context re-entry. The fix is to either pin the server to a single worker or call `cudaSetDevice(0)` + `cudaStreamSynchronize(cudaStreamDefault)` at handler entry.

### 4. Third investigation — galois key immutability

Before each `rotate_rows_inplace`, checksum the galois key's device memory (or the specific key selected for `shift`). If the checksum differs between the first request and the second, a prior op is writing back into the galois key. The fix would be to reload galois keys between requests (expensive — 280 MB per session) or file an upstream HEonGPU issue.

### 5. Key files

| File | Purpose |
|------|---------|
| `pkg/crypto/ntt_convert.go` | Go-side NTT domain conversion (Lattigo ↔ HEonGPU). Done. |
| `pkg/crypto/ciphertext_convert.go` | Ciphertext format conversion for the gRPC bridge. Done. |
| `deploy/gpu/gpu-he-server/main.cpp` | C++ GPU server. NTT batch shape fix is here. State leak is here. |
| `deploy/gpu/gpu-he-server/bridge_suite.cpp` | 10-layer in-process GPU test. PASSES 20/20. |
| `deploy/gpu/gpu-he-server/encode_compare.cpp` | Proves Lattigo and HEonGPU coefficients match byte-for-byte. |
| `cmd/bridge-export/main.go` | Dumps the test fixture bridge_suite consumes. |
| `test/bridge_grpc_test.go` | Production twin. First call PASS, rest FAIL — the remaining bug. |

### 6. Key tests

- **In-process GPU**: `./bridge_suite` — 20/20 PASS. Gold standard.
- **Production gRPC**: `TestBridge_GPURoundTrip` — 1/N PASS, N-1 FAIL. The bug to fix.
- **End-to-end SIFT 100K**: `TestGPU_E2E_SIFT100K` — 10–22 % recall. Will fix itself once the gRPC state leak is resolved.

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
