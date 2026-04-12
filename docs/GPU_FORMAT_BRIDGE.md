# GPU Format Bridge: Lattigo ↔ HEonGPU Ciphertext Interop

## Problem

Lattigo (Go) and HEonGPU (C++/CUDA) both implement CKKS but have different serialization formats. To offload HE computation to HEonGPU while keeping encryption/decryption in Lattigo, we need to bridge the ciphertext format.

## Key Finding: Raw Coefficient Data Is Compatible

Both libraries represent CKKS ciphertexts as arrays of `uint64` coefficients in RNS (Residue Number System) + NTT (Number Theoretic Transform) representation. The underlying mathematical objects are identical — only the serialization format differs.

### Lattigo Internal Layout

```go
// A ciphertext is Element[ring.Poly] with Value = [c0, c1] (two polynomials)
// Each ring.Poly has Coeffs = [][]uint64 (Matrix[uint64])
// Coeffs[level][coeff_index] = uint64 coefficient

ciphertext.Value[0].Coeffs  // c0: [level0[N], level1[N], ...]
ciphertext.Value[1].Coeffs  // c1: [level0[N], level1[N], ...]

// N = ring_size = 2^LogN = 16384
// len(Coeffs) = number of active levels (coeff_modulus_count - depth)
```

Access raw data: `ciphertext.Value[i].Coeffs[level]` returns `[]uint64` of length N.

Get modulus primes: `params.RingQ().ModuliChain()` returns `[]uint64`.

### HEonGPU Internal Layout

```cpp
// Ciphertext stores Data64* (uint64_t*) either on GPU or host
// Flat contiguous array: [c0_level0[N] | c0_level1[N] | ... | c1_level0[N] | ...]
// Total size: cipher_size × (coeff_modulus_count - depth) × ring_size

ciphertext.data()  // Returns Data64* pointer to flat array
ciphertext.get_data(host_vector)  // Copies GPU data to host

// Set exact modulus primes:
context->set_coeff_modulus_values(Q_primes, P_primes);
```

### The Bridge

Extract `[]uint64` from Lattigo → flatten → send via gRPC → reconstruct in HEonGPU:

```
Lattigo (Go):
  coeffs = []uint64{}
  for poly in [c0, c1]:
    for level in range(active_levels):
      coeffs = append(coeffs, poly.Coeffs[level]...)
  // Send coeffs + metadata via gRPC

HEonGPU (C++):
  // Receive flat uint64[] + metadata
  // Create Ciphertext, set data directly
  memcpy(ciphertext.host_data(), received_data, size * sizeof(uint64_t))
  ciphertext.store_in_device()  // Upload to GPU
```

### Critical Requirements

1. **Same modulus primes:** Extract exact `[]uint64` primes from Lattigo's `params.RingQ().ModuliChain()` and pass to HEonGPU's `set_coeff_modulus_values()`.

2. **Same NTT domain:** Both must be in NTT form. Lattigo defaults to NTT (`IsNTT = true`), HEonGPU defaults to NTT (`in_ntt_domain_ = true`).

3. **Same coefficient ordering:** Both use standard NTT on the cyclotomic ring Z[X]/(X^N+1). The NTT roots are derived deterministically from the modulus primes. If primes match, the NTT representation matches.

4. **Same scale:** Pass `ciphertext.Scale.Float64()` from Lattigo to HEonGPU's `scale_` field.

## Architecture

```
Client (Go/Lattigo)                         GPU Server (C++/HEonGPU)
│                                           │
├─ Setup (one-time):                        │
│  Extract Q primes, P primes from params   │
│  Extract Galois keys as raw coefficients  │
│  ──── RegisterEvalKeys (gRPC) ──────────► │ Create HEContext with exact same primes
│                                           │ Reconstruct Galois/Relin keys
│                                           │
├─ Per query:                               │
│  Encrypt query (Lattigo)                  │
│  Extract raw uint64[] from ciphertext     │
│  ──── BatchDotProduct (gRPC) ───────────► │ Reconstruct ciphertext from raw data
│                                           │ Upload to GPU
│                                           │ MulNew + Rescale + RotateNew (GPU)
│  ◄──── raw uint64[] result ──────────── │ Extract result coefficients
│  Reconstruct Lattigo ciphertext           │
│  Decrypt locally (secret key stays here)  │
│                                           │
└─ Privacy preserved:                       │
   Server never sees plaintext query        │
   Server never has secret key              │
   Only encrypted data crosses the wire     │
```

## Risks

1. **NTT root mismatch:** If Lattigo and HEonGPU derive NTT primitive roots differently for the same prime, the coefficient values will differ even with matching primes. This is the biggest risk. Mitigation: test with a known plaintext and verify the result matches.

2. **P primes (key-switching primes):** Galois keys use the extended modulus chain Q∪P. Both libraries must use the same P primes for key-switching to work.

3. **Polynomial ordering in ciphertext:** Both store (c0, c1) but we must verify the same ordering convention (c0 first, then c1).

## Evaluation Key Bridge: Decomposition Match Found

### The Problem

Lattigo and HEonGPU use different key-switching decomposition parameters by default:
- Lattigo with `LogP=[61,61]` (2 P primes): `d=4` decomposition, 10 MB per Galois key
- HEonGPU: `d=Q_size=8` decomposition, 18 MB per Galois key

With different decomposition, the raw polynomial data is mathematically different — not just differently formatted.

### The Solution: GPU-Compatible Parameters

Using `LogP=[61]` (single P prime) forces Lattigo's decomposition to match HEonGPU:

| Parameter | CPU Path `[61,61]` | GPU Path `[61]` | HEonGPU |
|-----------|-------------------|-----------------|---------|
| **d (decomposition)** | 4 | **8** | **8** |
| **P levels per poly** | 2 | **1** | **1** |
| **uint64 per Galois key** | 1,310,720 | **2,359,296** | **2,359,296** |
| **MB per Galois key** | 10.0 | **18.0** | **18.0** |

**Exact size match confirmed.** `NewParametersGPU()` uses `LogP=[61]` for GPU-compatible key generation.

The CPU path (`LogP=[61,61]`) remains unchanged — it's slightly more efficient for CPU-only operation (fewer decomposition components = fewer polynomial multiplications per rotation). The GPU path trades this for cross-library compatibility.

### Remaining Verification

The sizes match, but we need to verify the coefficient ORDERING within the arrays:
1. Generate a Galois key in Lattigo (GPU params) and extract raw uint64 arrays
2. Generate a Galois key in HEonGPU with the same secret key and parameters
3. Compare byte-for-byte — if they match, the bridge works without any conversion

This requires running both libraries with the same secret key on a CUDA-capable machine.

### Current Progress (as of testing)

**Format structure: CORRECT** — file loads in HEonGPU without crash, size matches byte-for-byte (283,115,733 bytes).

**Key issues found and fixed:**
1. HEonGPU enums (scheme_type, keyswitching_type, storage_type) are uint8, not int32
2. HEonGPU's `load()` has a bug in the `customized=true` path — uses uninitialized variable as read length. Workaround: use `customized=false` with galois_elt map format.
3. `d_` should be 0 for KEYSWITCHING_METHOD_I (not Q_size)
4. `group_order_` should be 5 (the NTT generator for N=16384), not 3

**Remaining issue:** Rotation produces incorrect results (garbage values). The polynomial coefficient ORDERING within each key's data block differs between Lattigo and HEonGPU. Both use Cooley-Tukey NTT with bit-reversed output, but may derive different primitive roots of unity for the twiddle factors.

### NTT Root Analysis (completed)

**Root cause confirmed:** Lattigo and HEonGPU use different primitive roots of unity (ψ) for their NTT. All 9 primes have different ψ values between libraries.

**Why they differ:**
- Lattigo deterministically selects roots based on its internal SubRing initialization
- HEonGPU uses `std::random_device` to find an initial root, then minimizes from that starting point via `find_minimal_primitive_root`. The minimization depends on the random starting point, making the roots **non-deterministic across HEonGPU instances**

**Solution: NTT root exchange protocol.** The GPU server creates its HEonGPU context, extracts the ψ values it chose for each prime, and sends them to the Go client during registration. The client uses these exact roots to convert evaluation key coefficients from Lattigo's NTT domain to HEonGPU's NTT domain.

The conversion per polynomial per modulus level is:
1. Apply Lattigo's INTT (using Lattigo's own `ring.INTTStandard`, handles Montgomery form correctly)
2. Apply HEonGPU's NTT (using the server-provided ψ values)

This is O(N log N) per polynomial, done once during key setup (~2 min for all 14 Galois keys).

### Implementation Status

| Component | Status |
|-----------|--------|
| File format header | ✅ Correct (scheme uint8, keyswitching uint8, storage uint8, d=0, group_order=5) |
| File size | ✅ Matches HEonGPU native (283,115,733 bytes) |
| Key loading | ✅ HEonGPU loads without crash |
| NTT root analysis | ✅ Root cause identified (different ψ values) |
| NTT root exchange | 🔲 Need proto update + server-side root extraction |
| NTT domain conversion | 🔲 Need server's ψ values to apply correct conversion |
| NTT table generation fix | ✅ Fixed: was using wrong formula, now uses simple `table[bitrev(j)] = psi^j` |
| Montgomery removal | ✅ Fixed: Lattigo stores data in Montgomery form, must divide by R before HEonGPU NTT |
| Rotation verification | ✅ **VERIFIED ON GPU** — `[10,20,30,40]` → rotate → `[20,30,40,0]` CORRECT |
| End-to-end benchmark | 🔲 Next: full Opaque pipeline through GPU on SIFT 100K |

### Bridge Verified (Tesla T4, AWS g4dn.xlarge)

```
Decrypt (no rotation): [10.00, 20.00, 30.00, 40.00] ✓
Rotated by 1:           [20.00, 30.00, 40.00, 0.00]  ✓
GPU rotation speed:     66-630 us per rotation
```

The full conversion pipeline (done entirely in Go, no GPU-side NTT needed):
1. Extract raw uint64 coefficients from Lattigo's GadgetCiphertext
2. Apply Lattigo's INTT (removes NTT evaluation, keeps Montgomery scaling)
3. Remove Montgomery factor (divide by R = 2^64 mod Q per coefficient)
4. Apply HEonGPU's NTT using server-provided psi roots
5. Write to HEonGPU binary format → `load()` reads directly

Privacy: evaluation keys are public data. Secret key never leaves client in production.

## Verification Plan

Before building the full pipeline, verify format compatibility:
1. Create a ciphertext in Lattigo with known values
2. Extract raw coefficients
3. Reconstruct in HEonGPU
4. Decrypt in HEonGPU (using same secret key raw data)
5. Compare plaintext — if it matches, the bridge works
