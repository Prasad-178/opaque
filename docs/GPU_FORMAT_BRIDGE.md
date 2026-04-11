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

## Verification Plan

Before building the full pipeline, verify format compatibility:
1. Create a ciphertext in Lattigo with known values
2. Extract raw coefficients
3. Reconstruct in HEonGPU
4. Decrypt in HEonGPU (using same secret key raw data)
5. Compare plaintext — if it matches, the bridge works
