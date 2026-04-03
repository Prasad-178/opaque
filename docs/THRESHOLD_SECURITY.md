# Threshold CKKS Security Audit

## Summary

Opaque's threshold CKKS implementation uses Lattigo v5's `mhe` package with noise flooding (sigma=2^20). This audit examines known attacks on threshold CKKS, Lattigo's security posture, and whether our parameters provide adequate protection.

**Key finding:** Sigma=2^20 is a common practical default but is **not sufficient for provable 128-bit simulation security**. The practical risk for Opaque's use case is low (centroid scores need only ~4-5 bits of precision), but a paper or production deployment should address this.

## The Li-Micciancio Attack (Eurocrypt 2021)

**Paper:** Li & Micciancio, "On the Security of Homomorphic Encryption on Approximate Numbers" ([ePrint 2020/1533](https://eprint.iacr.org/2020/1533))

### What They Found

CKKS decryption outputs an *approximation* of the plaintext, not the exact value. The approximation error encodes information about the secret key:

```
Dec(sk, ct) = m + e    where e is the CKKS noise term
```

Given a known plaintext `m`, ciphertext `ct`, and decryption result `m' = Dec(sk, ct)`, an adversary can compute `e = m' - m`. This noise `e` is a linear function of the secret key `sk`. With as few as **1-2 decryption results** (plaintext-ciphertext-decryption triples), the attacker can perform **complete key recovery** in polynomial time.

### New Security Model: IND-CPA^D

They defined IND-CPA^D (IND-CPA with Decryption oracle), which extends IND-CPA to allow the adversary limited access to decryption. Standard CKKS is IND-CPA secure but **NOT** IND-CPA^D secure.

### Why This Matters for Threshold CKKS

In threshold decryption, the result is shared with the querying client. If the client knows or can guess the original plaintext (e.g., because they constructed the query), they obtain approximation errors that leak the collective secret key. This is exactly the Li-Micciancio scenario.

**In Opaque's case:** The client encrypts a known query vector with CKKS, the server computes dot products, and the client receives decrypted scores. The client knows the query and can infer the approximate expected scores (they have the centroids from credentials). This gives them the plaintext-ciphertext-decryption triples needed for the attack.

## Lattigo v5's Security Posture

Based on [Lattigo SECURITY.md](https://github.com/tuneinsight/lattigo/blob/main/SECURITY.md) and the `mhe` package:

### What Lattigo Provides

1. **Noise flooding** — configurable via `ring.DiscreteGaussian{Sigma: ..., Bound: ...}` parameter at protocol instantiation. It is NOT automatic — the developer must choose sigma.

2. **`DecodePublic`** — a sanitized decoding function that rounds/truncates output to eliminate residual noise leakage. Available but must be used explicitly instead of `Decode`.

3. **PublicKeySwitchProtocol (PCKS)** — the threshold decryption protocol. Each participant adds noise flooding to their partial share before transmitting.

### What Lattigo Does NOT Provide

1. **No simulation-based security claims.** The docs state: "The implemented MHE-MPC protocol is secure against passive adversaries." No UC-security or simulation proofs.

2. **No default secure sigma value.** The developer must choose. There is no `SecureDefault()` function.

3. **No retry protection.** SECURITY.md explicitly warns: "The current implementation...assumes that any party...will not generate and transmit its share more than once." Retrying MHE protocols enables key-recovery attacks.

4. **No automatic key rotation.** The developer must implement bounded-query key refresh.

### What Lattigo Warns About

From SECURITY.md:
> "Applications that use the CKKS scheme for MHE protocols should be aware of the Li-Micciancio attack and take appropriate countermeasures."

> "The countermeasures from [Mouchet et al. 2024] and [Colin de Verdiere et al.] are **not currently implemented** in Lattigo."

## Opaque's Current Implementation

**File:** `pkg/crypto/threshold/threshold.go`, line 237

```go
noiseFlood := ring.DiscreteGaussian{Sigma: 1 << 20, Bound: 6 * (1 << 20)}
```

**Sigma = 2^20 ≈ 1,048,576** (~20 bits of masking)

### Is sigma=2^20 Sufficient?

**For provable security: No.**

From the [OpenFHE community](https://openfhe.discourse.group/t/appropriate-error-parameters-for-the-noise-flooding/95), OpenFHE developers acknowledge that "adding 20 bits of noise is not sufficient to achieve 128 bits of security in the general case." The 20-bit value was chosen for efficiency, assuming limited queries.

From [Bergamaschi et al., PKC 2025](https://eprint.iacr.org/2024/424) ("Revisiting the Security of Approximate FHE with Noise-Flooding Countermeasures"):
- For 128-bit IND-CPA^D security: **45-47 bits of noise flooding** may be needed
- This severely limits precision (only 8-16 bits of message precision remain)
- There is a three-way tradeoff: number of allowed decryptions, flooding variance, and concrete security bits

**For Opaque's practical use case: Probably adequate.**

- Centroid similarity scores are in [-1, 1] and differ by magnitudes of 0.01-0.1
- Only ~4-5 bits of precision are needed to distinguish top-K clusters
- The client does not have an unlimited decryption oracle — queries are bounded
- The client already knows the centroids (from credentials), limiting what they can learn

### Headroom Analysis

Opaque's CKKS parameter set:
- LogQ = [60, 45, 45, 45, 45, 45, 45, 45] = 375 bits total
- LogP = [61, 61] = 122 bits
- Dot product consumption: ~60-80 bits (multiply + rescale + rotations)
- Current noise flooding: ~20 bits (sigma=2^20)
- Remaining headroom: ~275 bits

**Even sigma=2^45 would leave ~250 bits of headroom** — more than sufficient. The precision loss from 2^45 flooding would give ~8-16 bits of message precision, which is still enough for centroid scoring (we need ~4-5 bits).

## Recommended Fixes (Prioritized)

### 1. Use `DecodePublic` Instead of `Decode` (Easy, High Impact)

**Current code** (`pkg/crypto/threshold/threshold.go`, ClientSession.DecryptScalar):
```go
s.encoder.Decode(pt, decoded)  // Leaks residual noise
```

**Should be:**
```go
s.encoder.DecodePublic(pt, decoded, logprec)  // Sanitizes output
```

This rounds the output to a fixed precision, eliminating the residual noise that enables the Li-Micciancio attack. Zero performance impact.

### 2. Increase Sigma to 2^30 (Easy, Moderate Impact)

Change line 237:
```go
noiseFlood := ring.DiscreteGaussian{Sigma: 1 << 30, Bound: 6 * (1 << 30)}
```

This provides ~30 bits of masking (vs current 20 bits). Still preserves sufficient precision for centroid scoring. The [Noah's Ark paper](https://eprint.iacr.org/2023/815) (WAHC/CCS 2023) shows that when both ciphertext noise and flooding noise are Gaussian, simulation security is achievable with smaller flooding than the worst-case analysis suggests.

### 3. Implement Key Rotation (Medium Effort, High Impact)

After a bounded number of threshold decryptions (e.g., tau=10,000 queries), rotate keys:
- Generate new key shares
- Re-distribute via Shamir
- Regenerate collective keys

This bounds the number of decryption triples an adversary can collect, tightening the security analysis regardless of sigma.

### 4. Enforce No-Retry on MHE Shares (Easy)

Add a per-ciphertext nonce tracked by each participant. If a participant is asked to generate a share for the same ciphertext twice, reject the request. This prevents the retransmission attack Lattigo warns about.

### 5. Document the Security Model (Easy, Important for Paper)

Explicitly state in the paper/docs:
- Threat model: honest-but-curious server, passive adversaries
- Security guarantees: IND-CPA for HE operations, statistical noise masking for threshold decryption
- Bounded queries per key epoch (tau)
- Not provably IND-CPA^D-128 secure, but practically secure for the centroid scoring use case
- Discuss the tradeoff: provable security requires 2^45 flooding which limits precision; practical security with 2^30 flooding preserves application utility

## Key References

| Paper | Venue | Relevance |
|-------|-------|-----------|
| Li & Micciancio, "On the Security of HE on Approximate Numbers" | Eurocrypt 2021 | The original attack on CKKS decryption |
| Bergamaschi et al., "Revisiting Security of Approximate FHE with Noise Flooding" | PKC 2025 | Concrete security analysis, sigma tradeoffs |
| Dahl et al., "Noah's Ark: Efficient Threshold-FHE Using Noise Flooding" | WAHC/CCS 2023 | Simulation security with small flooding when noise is Gaussian |
| Bossuat et al., "Practical q-IND-CPA-D-Secure Approximate HE" | 2023 | Practical parameter selection |
| Lattigo SECURITY.md | — | Explicit warnings about mhe security limitations |
| OpenFHE Noise Flooding Guide | — | Practical sigma recommendations and limitations |
