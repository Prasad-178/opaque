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

## Opaque's Current Implementation (post-mitigation, 2026-04)

**File:** `pkg/crypto/threshold/threshold.go`, line 237

```go
noiseFlood := ring.DiscreteGaussian{Sigma: 1 << 30, Bound: 6 * (1 << 30)}
```

**Sigma = 2^30 ≈ 1.07e9** (~30 bits of masking, +10 bits over the previous 2^20).

Composed with `DecodePublic(pt, values, 10)` on every client-facing decryption
(see `pkg/crypto/crypto.go` and `pkg/crypto/threshold/threshold.go` `DecryptScalar` /
`DecryptBatchScalars`). DecodePublic rounds output to 2^-10 ≈ 1e-3 precision per
Lattigo SECURITY.md, sanitizing residual noise that the Li-Micciancio attack uses.

### Why not sigma=2^45?

Bergamaschi PKC 2025 (eprint 2024/424) recommends sigma~2^45 for provable 128-bit
IND-CPA^D with tau<=2^20 decryptions. We tried this. **It catastrophically destroys
signal in our parameter set**: CKKS plaintext scale per LogQ prime is 2^45, so
flooding sigma=2^45 produces post-decode noise of magnitude ~1.0 on [-1, 1] scores
(verified: TestThresholdDecryptScalar produced 10^88-magnitude garbage). Provable
sigma=2^45 requires enlarging the LogQ chain so flooding stays << scale. **Deferred
as a parameter-restructure task.**

Sigma=2^30 is the highest value that preserves signal in the current parameter set:
post-decode flooding noise ~ 2^30/2^45 = 2^-15 ≈ 3e-5, well below the 2^-10
DecodePublic rounding precision (so the rounding masks it).

### Is sigma=2^30 + DecodePublic Sufficient?

**For provable 128-bit IND-CPA^D under Bergamaschi: No** — needs ~2^45 with bigger
LogQ chain.

**For Opaque's practical use case: Yes** — Li-Micciancio key recovery requires the
attacker to recover the residual noise polynomial. DecodePublic destroys that
information by rounding. Sigma=2^30 strictly dominates the previous 2^20 and is
the maximum compatible with the current parameter set.

### Was sigma=2^20 Sufficient? (historical)

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

## Status of Recommended Fixes

### 1. Use `DecodePublic` Instead of `Decode` — **DONE 2026-04-24**

Applied in 7 sites: `pkg/crypto/crypto.go` (3), `pkg/crypto/threshold/threshold.go`
(2), `pkg/crypto/threshold_provider.go` (2). Logprec=10 (rounds to 2^-10 ≈ 1e-3).
`decodeLogPrec` const declared in both `crypto` package and `threshold` subpackage.

### 2. Increase Sigma to 2^30 — **DONE 2026-04-24**

Applied at `pkg/crypto/threshold/threshold.go:237`.

```go
noiseFlood := ring.DiscreteGaussian{Sigma: 1 << 30, Bound: 6 * (1 << 30)}
```

~30 bits of masking (+10 bits over previous 2^20). [Noah's Ark](https://eprint.iacr.org/2023/815) (WAHC/CCS 2023) shows that when both ciphertext noise and flooding noise are Gaussian, simulation security is achievable with smaller flooding than worst-case analysis suggests.

### Post-mitigation SIFT1M verification (M4 Pro, 10 CPU)

| Config | Recall@1 | Recall@10 | Avg Query |
|---|---|---|---|
| strict-4 (3% probe) | 82.0% | 87.8% | 152.7 ms |
| strict-8 (6%) | 98.0% | 97.0% | 177.8 ms |
| strict-16 (12%) | 100.0% | 99.8% | 333.5 ms |
| probe-8 (6%, multi-probe) | 100.0% | 100.0% | 262.6 ms |
| probe-16 (12%, multi-probe) | 100.0% | 100.0% | 455.7 ms |

**Zero recall regression vs pre-mitigation baseline.** Latency unchanged within noise.

## Remaining Recommended Fixes

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
