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

## Opaque's Current Implementation (post-restructure, 2026-05)

**File:** `pkg/crypto/threshold/threshold.go`, line 237

```go
noiseFlood := ring.DiscreteGaussian{Sigma: 1 << 45, Bound: 6 * (1 << 45)}
```

**Sigma = 2^45 ≈ 3.5e13** (~45 bits of masking).

LogQ chain restructured (commit `e42338f`):
- Old: `LogQ=[60, 45, 45, 45, 45, 45, 45, 45]` + `LogDefaultScale=45`
- New: `LogQ=[60, 60, 60, 60, 60]` + `LogDefaultScale=60`

Composed with `DecodePublic(pt, values, 10)` on every client-facing decryption
(see `pkg/crypto/crypto.go` and `pkg/crypto/threshold/threshold.go` `DecryptScalar` /
`DecryptBatchScalars`). DecodePublic rounds output to 2^-10 ≈ 1e-3 precision per
Lattigo SECURITY.md, sanitizing residual noise that the Li-Micciancio attack uses.

### Why σ=2^45 with restructured chain

Bergamaschi PKC 2025 (eprint 2024/424) gives σ ≈ 2^45 for **provable 128-bit
IND-CPA^D security** with τ ≤ 2^20 decryptions per key. With the previous
`LogDefaultScale=45`, σ=2^45 had the same magnitude as the plaintext scale and
destroyed signal entirely (post-decode noise ~1.0 on [-1,1] scores). Bumping
`LogDefaultScale` to 60 gives 2^45/2^60 = 2^-15 ≈ 3e-5 post-decode noise — same
ratio as the old σ=2^30/scale=2^45 setup, signal preserved.

5 primes (vs old 8) is sufficient because Opaque's HE circuit depth is 1
(multiply + rotations + adds). Total `LogQ + LogP = 422` bits at `LogN=14`
maintains 128-bit RLWE security.

### Is σ=2^45 + DecodePublic Sufficient?

**For provable 128-bit IND-CPA^D under Bergamaschi PKC 2025: Yes** — σ=2^45
satisfies the bound for τ ≤ 2^20 decryptions per key.

**Defense in depth:** DecodePublic compositionally destroys residual noise the
Li-Micciancio attack uses, regardless of σ. The combination of parametric σ
(provable) + DecodePublic (operational) gives belt-and-suspenders coverage.

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

### Post-mitigation SIFT1M verification (AWS c6i.2xlarge, 8 vCPU Intel Ice Lake)

Full-mitigation run (2026-04-30 19:08, commit `e45223b`+`4c5abf8`): σ=2^30 +
DecodePublic, per-tenant blob ID permutation π (now correctly wired through
public `opaque.NewDB` API), `PaddingMode=Bucketed`, `TargetEpsilon=2.0`
(NumDecoys derived to 17).

| Config | Recall@1 | Recall@10 | Avg Query |
|---|---|---|---|
| strict-4 (3% probe) | 94.0% | 88.4% | 410 ms |
| strict-8 (6%) | 98.0% | 95.4% | 466 ms |
| strict-16 (12%) | 100.0% | 99.2% | 640 ms |
| probe-8 (6%, multi-probe) | 100.0% | 99.4% | 630 ms |
| probe-16 (12%, multi-probe) | 100.0% | 100.0% | 815 ms |

**Recall identical-or-better across configs vs partial-mitigation run.** Latency
adds ~30-65 % (extra decoys via ε=2 + Bucketed padding bandwidth). See
`deploy/bench-cpu/results/SUMMARY.md` for full pre/partial/full comparison.

## Status of Decoy / Volume / Permutation Mitigations (added 2026-04-30)

- **Per-tenant blob ID permutation π** — **Done (commit `bc0ec45`)**. Hides centroid-to-storage link from server.
- **Constant-volume padding** — **Done (commit `414aa8e`)**. `enterprise.PaddingMode` (None | Bucketed | MaxFixed) closes the volume side-channel.
- **TargetEpsilon-tunable decoy count** — **Done (commit `990e2be`)**. ε-style upper bound on per-query distinguishability; customers pick ε directly.
- **DP formalization writeup** — **Done**. See `docs/SECURITY_MODEL.md` §5.1.

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
