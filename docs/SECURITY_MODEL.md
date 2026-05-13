# Opaque Security Model

This document is the canonical statement of Opaque's threat model, security
guarantees, and known limitations. It is the "Security Considerations" section
that should ship with any paper or external technical communication.

## 1. Threat Model

### Server: honest-but-curious (HBC)

Opaque assumes a passive adversary controlling the server. The server is
trusted to **follow the protocol** (does not deviate, does not actively tamper,
does not selectively drop or reorder responses) but is **not trusted with the
contents** of queries, vectors, or results. The server may inspect all data on
disk, all data in RAM, all network traffic in and out of the server process,
and all code execution traces.

This is the same threat model used by Tiptoe (SOSP'23), PPMI (arXiv 2025), and
RemoteRAG (ACL 2025). It is **strictly weaker** than Compass's malicious-server
model (OSDI 2025), Pacmann's semi-honest model with simulation-based proofs
(ICLR 2025), and SANNS's 2PC simulation security (USENIX Sec 2020). See §6.

### Client: trusted

The client holds CKKS secret-key material (single-key mode) or one share of a
threshold-CKKS committee (threshold mode). Client device compromise yields
full data access in single-key mode; in threshold mode, requires compromise of
≥ t parties.

### Network adversary: subsumed by server

Because the HBC server already observes all network traffic to/from itself, a
strictly-weaker network observer learns nothing new. Network confidentiality
is enforced via mTLS / authenticated transport (gRPC).

### Out of scope

- **Active server tampering**: deliberate reorder, drop, replay, or substitution
  of responses. Opaque's AES-256-GCM blob layer detects substitution via the
  GCM auth tag, but cluster-index metadata has no integrity layer today. The
  HBC assumption is what permits this gap.
- **Side channels**: cache-timing, power, electromagnetic, or microarchitectural
  side channels on the server. Out of scope.
- **Denial of service**: not addressed; Opaque is a confidentiality / privacy
  system, not an availability system.
- **Client device compromise**: see §3.
- **Multi-tenant collusion**: each tenant's keys and ciphertexts are
  cryptographically independent under standard CKKS / AES-GCM separation;
  cross-tenant attacks are not analyzed beyond this.

## 2. Cryptographic Primitives and What They Protect

| Layer | Primitive | Protects | Standard |
|---|---|---|---|
| Query confidentiality | CKKS (Lattigo v5) | Query vector + intermediate dot-product results | RLWE / IND-CPA |
| Query result | CKKS partial decryption + DecodePublic | Centroid similarity scores | IND-CPA + Li-Micciancio mitigation |
| Encrypted vectors at rest | AES-256-GCM | Per-cluster vector blobs on disk | AEAD / IND-CCA2 |
| Access pattern | Decoy clusters (statistical k-anonymity) | Which cluster the client cares about | **Statistical, not cryptographic** — see §5 |
| Key ownership (optional) | Threshold CKKS via Lattigo `mhe` | No single party can decrypt unilaterally | Mouchet et al. PoPETs 2021 (semi-honest) |

### CKKS noise flooding and DecodePublic (Li-Micciancio mitigation)

Opaque's threshold-decryption path applies Gaussian noise flooding with
`σ = 2^45` before partial decryption (`pkg/crypto/threshold/threshold.go:237`)
and routes every plaintext released to the client through
`DecodePublic(_, _, logprec=10)` (rounds output to 2^-10 ≈ 1e-3 precision).

These two mitigations compose to defeat the Li-Micciancio key-recovery attack
on CKKS approximate decryption (Eurocrypt 2021, eprint 2020/1533) and provide
**provable 128-bit IND-CPA^D security** under the bound of Bergamaschi et al.
PKC 2025 (eprint 2024/424) for τ ≤ 2^20 decryptions per key.

Achieving σ=2^45 required a LogQ chain restructure (commit `e42338f`):
- Old: `LogQ=[60, 45, 45, 45, 45, 45, 45, 45]` (375 bits, `LogDefaultScale=45`)
- New: `LogQ=[60, 60, 60, 60, 60]` (300 bits, `LogDefaultScale=60`)

With the old chain, σ=2^45 noise had the same magnitude as the plaintext scale
2^45 and destroyed signal entirely (post-decode noise ~1.0 on [-1,1] scores —
verified empirically when σ=2^45 was first attempted). Restructuring to
`LogDefaultScale=60` gives 2^45/2^60 = 2^-15 ≈ 3e-5 post-decode noise, well
below the DecodePublic 2^-10 rounding precision so signal is preserved.

5 primes is sufficient because Opaque's HE circuit depth is 1 (one multiply +
log₂(d) rotations + adds). The 8-prime chain was over-provisioned.

Total chain `LogQ + LogP = 300 + 122 = 422` bits at `LogN=14` → 128-bit RLWE
security maintained per the standard CKKS security tables.

## 3. Threshold Mode (Optional)

Opaque ships an optional threshold-CKKS backend (`pkg/crypto/threshold/`) that
splits the CKKS secret across a t-of-N committee using Lattigo's `mhe` package.
No single party (including the DB server) can decrypt; t parties must
cooperate. Per-query overhead is ~0–10% in benchmarked configurations
(`docs/THRESHOLD_CKKS.md`).

### Known limitations of the threshold layer

The threshold layer is built on Mouchet et al. PoPETs 2021, which claims only
**semi-honest / passive security** with no simulation-based or UC proof.
Lattigo's own SECURITY.md documents three relevant unmitigated concerns:

1. **Li-Micciancio IND-CPA^D** — partially mitigated by σ=2^30 + DecodePublic
   (see §2). Not provably IND-CPA^D-128.
2. **Retry vulnerability (Mouchet'24, Okada'25, Colin de Verdière 2026)** — if
   any party emits a share twice for the same public polynomial `a`, key
   recovery is feasible. **No countermeasure shipped in Lattigo today.** Opaque
   plans to enforce a fresh-CRS-per-protocol-instance invariant
   ("Public Randomness Initialization" per Helium / eprint 2024/194).
3. **Concrete attack** — Checri-Sirdey Crypto 2024 (eprint 2024/116) attacked
   Lattigo by name in <1 hour on a laptop in the IND-CPA^D regime.

Opaque's threshold mode therefore gives **stronger key-distribution properties
than any other published private vector search system** (no competitor
distributes keys), but the underlying threshold-CKKS layer carries known
academic limitations that Opaque does not yet fully close.

## 4. Bounded-Decryption Assumption

The Li-Micciancio mitigation applies under a bounded-decryption assumption:
the adversary observes only τ < 2^20 plaintext-ciphertext-decryption triples per
key epoch.

In practice this is enforced by:
- **DecodePublic(logprec=10)** rounding away the residual noise that the
  attack uses (independent of τ);
- **Ephemeral key rotation** (shipped 2026-05-13). Both
  `pkg/crypto.Engine` (direct mode) and
  `pkg/crypto/threshold.Committee` (threshold mode) track decrypt counts
  via `DecryptCount()`, expose a `SetRotationLimit(tau)` gate that
  drives `ShouldRotate()`, and offer `RotateKeys()` /
  `RotateEpoch()` to mint fresh keys + reset the counter. Default τ=0
  (gate disabled); recommended τ=2^20 per Bergamaschi PKC 2025.
  Rotation is opt-in and caller-paced — typical patterns are:
  1. Poll `ShouldRotate()` between queries and rotate when it fires; or
  2. Rotate on a wall-clock schedule (e.g., daily).

## 5. Access-Pattern Privacy: Statistical, Not Cryptographic

Opaque's access-pattern hiding uses **decoy clusters**, not cryptographic
ORAM/PIR. On every query the client fetches the top-K real clusters plus K_decoy
randomly-chosen decoy clusters. The server cannot distinguish real from decoy
in any single query.

### What this protects

A passive HBC server observing a single query cannot tell which fetched cluster
was the real one. Under a uniform prior on cluster popularity, this gives
k-anonymity with k = 1 + K_decoy.

### What this does NOT protect

The leakage-abuse literature provides several attacks on statistical
access-pattern hiding:

- **Cash, Grubbs, Perry, Ristenpart (CCS 2015, "Leakage-Abuse Attacks Against
  Searchable Encryption")** — practical attacks on IND-CPA-secure SSE schemes
  exploiting access-pattern + auxiliary data.
- **Kellaris, Kollios, Nissim, O'Neill (CCS 2016)** — generic reconstruction
  attacks from access-pattern leakage alone, no query knowledge required.
- **Grubbs, Lacharité, Minaud, Paterson (S&P 2019)** — statistical-learning
  reconstruction from access patterns on range and k-NN queries.
- **Oya, Kerschbaum (USENIX Sec 2021, "Hiding the Access Pattern is Not
  Enough")** — search-pattern leakage alone breaks query privacy in
  obfuscated-access schemes.

For Opaque's decoy scheme specifically, three exploitable signals exist:

1. **Cluster-popularity skew.** Real queries follow Zipf over clusters; uniform
   decoys do not. ML over N queries reduces effective anonymity below k.
2. **Temporal correlation.** User queries are correlated; decoys are i.i.d.
   uniform. A simple Markov-chain LM separates real from decoy trajectories.
3. **Cross-query intersection.** Repeated real hits on the same cluster spike
   non-uniformly; decoys spread evenly.

A cryptographic-access-pattern competitor (Compass, Pacmann, Tiptoe, Panther)
is immune to all three by construction.

### Active mitigations (non-PIR path)

Opaque opted against a full PIR/ORAM backend for the HBC threat model after
detailed evaluation (commit history + `docs/THRESHOLD_SECURITY.md`). The
combined effect of the four mitigations below provides defense-in-depth at
zero or near-zero query latency cost:

1. **Per-tenant blob ID permutation π** — shipped (commit `bc0ec45`). At
   build time each enterprise gets a uniform random π over super-bucket IDs;
   blobs are stored at `storage_id = π[logical_id]`. Server cannot link a
   fetched storage ID back to its centroid coordinates without π. Closes
   semantic-cluster-ID inference and cross-tenant chosen-plaintext attacks.

2. **Constant-volume padding** — shipped (commit `414aa8e`). Tunable via
   `enterprise.PaddingMode` (None | Bucketed | MaxFixed). Pads each sub-bucket
   with random AES-GCM-shaped dummy blobs to equalize per-cluster fetch
   sizes. Closes the volume side-channel where unequal real-cluster sizes
   let an HBC server distinguish real from decoy by comparing total bytes
   returned. Storage cost: ~6-25% depending on mode and cluster size variance.

3. **TargetEpsilon-tunable decoy count** — shipped (commit `990e2be`).
   Customers specify a privacy budget ε; the system derives `NumDecoys =
   ⌈(N - K_real) · e^(-ε)⌉` automatically. See §5.1 below for the formal
   bound and limitations.

4. **Threshold-CKKS retry-vulnerability mitigation** — planned (Lattigo
   `mhe` no-retry invariant + fresh CRS per protocol instance). Closes the
   Mouchet'24 / Okada'25 / Colin de Verdière 2026 attacks. Tracked under §8.

### Cryptographic access-pattern hiding (deferred)

PIR via SimplePIR / YPIR remains a research direction for deployments
demanding Compass-tier (malicious-server) security. SimplePIR forces
prohibitive client hint sizes (~16 GB) at our cluster geometry (N=128
clusters × ~5 MB per cluster blob); YPIR via Rust FFI was deemed
disproportionate engineering cost (4-6 weeks) for the incremental security
gain over mitigations (1)-(3) above in the HBC threat model.

### 5.1 Formal access-pattern privacy bound

**Decoy mechanism:** for each search query, the client computes top-K_real
logical clusters via HE scoring, then fetches K_real + K_decoy storage IDs
where K_decoy random IDs are drawn uniformly without replacement from the
N - K_real non-selected pool.

**ε-style upper bound (informal):** an HBC server observing the fetched
storage-ID set S of size K_real + K_decoy cannot distinguish "real cluster
is c" from "real cluster is c' ∈ S" with probability ratio greater than

    (N - K_real) / K_decoy

Setting this ratio ≤ e^ε gives:

    K_decoy ≥ (N - K_real) · e^(-ε)

For SIFT1M (N=128, K_real=8) this yields:

| Target ε | Required K_decoy | Total fetched | % of all clusters |
|---|---|---|---|
| 1.0  | 45 | 53  | 41% |
| 2.0  | 17 | 25  | 20% |
| 3.0  |  6 | 14  | 11% |
| 4.0  |  3 | 11  |  9% |

The current default (`NumDecoys = 8`) corresponds to roughly ε ≈ 2.7.

**Caveats:**
- The bound above is an *informal* upper-bound on per-query
  distinguishability for the default uniform-K-from-non-selected scheme,
  not a formally tight (ε,δ)-DP guarantee.
- Setting `Config.BernoulliDecoys = true` switches the sampler to
  per-cluster i.i.d. Bernoulli at p = E[K_decoy] / (N - K_real). Same
  expected decoy count, but K is now binomial Bin(N - K_real, p)
  per query. This makes the mechanism amenable to the standard
  subsampled-mechanism (ε,δ)-DP composition framework (Dwork-Roth,
  Theorem 3.5+; Mironov RDP, 2017). The (ε,δ) for τ-fold composition
  follows directly from the strong-composition theorem rather than the
  loose τ · ε union bound. Trade-off: K_decoy is no longer
  deterministic, so per-query latency picks up modest variance
  (σ ≈ √(N-K_real)·p(1-p) clusters).
- The bound covers a *single* query. Composition over τ queries scales
  approximately as τ · ε under union-bound composition (with no
  cross-query correlation), or √(2τ ln(1/δ)) · ε + τ ε(e^ε - 1) under
  strong composition with the Bernoulli sampler. Long-running
  deployments should pair tunable ε with ephemeral-key rotation
  (§8 roadmap).
- The bound assumes uniform priors over real cluster choice. A workload
  with heavily-skewed cluster popularity gives the adversary a Bayesian
  prior advantage *not* covered by the bound. This is the well-known
  leakage-abuse limitation of statistical k-anonymity (Cash CCS'15, Oya
  USENIX Sec'21).

**Tunability:** `Config.TargetEpsilon = 2.0` is recommended for typical
HBC enterprise deployments. Push to ε=1.0 for high-sensitivity workloads
(more bandwidth). Use ε=3-4 for cost-sensitive deployments where workload
priors are weak.

**Composing with permutation π and padding:** these three mitigations target
distinct channels:
- π hides the **semantic identity** of fetched IDs.
- Padding hides the **volume** of fetched data.
- Tunable decoy ε bounds the **per-query distinguishability** of which ID
  in the fetched set is real.

A workload-prior adversary still has residual frequency-analysis power
across many queries; the bound on that residual is the τ · ε composition
limit modulated by ephemeral-key rotation. PIR remains the only mitigation
that closes this composition channel cryptographically; the four-layer
defense above bounds it statistically with a quantifiable parameter.

## 6. Comparison to Competitor Threat Models

| System | Server model | Access-pattern hiding | Formal proof |
|---|---|---|---|
| **Opaque** | **HBC** | **Statistical (decoys)** | IND-CPA + AEAD composition |
| Compass (OSDI '25) | **Malicious / fully compromised** | **Cryptographic (Ring ORAM)** | Theorem 1 (IND-CCA2 + CRHF) |
| Pacmann (ICLR '25) | Semi-honest | Cryptographic (PIR) | **Simulation-based (Defn 2.1)** |
| Tiptoe (SOSP '23) | HBC | Cryptographic (SimplePIR) | Hybrid argument |
| SANNS (USENIX Sec '20) | Semi-honest 2-party | Cryptographic (DORAM) | **Simulation-based 2PC** |
| Panther (CCS '25) | HBC single-server | Cryptographic (PIR+SS) | **Simulation-based single-server** |
| PPMI (arXiv '25) | 3-adversary, untrusted LLM + DB | Cryptographic (PIR) | IND-CPA |
| RemoteRAG (ACL '25) | HBC | Statistical (DP) + OT fallback | (n,ε)-DistanceDP |
| FedSQ (VLDB '24) | Semi-honest federated | Plaintext per owner | Inherited from CrypTen |
| SecureRAG (NeurIPS '25 ws) | 4-party, **trusted reader** | Pseudonymous IDs | IND-CPA + IND-sCPA |

### Where Opaque is genuinely stronger than these

- **Threshold key ownership.** No competitor distributes the decryption key.
  Compass, Pacmann, PPMI, SecureRAG all assume a single-client (or
  single-trusted-party) key. Opaque's t-of-N committee survives up to t-1
  compromises.
- **Operational simplicity.** Single server, no offline preprocessing
  (Pacmann), no per-client server state (Compass), no trusted reader role
  (SecureRAG), no non-colluding-server assumption (Servan-Schreiber et al.).
- **Latency at million scale.** Sub-second on commodity hardware (AWS c6i
  benchmarks).
- **At-rest defense in depth.** AES-GCM blob layer is a real second cryptographic
  boundary; competitor systems with single-key models lose everything on key
  compromise. (SecureRAG matches this with KP-ABE.)

### Where Opaque is genuinely weaker

- **Server threat model is HBC, not malicious.** Compass survives an active
  server; Opaque does not.
- **Access-pattern hiding is statistical, not cryptographic.** See §5.
- **Threshold layer carries known retry-vulnerability concerns.** See §3.
- **Not provably IND-CPA^D-128-secure.** See §2.

## 7. Reproducibility

All Opaque benchmarks are reproducible:
- AWS c6i.2xlarge / c6i.4xlarge: `bash deploy/bench-cpu/run_bench.sh c6i.2xlarge`
- Local SIFT1M: `go test -tags=sift1m -run TestSIFT1MAccuracy ./test/...`
- Threshold-mode benchmarks: see `docs/THRESHOLD_CKKS.md`

Hardware context for every published latency number is reported inline (no
hidden hardware effects).

## 8. Roadmap of Open Security Items

| Item | Status | Priority |
|---|---|---|
| σ flood 2^20 → 2^30 → 2^45 | **Done (commits `10b7850`, `e42338f`)** | — |
| LogQ chain restructure for provable IND-CPA^D-128 | **Done (commit `e42338f`)** | — |
| DecodePublic at all client-facing decryption sites | **Done (commit `10b7850`)** | — |
| Per-tenant blob ID permutation π | **Done (commit `bc0ec45`)** | — |
| Constant-volume padding (closes volume side-channel) | **Done (commit `414aa8e`)** | — |
| DP-grounded decoy mechanism with tunable ε | **Done (commit `990e2be`)** | — |
| DP formalization writeup in security model doc | **Done (this commit)** | — |
| No-retry invariant + fresh CRS per MHE protocol instance | Planned | High (Mouchet'24 / Colin de Verdière 2026) |
| Ephemeral-key rotation policy with bounded τ | **Done 2026-05-13** — `Engine.RotateKeys()` / `Committee.RotateEpoch()` + `DecryptCount()` + `SetRotationLimit(tau)` + `ShouldRotate()`. Caller-paced, default off. | Medium |
| Bernoulli per-cluster decoy sampling for tight (ε,δ)-DP | **Done** (commit landing now — `Config.BernoulliDecoys` flag, off by default) | Low |
| PIR backend as opt-in alternative to decoys | Deferred (statistical hiding + permutation + padding + tunable ε deemed sufficient for HBC; PIR remains research direction for Compass-tier malicious-server deployments) | Low |
| Pin Lattigo to v6.x line for security patches | Deferred (v6 is breaking; v5 doesn't bootstrap during search) | Low |
| Active-server integrity for cluster-index metadata | Not yet planned | Low (out of scope under HBC) |

## 9. References

- Lattigo SECURITY.md — https://github.com/tuneinsight/lattigo/blob/main/SECURITY.md
- Li & Micciancio, "On the Security of Homomorphic Encryption on Approximate Numbers", Eurocrypt 2021 — https://eprint.iacr.org/2020/1533
- Bergamaschi et al., "Revisiting the Security of Approximate FHE with Noise-Flooding Countermeasures", PKC 2025 — https://eprint.iacr.org/2024/424
- Checri, Sirdey, Boudguiga, Bultel, "On the Practical CPA^D Security of 'exact' and threshold FHE schemes and libraries", Crypto 2024 — https://eprint.iacr.org/2024/116
- Mouchet et al., "Multiparty Homomorphic Encryption from Ring-Learning-with-Errors", PoPETs 2021 — https://eprint.iacr.org/2020/304
- Mouchet et al., "Helium: Scalable MPC among Lightweight Participants and under Churn", CCS 2024 — https://eprint.iacr.org/2024/194
- Colin de Verdière, Passelègue, Stehlé, "On Threshold FHE with Synchronized Decryptors", 2026 — https://eprint.iacr.org/2026/031
- Cash, Grubbs, Perry, Ristenpart, "Leakage-Abuse Attacks Against Searchable Encryption", CCS 2015
- Oya, Kerschbaum, "Hiding the Access Pattern is Not Enough", USENIX Security 2021
- Compass — https://eprint.iacr.org/2024/1255
- Pacmann — https://eprint.iacr.org/2024/1600
- Tiptoe — https://eprint.iacr.org/2023/1438
- SANNS — https://www.usenix.org/conference/usenixsecurity20/presentation/chen-hao
- Panther — https://eprint.iacr.org/2024/1774
- SimplePIR / DoublePIR — https://eprint.iacr.org/2022/949
- YPIR — https://eprint.iacr.org/2024/270
