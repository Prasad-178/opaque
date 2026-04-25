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
`σ = 2^30` before partial decryption (`pkg/crypto/threshold/threshold.go:237`)
and then routes every plaintext released to the client through
`DecodePublic(_, _, logprec=10)` (rounds output to 2^-10 ≈ 1e-3 precision).

These two mitigations compose to defeat the Li-Micciancio key-recovery attack
on CKKS approximate decryption (Eurocrypt 2021, eprint 2020/1533). Lattigo's
own SECURITY.md documents this mitigation pattern.

The mitigation is **not provably IND-CPA^D-128-secure** under Bergamaschi et
al. PKC 2025 (eprint 2024/424), which would require σ ≈ 2^45 with an enlarged
LogQ chain (current chain has scale 2^45 per prime, which makes σ=2^45 destroy
signal). Restructuring LogQ to enable provable σ=2^45 is on the roadmap.

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
- **Planned key rotation** (roadmap item) to bound τ explicitly.

Without explicit key rotation, a long-running deployment with many queries
under the same key will slowly accumulate the attack surface.

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

### Mitigation in roadmap: PIR backend

Opaque plans an opt-in PIR backend (SimplePIR-style, single-server, LWE-based)
as an alternative to decoy clusters. Customers and deployments with a stronger
threat model select PIR; the default decoy backend remains for latency-sensitive
deployments. See `docs/PIR_DESIGN.md` (TBD) for the PIR architecture.

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
| σ flood 2^20 → 2^30 | **Done (commit 10b7850)** | — |
| DecodePublic at all client-facing decryption sites | **Done (commit 10b7850)** | — |
| PIR backend as opt-in alternative to decoys | In progress | High (story-level) |
| Provable IND-CPA^D-128 via LogQ chain restructure → σ=2^45 | Planned | Medium |
| No-retry invariant + fresh CRS per MHE protocol instance | Planned | High (Mouchet'24 / Colin de Verdière 2026) |
| Ephemeral-key rotation policy with bounded τ | Planned | Medium |
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
