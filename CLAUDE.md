# Opaque — project guide for Claude

Privacy-preserving vector search using CKKS homomorphic encryption + AES-256-GCM
+ k-means clustering. Server computes on encrypted data and learns nothing.
Sub-second on 1M vectors, 8 vCPU AWS, 99.8% Recall@10.

This file is the entry point for any new Claude session. Skim everything below
before touching code. Cross-references point to deeper docs in `docs/`.

---

## Architecture in one minute

```
Client                                              Server
──────                                              ──────
1. Encrypt query under CKKS (Lattigo v5)    ──>     receive encrypted query
                                                    HE dot product vs PLAINTEXT centroids
                                                    (centroids are part of credentials,
                                                    public to authorized clients)
2. Receive encrypted scores                 <──     return encrypted scores
3. Decrypt scores → top-K logical cluster IDs
4. Translate logical → storage IDs via π    ──>     receive storage ID fetch list
                                                    (real top-K + decoys, server can't tell which)
5. Receive AES-GCM blobs                    <──     return blobs (real + padding, both look same)
6. AES-decrypt locally + score + return top-K
```

**Key insight:** HE protects the QUERY only. Centroids are plaintext-on-server
because the client already has them in credentials. AES protects the actual
vector data at rest. Decoys + permutation hide which clusters are accessed.

**Threshold mode (optional):** the CKKS key is split across a 3-of-5 committee
via Lattigo's `mhe` package. No single party can decrypt. ~0-10% latency overhead.

---

## File layout you actually need

| Path | What's there |
|---|---|
| `opaque.go` | Public API: `NewDB`, `Add`, `Build`, `Search`, `Config` |
| `pkg/crypto/crypto.go` | CKKS engine (`NewParameters`, `Engine`, HE dot product) |
| `pkg/crypto/threshold/threshold.go` | t-of-N threshold CKKS via Lattigo `mhe` |
| `pkg/crypto/threshold_provider.go` | `HEProvider` interface — direct or threshold |
| `pkg/encrypt/encrypt.go` | AES-256-GCM blob encryption |
| `pkg/hierarchical/kmeans_builder.go` | **Public API uses this builder** (not enterprise_builder) |
| `pkg/hierarchical/enterprise_builder.go` | Older path, used by some tests |
| `pkg/hierarchical/types.go` | `Config` for hierarchical search; `ComputeDecoyCountForEpsilon` |
| `pkg/hierarchical/index.go` | `ResolveDecoyCount` — applies TargetEpsilon override |
| `pkg/client/enterprise_hierarchical.go` | Search client; `translateToStorage` for π |
| `pkg/client/hierarchical.go` | `generateDecoySupers` (uniform K-from-non-selected) |
| `pkg/enterprise/config.go` | Per-tenant secrets: AESKey, BlobIDPermutation, PaddingMode |
| `pkg/auth/types.go` + `service.go` | ClientCredentials (incl. BlobIDPermutation) |
| `pkg/blob/memory.go` + `file.go` | Storage backends |
| `deploy/bench-cpu/run_bench.sh` | One-command AWS benchmark on c6i/m6i instance |
| `test/sift1m_benchmark_test.go` + `pq_sift1m_benchmark_test.go` | Headline benchmarks (`-tags=sift1m`) |
| `docs/SECURITY_MODEL.md` | **The canonical security writeup**; threat model, mitigations, ε bound |
| `docs/THRESHOLD_SECURITY.md` | Threshold-CKKS specifics (Li-Micciancio, σ, Mouchet retry) |
| `docs/LITERATURE_REVIEW.md` | Competitor analysis (Compass, Pacmann, PPMI, Tiptoe, SANNS, etc.) |
| `docs/ARCHITECTURE.md` | Architecture deep-dive |
| `docs/THRESHOLD_CKKS.md` | Threshold mode protocol + benchmarks |
| `deploy/bench-cpu/results/SUMMARY.md` | Canonical AWS bench history |
| `GUY_MEETING_NOTES.md` | Local memory — Telegram history with Guy Zyskind (Fhenix founder) |

---

## Current security state (2026-05)

Opaque ships with **all of these mitigations live by default in the public API**:

1. **σ=2^45 noise flooding + DecodePublic(logprec=10)** — provably 128-bit
   IND-CPA^D-secure under Bergamaschi PKC 2025 (eprint 2024/424). Closes the
   Li-Micciancio key-recovery attack (Eurocrypt 2021) and the Checri-Sirdey
   Crypto 2024 attack that targeted Lattigo by name.
2. **LogQ chain restructure** (8 primes → 5 primes, `LogDefaultScale=60`).
   Required for σ=2^45 — at the old scale=2^45, flooding noise = signal scale
   and destroyed output. New chain: `LogQ=[60,60,60,60,60]`, total Q+P=422 bits
   at LogN=14 → 128-bit RLWE security maintained.
3. **Per-tenant blob ID permutation π** — at index build, every enterprise
   gets a uniform random permutation π over storage IDs. Server can't link
   fetched blob → centroid coordinates. Closes semantic-cluster-ID inference
   and cross-tenant chosen-plaintext attacks. Zero query-time cost.
4. **Constant-volume padding** — `PaddingMode=None | Bucketed | MaxFixed`.
   Closes volume side-channel where unequal cluster sizes leak which fetched
   cluster is real. Bucketed = next-pow2-vector tier (~6-12% storage waste).
5. **TargetEpsilon-tunable decoys** — customer specifies ε directly; system
   derives `NumDecoys = ⌈(N - K_real) · e^(-ε)⌉`. Default ε=2.5 → ~10 decoys.
   Formal upper-bound on per-query distinguishability ratio (≤ e^ε). See
   `SECURITY_MODEL.md` §5.1 for the bound + caveats.

**Threat model:** honest-but-curious server, single-server. **NOT** Compass-tier
malicious-server — active tampering / reorder / drop / replay is not detected
today. See `SECURITY_MODEL.md` §6 for the full competitor comparison.

**Pending** (see `SECURITY_MODEL.md` §8 for the live list):
- No-retry invariant + fresh CRS per MHE protocol instance — closes
  Mouchet'24 / Okada'25 / Colin de Verdière 2026 retry attack on threshold mode
- Ephemeral key rotation for τ-bounded composition
- Bernoulli per-cluster decoy sampling for tight (ε,δ)-DP

---

## Where Opaque sits competitively

| System | Threat model | Access pattern | Latency | Recall | Notes |
|---|---|---|---|---|---|
| **Opaque (full mit)** | HBC | Statistical (DP-bound + π) | **464 ms** | **99.8%** | Threshold key (unique) |
| Compass (OSDI '25) | Malicious | Cryptographic (Ring ORAM) | ~600-900ms | high | Single-key, 500MB client mem |
| Pacmann (ICLR '25) | Semi-honest | Cryptographic (PIR) | ~3.1s | ~90% | 100M scale |
| Tiptoe (SOSP '23) | HBC | Cryptographic (PIR) | 2.7s | ~40% MS-MARCO | 360M pages |
| SANNS (USENIX '20) | 2-party HBC | Cryptographic (DORAM) | ~1.4s (72t) | ~90% | Heavy crypto stack |
| PPMI (arXiv '25) | 3-adversary | Cryptographic (PIR) | 951ms | >99% | Local trusted device required |

**Opaque's defensible angles:** sub-second latency, threshold key ownership
(no competitor has this), tunable privacy (ε), operational simplicity (1 server,
no per-client server state, KB client memory, shareable index).

**Opaque's weak spots:** statistical (not cryptographic) access-pattern hiding,
HBC (not malicious) server model, long-term ε composition over τ queries.

---

## Conventions

### Code

- **Go 1.25**. `lattigo/v5 v5.0.7`. `gonum/v1/gonum v0.17.0` for k-means.
- Public API in `opaque.go`. Internal packages in `pkg/`. Don't add to root.
- **Both `kmeans_builder.go` AND `enterprise_builder.go` are live code paths.**
  Public `opaque.NewDB` uses `kmeans_builder`. Some tests use
  `enterprise_builder`. Mitigations need to land in BOTH or one path silently
  bypasses them — this caused a bug in commit `bc0ec45` that took commit
  `e45223b` to fix.
- **`makeCredentials` in `opaque.go`** — easy to forget. When adding a new
  enterprise.Config field that must reach client search code, also pipe it
  through `makeCredentials()` (see commit `2368f0b` for what happens otherwise).
- DecodePublic(logprec=10) on every client-facing decryption site. Don't add
  raw `Decode` in production paths — it leaks residual noise.
- Tests with hardcoded expected values (e.g., `TestThresholdDecryptScalar`
  expects 0.52 ± 0.01) are the strongest correctness signal — preserve them
  when changing CKKS params.

### Commits

- Conventional Commits (`security:`, `fix:`, `docs:`, `bench:`, `feat:`).
- For security commits, cite the paper / attack closed in the message body.
- Always include `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` when Claude wrote the diff.

### Benchmarks

- **Reproducible AWS run:** `bash deploy/bench-cpu/run_bench.sh c6i.2xlarge`
  (8 vCPU 16 GB) or `m6i.2xlarge` (8 vCPU 32 GB — needed for full-mit + PQ).
- Cost: ~$0.35-1 per run, ~1-1.5 hr. Always destroyed at end via terraform trap.
- `c6i.2xlarge` will OOM during PQ codebook training with full mitigations live.
  **Use `m6i.2xlarge` for bench going forward.**
- Local SIFT1M data (`data/sift/`) is ignored by git and can be deleted —
  bench script downloads on demand on the EC2 instance.
- `setup.sh` tolerates missing `build-essential` (Opaque is pure Go; some
  Ubuntu AMIs ship without main-universe wired up).

### Recall sampling noise

- 50 queries → ±2-4 pp sampling noise on Recall@10. Don't over-interpret
  small recall deltas across runs as regressions. Tight tolerance check is
  the unit-test path with hardcoded expected values.
- 100% Recall@10 = exact-match against brute-force ground truth. If math
  silently broke, you'd see WRONG vectors → low recall, not 100%.

---

## Recent commit highlights (most recent first)

- `5b55d0d` bench: NC=256 SIFT1M variants for cluster-count tuning
- `9f0dc2f` docs: σ=2^45 + LogQ-restructure bench numbers
- `e42338f` security: LogQ chain restructure → σ=2^45 (provable IND-CPA^D-128)
- `cc1d140` docs: DP formalization writeup
- `990e2be` security: TargetEpsilon-tunable decoy count + DP-style upper bound
- `414aa8e` security: tunable constant-volume padding
- `bc0ec45` / `e45223b` / `2368f0b` security: per-tenant blob ID permutation π
  (and the kmeans_builder + makeCredentials wire-up bugs to know about)
- `10b7850` security: σ flood 2^20→2^30 + DecodePublic for Li-Micciancio
- `276b579` docs: SECURITY_MODEL.md + lit-review fixes for FedSQ/SecureRAG

## Pending work shipped behind incomplete-but-merged scaffolds

These pieces sit on `main` but have known follow-ups that the next session
should pick up. Both have full design docs / call-site comments — read those
first before extending.

- **DBpedia 1M bench** — Go scaffold landed (`pkg/embeddings/loader.go` +
  `test/dbpedia1m_benchmark_test.go` build-tag `dbpedia1m` + the HF parquet →
  fvecs converter `scripts/download_dbpedia1m.sh`). Fires via
  `deploy/bench-cpu/run_dbpedia_bench.sh`. Optional follow-ups: NC=256 variant
  if some future config justifies it (current 2026-05-09 SIFT bench shows
  NC=128 wins decisively, so don't pre-emptively add); larger M (M=192) if
  ada-002 PQ recall is too lossy.
- **Threshold retry-attack fix** — Phase 1 only (`docs/THRESHOLD_RETRY_FIX.md`
  + `pkg/crypto/threshold/retry_guard.go` standalone). Phases 2 + 3 pending —
  wiring `RetryGuard` into `ThresholdDecrypt` (API change: needs `instanceID`
  parameter) and per-round CRS for keygen + coordinator abort state machine.
  See the design doc §3 for the phased plan.

## Tested-and-rejected variations (don't re-test)

- **NC=256 vs NC=128 (2026-05-09 m6i.2xlarge bench).** Hypothesis was that
  smaller per-cluster blobs would cut fetch + local-scoring time and win
  latency at equal recall. Disproven: NC=256 is **60-67 % slower** at equal
  recall on both no-PQ and PQ paths because doubling cluster count doubles
  level-1 HE scoring cost (256 vs 128 centroid HE dot products on the
  encrypted query) and that dominates the small fetch+local saving. Build
  time also worsens by ~40 %. Tests `TestSIFT1MAccuracy_NC256` +
  `TestPQ_SIFT1M_NC256` are kept in the tree as documentation and as a
  guard against future hand-wavy "what if we tried more clusters?"
  suggestions — see `deploy/bench-cpu/results/SUMMARY.md` 2026-05-09 section
  for the full apples-comparison table.

---

## Working with the user (Prasad)

- Senior Go / systems engineer building Opaque as a research-leaning project
  with paper-publication aspirations. Founder of Realfy professionally.
- Prefers terse, technical responses. Caveman mode is sometimes active —
  follow it; drop it for security-critical / multi-step content where fragment
  order risks misread.
- **Wants regular commits, not one giant end-of-session commit.**
- Pushes back on bad recommendations. If he questions a number or premise,
  re-derive it before defending — he's caught real bugs that way (e.g., the
  ε=3 default was actually weaker than baseline; I had it wrong).
- For external comms (Telegram to Guy, portfolio copy): show-off tone, no
  attack-detail / Li-Micciancio / σ specifics. Specifics belong in commits +
  `docs/`. The portfolio `/opaque` page is a marketing surface, not a security
  writeup.
- AWS personal profile is configured. Use `bash deploy/bench-cpu/run_bench.sh`
  for benchmarks, not local M4 (system contention pollutes numbers).

---

## Open contact threads

- **Guy Zyskind** (Fhenix founder, PRAG co-author with Alex Pentland) — last
  Telegram round 23 April. Asked about Compass model. Reply drafted in a prior
  session but was waiting on the σ=2^45 numbers. See `GUY_MEETING_NOTES.md`.
- M1 demo (`docs/M1_DEMO_PLAN.md`) — symptom-to-condition search planned
  for `prasadjs.me/opaque`. Not yet built. ~$30-70/month always-on backend.

---

## When in doubt

1. Read `docs/SECURITY_MODEL.md` first for any security-touching change.
2. Run unit tests + e2e + threshold tests before any commit:
   ```
   go test ./pkg/crypto/... ./pkg/client/... ./pkg/hierarchical/... \
     ./pkg/enterprise/... ./pkg/auth/... . -count=1 -timeout 240s -short
   ```
3. For mitigation changes touching the search path, **also** verify on AWS
   via the bench script — m6i.2xlarge is the canonical 8-vCPU+enough-RAM tier.
4. Don't push to GitHub unless asked; commits are local until the user says
   "push" / "deploy" / similar.
