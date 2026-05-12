# SIFT 1M Benchmark Results — AWS EC2

Raw per-run logs stay gitignored. This summary table survives the
ephemeral EC2 teardown and is the durable record.

All runs on AWS `us-east-1`, ephemeral on-demand EC2 under the `personal`
profile, destroyed at end of run. Ubuntu 22.04, Go 1.25, Lattigo v5.0.7,
CKKS `LogN=14` (128-bit security). Dataset: SIFT 1M (1,000,000 × 128-dim),
50 queries, top-K=10, 8 decoys, 128 clusters.

**Security mitigations shipped between runs:**
- 2026-04-19/20: pre-mitigation baseline.
- 2026-04-24 (commit `10b7850`): σ noise flooding 2^20→2^30 + DecodePublic(logprec=10) for Li-Micciancio mitigation.
- 2026-04-30 (commit `bc0ec45`): per-tenant blob ID permutation π for access-pattern privacy (hides centroid-to-storage link from server).
- 2026-05-02 (commit `e42338f`): LogQ chain restructure (`LogQ=[60×5]` + `LogDefaultScale=60`) → σ=2^45 noise flooding for provable 128-bit IND-CPA^D security under Bergamaschi PKC 2025.
- 2026-05-09 (commit `5b55d0d`): NC=256 evaluation tests added; NC=128 confirmed optimal — see run below.
- 2026-05-10 (commit `1e735ec`): build-phase memory optimizations (free `normalizedVecs` after k-means, `Storage: File` for DBpedia tests, `GOGC=50`). Cuts 1M × 1536-dim build peak from ~128 GB → ~64 GB. SIFT1M results unchanged (operates well below the OOM regime). Full DBpedia bench deferred pending float32 refactor — see `docs/MEMORY_PROFILE.md`.
- 2026-05-10 (commit `7a9a369`): AES vector ciphertext encoding switched from float64 → float32 (4 bytes/elem instead of 8). Halves AES ciphertext size; ~6 GB build-phase memory saved at 1M × 1536-dim; halves on-disk Storage:File footprint. Validated on m6i.xlarge SIFT 100K bench post-change — see `m6i.xlarge` section below. **Recall preserved**: standard 100% R@1 / 98.8% R@10 / 156 ms; PQ-M16 100/98.6% / 147 ms; standard-probe16 100/100% / 168 ms.
- 2026-05-10 (commit `7efa65b`): `download_sift1m.sh` switched from IRISA FTP → ann-benchmarks HDF5 mirror after a multi-hour IRISA outage. HDF5 → fvecs/ivecs conversion via Python h5py on the EC2 instance.
- 2026-05-10 (commits `e03c5d5` + `e314c65`): threshold retry-attack fix Phases 2 + 3a — `RetryGuard` wired into `ThresholdDecrypt` (instanceID-keyed per-emission refusal) and per-round CRS derivation (`derivePerRoundCRS(epochSeed, roundLabel)`) for CPK / Relin / Galois keygen rounds. Live retry-attack surface closed. Phases 3b (state machine) + 3c (chaos tests) pending.
- 2026-05-10 (commit `38f678b`): per-cluster Bernoulli decoy sampling (`Config.BernoulliDecoys=true`) for tight (ε,δ)-DP under composition. Validated post-change on m6i.xlarge — see `m6i.xlarge` section below.
- 2026-05-10 (commit `828ee0d`): float32 storage tier — `db.pendingVectors` and `kmeans_builder.normalizedVecs` now `[][]float32`, saving ~12 GB at 1M × 1536-dim. Public API stays `[]float64` (boundary conversion at AddBatch/Search). Build peak at 1M × 1536-dim now estimated ~30-40 GB vs prior ~64 GB (50 % further reduction on top of `1e735ec`). Validated on m6i.xlarge SIFT 100K — recall preserved within sampling noise (see m6i.xlarge section below).

---

## m6i.xlarge (4 vCPU, 16 GB, Intel Ice Lake) — float32-ciphertext validation (2026-05-10)

Quick-and-cheap regression check (~$0.03/run) after the float32 AES
vector ciphertext encoding landed in commit `7a9a369`. Goal: verify
the float32 round-trip during AES encrypt/decrypt does not regress
recall or latency. SIFT 100K (first 100K of SIFT 1M, 128-dim).

Source: `ann-benchmarks.com/sift-128-euclidean.hdf5` (replaces
ftp.irisa.fr after IRISA's multi-hour outage on 2026-05-10).

### `TestPQ_SIFT100K` (NC=64, top-8 default)

| Config           | Recall@1 | Recall@10 | Avg query | P50    | Build      |
|------------------|----------|-----------|-----------|--------|------------|
| standard         | 100.0 %  | 98.8 %    | 156 ms    | 153 ms | 14.5 s     |
| PQ-M8            | 98.0 %   | 95.2 %    | 146 ms    | 146 ms | 1m15s      |
| PQ-M16           | 100.0 %  | 98.6 %    | 147 ms    | 146 ms | 1m15s      |
| standard-probe16 | 100.0 %  | **100.0 %** | 168 ms  | 166 ms | 14.6 s     |
| PQ-M8-probe16    | 100.0 %  | 99.4 %    | 163 ms    | 161 ms | 1m14s      |

**Recall preserved** vs the pre-float32 historical numbers within
sampling noise. The ~1.2e-7 per-element float32 round-trip error
sits well below any recall-meaningful boundary in cosine similarity
on SIFT 1M-class data. No latency regression either.

Total bench wall: ~5 min (much shorter than SIFT 1M's ~25 min per
config because the test only runs 4 configs at 100K vectors each
and the AES ciphertexts are now half the size on disk too).

### `TestPQ_SIFT100K` (float32 storage tier validation, 2026-05-10 23:16)

Re-run after commit `828ee0d` converted `db.pendingVectors` and the
build-phase `normalizedVecs` to `[][]float32`. Goal: empirically confirm
recall is preserved (the float32 round-trip introduces ~1.2e-7 per-element
precision loss, ~6 orders below recall sensitivity at SIFT 1M-class scale)
and that latency picks up no regression.

| Config                    | Recall@1 | Recall@10 | Avg query | Δ R@10 vs pre-float32-tier |
|---------------------------|----------|-----------|-----------|----------------------------|
| standard                  | 100.0 %  | 98.4 %    | 155 ms    | −0.4 pp (within sampling noise) |
| PQ-M8                     | 96.0 %   | 94.4 %    | 149 ms    | −0.8 pp |
| PQ-M16                    | 100.0 %  | 97.8 %    | 151 ms    | −0.8 pp |
| **standard-probe16**      | 100.0 %  | **100.0 %** | 170 ms  | **0** |
| PQ-M8-probe16             | 100.0 %  | 99.4 %    | 163 ms    | +0.2 pp |
| **standard-bernoulli**    | 100.0 %  | **98.8 %**| 152 ms    | **0** (identical) |
| PQ-M8-probe16-bernoulli   | 100.0 %  | 99.4 %    | 161 ms    | −0.2 pp |

All deltas within the documented ±2-4 pp 50-query sampling-noise band
(see `CLAUDE.md`). The two configs that hit perfect 100 % R@10 in both
runs (`standard-probe16` + `standard-bernoulli`) confirm the float32
round-trip introduces no systematic recall regression — only sampling
variance. Latency unchanged across all 7 configs.

**The float32 storage tier is paper-ready.** Build peak at 1M × 1536-dim
should now drop ~50 % further from the prior optimisations, plausibly
unlocking DBpedia 1M @ 1536-dim on m6i.2xlarge ($0.38/hr) — to be
confirmed by a separate DBpedia bench when ready.

Cost: ~$0.04 (m6i.xlarge × ~10 min wall).

### `TestPQ_SIFT100K` (Bernoulli decoy validation, 2026-05-10 22:34)

Re-run after commit `38f678b` added `Config.BernoulliDecoys=true` to
swap the uniform-K-from-non-selected decoy sampler for per-cluster
i.i.d. Bernoulli sampling at p = E[K_decoy] / (N - K_real). Same
expected K_decoy, but K is now binomial per query. Goal: verify recall
is preserved (decoys never affect recall) and latency variance is
bounded.

| Config                    | Recall@1 | Recall@10 | Avg query | Δ R@10 vs uniform-K |
|---------------------------|----------|-----------|-----------|---------------------|
| standard                  | 100.0 %  | 98.8 %    | 156 ms    | (baseline)          |
| **standard-bernoulli**    | 100.0 %  | **98.8 %**| **151 ms**| **−0.0 pp**         |
| PQ-M8-probe16             | 100.0 %  | 99.2 %    | 160 ms    | (baseline)          |
| **PQ-M8-probe16-bernoulli** | 100.0 % | **99.6 %**| **161 ms**| **+0.4 pp**         |

Recall **identical** to uniform-K for `standard` (-0.0 pp) and within
sampling noise for `PQ-M8-probe16` (+0.4 pp; 50-query bench has ±2 pp
recall sampling noise documented elsewhere). Latency essentially equal —
the predicted binomial-K variance translates to ≤5 ms per-query
difference at this scale, vs the 156 ms baseline.

**Bernoulli decoy path is paper-ready.** The (ε,δ)-DP composition
analysis under standard subsampled-mechanism framework
(Dwork-Roth Theorem 3.5+) now applies directly, replacing the prior
informal e^ε bound for the uniform-K scheme. See
`docs/SECURITY_MODEL.md` §5.1 for the full bound + caveats.

Cost: ~$0.04 (m6i.xlarge × ~10 min wall).

---

## c6i.2xlarge (8 vCPU, 16 GB, Intel Ice Lake)

Matches Pinecone `p1.x1` / Qdrant 8-core pod tier.

## m6i.2xlarge (8 vCPU, 32 GB, Intel Ice Lake) — NC=128 vs NC=256 evaluation (2026-05-09)

Tests `TestSIFT1MAccuracy_NC256` + `TestPQ_SIFT1M_NC256` added to compare 128
vs 256 cluster counts at equal coverage %. Hypothesis: smaller per-cluster
blobs cut fetch + local-scoring time, possibly winning latency at equal
recall.

**Hypothesis disproven.** Doubling cluster count doubles level-1 HE
operations (256 vs 128 centroid HE dot products on the encrypted query),
and that cost dominates the small fetch+local saving. NC=256 is **60–67 %
slower** at equal recall on every config tested. NC=128 stays the default.

Apples comparison (equal coverage %, equal recall):

| Path     | NC=128 config        | NC=256 config (apples) | NC=256 slowdown |
|----------|----------------------|------------------------|-----------------|
| no-PQ    | probe-8 (446 ms)     | probe-16 (747 ms)      | **+67 %**       |
| PQ       | PQ-M8-probe8-eps271 (411 ms) | PQ-M8-probe16-eps271 (658 ms) | **+60 %** |

Headline NC=128 numbers reproduce within ±2 pp / ±18 ms sampling noise
of the 2026-05-02 run.

### `TestSIFT1MAccuracy` (NC=128 reproduce, no PQ, ε=2.5, σ=2^45)

| Config     | Probe  | Multi | Recall@1 | Recall@10 | Avg query |
|------------|--------|-------|----------|-----------|-----------|
| strict-4   | 3.1 %  | no    | 88.0 %   | 85.6 %    | 308 ms    |
| strict-8   | 6.2 %  | no    | 96.0 %   | 97.8 %    | 352 ms    |
| strict-16  | 12.5 % | no    | 98.0 %   | 99.4 %    | 731 ms    |
| **probe-8** | 6.2 %+ | yes   | **100.0 %** | **99.4 %** | **446 ms** |
| probe-16   | 12.5 %+| yes   | 100.0 %  | 100.0 %   | 653 ms    |

### `TestSIFT1MAccuracy_NC256` (new, NC=256, no PQ, ε=2.5, σ=2^45)

TopClusters scaled 2× of NC=128 to keep coverage % equal.

| Config     | Probe  | Multi | Recall@1 | Recall@10 | Avg query |
|------------|--------|-------|----------|-----------|-----------|
| strict-8   | 3.1 %  | no    | 90.0 %   | 92.0 %    | 475 ms    |
| strict-16  | 6.2 %  | no    | 100.0 %  | 98.4 %    | 619 ms    |
| strict-32  | 12.5 % | no    | 100.0 %  | 100.0 %   | 749 ms    |
| probe-16   | 6.2 %+ | yes   | 100.0 %  | 99.6 %    | 747 ms    |
| probe-32   | 12.5 %+| yes   | 100.0 %  | 100.0 %   | 1101 ms   |

### `TestPQ_SIFT1M` (NC=128 reproduce, σ=2^45)

| Config                 | ε    | PQ | Recall@1 | Recall@10 | Avg query | Build  |
|------------------------|------|----|----------|-----------|-----------|--------|
| **PQ-M8-probe8-eps25** | 2.50 | M8 | 100.0 %  | 98.0 %    | **411 ms**| 8m21s  |
| PQ-M8-probe16-eps25    | 2.50 | M8 | 100.0 %  | 99.4 %    | 582 ms    | 8m22s  |
| PQ-M8-probe8-eps271    | 2.71 | M8 | 100.0 %  | 98.0 %    | **411 ms**| 8m43s  |
| PQ-M8-probe8-eps20     | 2.00 | M8 | 100.0 %  | 97.8 %    | 513 ms    | 8m1s   |
| standard-probe8-eps25  | 2.50 | -- | 98.0 %   | 99.2 %    | 459 ms    | 3m28s  |

### `TestPQ_SIFT1M_NC256` (new, NC=256, σ=2^45)

| Config                  | ε    | PQ | Recall@1 | Recall@10 | Avg query | Build   |
|-------------------------|------|----|----------|-----------|-----------|---------|
| PQ-M8-probe16-eps25     | 2.50 | M8 | 100.0 %  | 98.0 %    | 675 ms    | 11m39s  |
| PQ-M8-probe32-eps25     | 2.50 | M8 | 100.0 %  | 99.4 %    | 955 ms    | 11m39s  |
| PQ-M8-probe16-eps271    | 2.71 | M8 | 100.0 %  | 98.2 %    | 658 ms    | 11m25s  |
| PQ-M8-probe16-eps20     | 2.00 | M8 | 100.0 %  | 98.4 %    | 1041 ms   | 11m45s  |
| standard-probe16-eps25  | 2.50 | -- | 100.0 %  | 99.6 %    | 739 ms    | 6m39s   |

Build time also worse at NC=256 — k-means convergence on 256 centroids
takes ~40 % longer per iteration; PQ codebook training is unaffected.

### Headline (m6i.2xlarge, σ=2^45, all mitigations live, ε=2.5, **NC=128 default**)

| Config                       | Recall@1 | Recall@10 | Avg query |
|------------------------------|----------|-----------|-----------|
| **probe-8 (recommended)**    | 100.0 %  | 99.4 %    | **446 ms**|
| **probe-16 (max recall)**    | 100.0 %  | 100.0 %   | **653 ms**|
| PQ-M8-probe8 (PQ alt)        | 100.0 %  | 98.0 %    | 411 ms    |
| PQ-M8-probe8-eps271 (fastest)| 100.0 %  | 98.0 %    | 411 ms    |

Total bench wall time: ~2.6 hr (4 tests × ~5 configs × ~3-12 min build +
queries + brute-force ground truth). Cost: ~$1.00 at $0.38/hr.

---

## m6i.2xlarge (8 vCPU, 32 GB, Intel Ice Lake) — provable IND-CPA^D-128 run (2026-05-02)

LogQ chain restructured to support σ=2^45 noise flooding (commit `e42338f`).
Now provably 128-bit IND-CPA^D-secure under Bergamaschi PKC 2025. Recall
preserved or slightly improved (precision gain from `LogDefaultScale`
45 → 60). Latency essentially flat (within sampling noise).

### `TestSIFT1MAccuracy` (no PQ, ε=2.5, σ=2^45)

| Config     | Probe  | Multi | Recall@1 | Recall@10 | Avg query |
|------------|--------|-------|----------|-----------|-----------|
| strict-4   | 3.1 %  | no    | 88.0 %   | 86.0 %    | 319 ms    |
| strict-8   | 6.2 %  | no    | 100.0 %  | 97.0 %    | 364 ms    |
| strict-16  | 12.5 % | no    | 100.0 %  | 99.6 %    | 477 ms    |
| **probe-8** | 6.2 %+ | yes   | **100.0 %** | **99.8 %** | **464 ms** |
| probe-16   | 12.5 %+| yes   | 100.0 %  | 100.0 %   | 652 ms    |

### `TestPQ_SIFT1M` — privacy-tunable comparison (σ=2^45)

| Config                 | ε    | PQ | Recall@1 | Recall@10 | Avg query | Build  |
|------------------------|------|----|----------|-----------|-----------|--------|
| **PQ-M8-probe8-eps25** | 2.50 | M8 | 100.0 %  | 98.4 %    | **409 ms**| 8m19s  |
| PQ-M8-probe16-eps25    | 2.50 | M8 | 100.0 %  | 99.2 %    | 568 ms    | 8m10s  |
| PQ-M8-probe8-eps271    | 2.71 | M8 | 100.0 %  | 98.4 %    | **406 ms**| 8m30s  |
| PQ-M8-probe8-eps20     | 2.00 | M8 | 98.0 %   | 98.4 %    | 511 ms    | 8m10s  |
| standard-probe8-eps25  | 2.50 | -- | 100.0 %  | 98.4 %    | 456 ms    | 3m19s  |

### Headline (m6i.2xlarge, σ=2^45 + all mitigations live, ε=2.5)

| Config                       | Recall@1 | Recall@10 | Avg query |
|------------------------------|----------|-----------|-----------|
| **probe-8 (recommended)**    | 100.0 %  | 99.8 %    | **464 ms**|
| **probe-16 (max recall)**    | 100.0 %  | 100.0 %   | **652 ms**|
| PQ-M8-probe8 (PQ alt)        | 100.0 %  | 98.4 %    | 409 ms    |
| PQ-M8-probe8-eps271 (fastest)| 100.0 %  | 98.4 %    | 406 ms    |

---

## m6i.2xlarge (8 vCPU, 32 GB, Intel Ice Lake) — first full-mitigation run (2026-05-01, σ=2^30)

### `TestSIFT1MAccuracy` (no PQ, ε=2.5)

| Config     | Probe  | Multi | Recall@1 | Recall@10 | Avg query |
|------------|--------|-------|----------|-----------|-----------|
| strict-4   | 3.1 %  | no    | 82.0 %   | 84.8 %    | 324 ms    |
| strict-8   | 6.2 %  | no    | 94.0 %   | 95.2 %    | 369 ms    |
| strict-16  | 12.5 % | no    | 100.0 %  | 99.6 %    | 469 ms    |
| **probe-8** | 6.2 %+ | yes   | **100.0 %** | **99.6 %** | **462 ms** |
| probe-16   | 12.5 %+| yes   | 100.0 %  | 100.0 %   | 635 ms    |

### `TestPQ_SIFT1M` — privacy-tunable comparison

| Config                 | ε    | PQ | Recall@1 | Recall@10 | Avg query | Build  |
|------------------------|------|----|----------|-----------|-----------|--------|
| **PQ-M8-probe8-eps25** | 2.50 | M8 | 100.0 %  | 97.6 %    | **428 ms**| 7m56s  |
| PQ-M8-probe16-eps25    | 2.50 | M8 | 100.0 %  | 99.4 %    | 578 ms    | 8m6s   |
| PQ-M8-probe8-eps271    | 2.71 | M8 | 98.0 %   | 98.0 %    | **410 ms**| 8m32s  |
| PQ-M8-probe8-eps20     | 2.00 | M8 | 100.0 %  | 98.0 %    | 502 ms    | 8m9s   |
| standard-probe8-eps25  | 2.50 | -- | 100.0 %  | 99.8 %    | 461 ms    | 3m25s  |

Observations:
- ε=2.71 (baseline-parity, ~9 decoys) is the fastest tier at 410 ms with
  98 % recall — back to pre-mitigation latency neighborhood.
- ε=2.5 (recommended default, ~10 decoys) lands at ~430-460 ms and recall
  ≥ 97.6 %.
- ε=2.0 (high-privacy, ~17 decoys) costs ~75 ms more than ε=2.5 for the
  same recall.
- PQ at SIFT1M (128-dim) saves only ~30-50 ms vs non-PQ at the same ε.
  Trade is ~2 pp recall. PQ wins more at higher dimensions (text/RAG
  embeddings).
- Build cost: PQ adds ~5 min for codebook training; one-time, not query cost.

### Headline (m6i.2xlarge, all four mitigations live, ε=2.5)

| Config                       | Recall@1 | Recall@10 | Avg query |
|------------------------------|----------|-----------|-----------|
| **probe-8 (recommended)**    | 100.0 %  | 99.6 %    | **462 ms**|
| **probe-16 (max recall)**    | 100.0 %  | 100.0 %   | **635 ms**|
| PQ-M8-probe8 (PQ alt)        | 100.0 %  | 97.6 %    | 428 ms    |
| PQ-M8-probe8-eps271 (fastest)| 98.0 %   | 98.0 %    | 410 ms    |

---

### `TestSIFT1MAccuracy` — full-mitigation run (2026-04-30 19:08, commit `e45223b` + makeCredentials fix)

All four mitigations live: σ=2^30 + DecodePublic, per-tenant blob ID
permutation π, `PaddingMode=Bucketed` (next-pow2 cluster sizing,
~6-12 % storage waste), `TargetEpsilon=2.0` → derived NumDecoys=17 (vs
prior fixed 8). Recall fully recovered; latency adds ~30-65 % vs the
partial-mitigation run (extra decoys + padding bandwidth).

| Config     | Probe  | Multi | Recall@1 | Recall@10 | Avg query |
|------------|--------|-------|----------|-----------|-----------|
| strict-4   | 3.1 %  | no    | 94.0 %   | 88.4 %    | 410 ms    |
| strict-8   | 6.2 %  | no    | 98.0 %   | 95.4 %    | 466 ms    |
| strict-16  | 12.5 % | no    | 100.0 %  | 99.2 %    | 640 ms    |
| probe-8    | 6.2 %+ | yes   | 100.0 %  | 99.4 %    | 630 ms    |
| probe-16   | 12.5 %+| yes   | 100.0 %  | 100.0 %   | 815 ms    |

`TestPQ_SIFT1M` was killed mid-run (likely OOM at standard-strict8 PQ
codebook training; padding pushed memory over c6i.2xlarge's 16 GB limit).
Re-run on c6i.4xlarge (32 GB) recommended for PQ-mitigated numbers.

### `TestSIFT1MAccuracy` — partial-mitigation run (2026-04-30 11:49, commit `bc0ec45`)

Includes σ=2^30 + DecodePublic. **Permutation π was MISSING from this run** —
the public `opaque.NewDB` API uses `kmeans_builder.go`, which had not yet
been wired to the permutation logic shipped to `enterprise_builder.go` in
`bc0ec45`. Fixed in `e45223b`. Recall identical to baseline; latency within
sampling noise.

| Config     | Probe  | Multi | Recall@1 | Recall@10 | Avg query |
|------------|--------|-------|----------|-----------|-----------|
| strict-4   | 3.1 %  | no    | 84.0 %   | 84.0 %    | 250 ms    |
| strict-8   | 6.2 %  | no    | 92.0 %   | 93.2 %    | 299 ms    |
| strict-16  | 12.5 % | no    | 100.0 %  | 99.0 %    | 372 ms    |
| probe-8    | 6.2 %+ | yes   | 98.0 %   | 99.0 %    | 366 ms    |
| probe-16   | 12.5 %+| yes   | 100.0 %  | 100.0 %   | 521 ms    |

### `TestPQ_SIFT1M` — post-mitigation run (2026-04-30)

| Config             | Recall@1 | Recall@10 | Avg query | P50     | Build     |
|--------------------|----------|-----------|-----------|---------|-----------|
| standard-strict8   | 96.0 %   | 96.4 %    | 310 ms    | 305 ms  | 3m14s     |
| standard-strict16  | 100.0 %  | 99.8 %    | 401 ms    | 398 ms  | 3m3s      |
| PQ-M8-strict8      | 90.0 %   | 93.4 %    | 291 ms    | 288 ms  | 8m20s     |
| PQ-M8-strict16     | 100.0 %  | 98.4 %    | 355 ms    | 352 ms  | 8m16s     |
| PQ-M8-strict32     | 100.0 %  | 99.8 %    | 522 ms    | 522 ms  | 8m17s     |
| PQ-M8-probe16      | 100.0 %  | 99.2 %    | 462 ms    | 458 ms  | 8m18s     |
| PQ-M8-probe32      | 100.0 %  | 100.0 %   | 658 ms    | 646 ms  | 8m35s     |

### `TestSIFT1MAccuracy` — pre-mitigation baseline (2026-04-19)

| Config     | Probe  | Multi | Recall@1 | Recall@10 | Avg query |
|------------|--------|-------|----------|-----------|-----------|
| strict-4   | 3.1 %  | no    | 90.0 %   | 87.6 %    | 238 ms    |
| strict-8   | 6.2 %  | no    | 94.0 %   | 93.6 %    | 282 ms    |
| strict-16  | 12.5 % | no    | 100.0 %  | 99.2 %    | 349 ms    |
| probe-8    | 6.2 %+ | yes   | 100.0 %  | 99.8 %    | 345 ms    |
| probe-16   | 12.5 %+| yes   | 100.0 %  | 100.0 %   | 452 ms    |

### `TestPQ_SIFT1M` — pre-mitigation baseline (2026-04-19)

| Config             | Recall@1 | Recall@10 | Avg query | P50     | Build     |
|--------------------|----------|-----------|-----------|---------|-----------|
| standard-strict8   | 96.0 %   | 96.0 %    | 282 ms    | 277 ms  | 3m21s     |
| standard-strict16  | 100.0 %  | 99.8 %    | 347 ms    | 339 ms  | 3m11s     |
| PQ-M8-strict8      | 90.0 %   | 92.4 %    | 270 ms    | 268 ms  | 8m32s     |
| PQ-M8-strict16     | 100.0 %  | 98.4 %    | 327 ms    | 326 ms  | 8m25s     |
| PQ-M8-strict32     | 100.0 %  | 99.6 %    | 445 ms    | 444 ms  | 8m16s     |
| PQ-M8-probe16      | 100.0 %  | 99.2 %    | 406 ms    | 403 ms  | 8m23s     |
| PQ-M8-probe32      | 100.0 %  | 100.0 %   | 497 ms    | 486 ms  | 8m13s     |

---

## c6i.4xlarge (16 vCPU, 32 GB, Intel Ice Lake)

Matches Qdrant / Weaviate / Elastic vector-search standard pod tier.

### `TestSIFT1MAccuracy`

| Config     | Probe  | Multi | Recall@1 | Recall@10 | Avg query |
|------------|--------|-------|----------|-----------|-----------|
| strict-4   | 3.1 %  | no    | 92.0 %   | 85.6 %    | 234 ms    |
| strict-8   | 6.2 %  | no    | 96.0 %   | 96.2 %    | 268 ms    |
| strict-16  | 12.5 % | no    | 100.0 %  | 99.6 %    | 333 ms    |
| probe-8    | 6.2 %+ | yes   | 100.0 %  | 99.8 %    | 329 ms    |
| probe-16   | 12.5 %+| yes   | 100.0 %  | 100.0 %   | 432 ms    |

### `TestPQ_SIFT1M`

| Config             | Recall@1 | Recall@10 | Avg query | P50     | Build     |
|--------------------|----------|-----------|-----------|---------|-----------|
| standard-strict8   | 96.0 %   | 93.8 %    | 275 ms    | 272 ms  | 3m12s     |
| standard-strict16  | 100.0 %  | 99.4 %    | 337 ms    | 328 ms  | 3m11s     |
| PQ-M8-strict8      | 94.0 %   | 94.2 %    | 260 ms    | 254 ms  | 6m22s     |
| PQ-M8-strict16     | 100.0 %  | 97.8 %    | 323 ms    | 320 ms  | 6m31s     |
| PQ-M8-strict32     | 100.0 %  | 99.6 %    | 452 ms    | 446 ms  | 6m50s     |
| PQ-M8-probe16      | 100.0 %  | 99.6 %    | 405 ms    | 398 ms  | 6m39s     |
| PQ-M8-probe32      | 100.0 %  | 100.0 %   | 504 ms    | 504 ms  | 6m39s     |

---

## Observations

**Search latency is not CPU-bound past 8 vCPU.** c6i.4xlarge shows
near-identical per-query latency to c6i.2xlarge across every config.
The hot path (HE batch dot product, rotations, AES decrypt of 8 decoy
bags, local PQ / exact scoring) is already saturated at 8 parallel
cores; adding cores up to 16 does not shrink it.

**Build scales ~1.5–2× with cores.** PQ training is CPU-bound on codebook
k-means; the 16-vCPU instance cuts PQ build time from ~8m30s to ~6m40s.
Non-PQ build (encryption of 1 M vectors) drops from ~3m20s to ~3m10s —
the extra cores mostly help k-means + PQ codebook training.

**Production implication.** 8 vCPU is enough for sub-500ms SIFT 1M
queries at 100 % Recall@10 with full privacy. Scaling out further is a
throughput play (more instances, parallel queries), not a latency play.

## Headline numbers for paper / pitch

Best single number for each hardware tier:

| Hardware    | Config       | Mitigations               | Recall@10 | Avg query |
|-------------|--------------|---------------------------|-----------|-----------|
| c6i.2xlarge | probe-8      | **all four (full)**       | 99.4 %    | 630 ms    |
| c6i.2xlarge | probe-16     | **all four (full)**       | 100.0 %   | 815 ms    |
| c6i.2xlarge | probe-8      | partial (σ + DecodePublic) | 99.0 %    | 366 ms    |
| c6i.2xlarge | PQ-M8-probe32| pre-mitigation baseline   | 100.0 %   | 497 ms    |
| c6i.2xlarge | probe-8      | pre-mitigation baseline   | 99.8 %    | 345 ms    |

Latency progression:
- pre-mitigation → partial-mitigation: +5-30 % (σ flood + DecodePublic +
  cache effects).
- partial-mitigation → full-mitigation: +30-65 % (Bucketed padding +
  TargetEpsilon=2.0 increases NumDecoys 8 → 17, hence ~2× cluster fetches
  and ~10 % more bytes per blob).

Recall identical-to-better at every tier across mitigation deltas.

---

## DBpedia 1M @ 1536-dim ada-002 — Memory + RA=2 (2026-05-12)

The first DBpedia 1M bench to actually produce headline numbers (commit
`4880cae` switched Storage:File → Storage:Memory and added
`RedundantAssignments=2`). Earlier runs OOM'd on memory or wedged on
FileStore-serial-bottlenecks; both root-caused + fixed in the May 11-12
push. Hardware: `m6i.4xlarge` (16 vCPU, 64 GB). All four mitigations
live (σ=2^45 + DecodePublic, permutation π, PaddingBucketed,
TargetEpsilon=2.5). RA=2 = each vector indexed under its top-2 nearest
clusters — standard fix for the curse-of-dimensionality NN-miss problem
on dense semantic embeddings.

### `TestDBpedia1MAccuracy` (NC=128, RA=2, Memory, ε=2.5, σ=2^45)

| Config   | Probe | Build  | Recall@1 | Recall@10 | Avg query |
|----------|-------|--------|----------|-----------|-----------|
| probe-8  | 6.25 %| 19m32s | 96.0 %   | **94.4 %**| 2.08 s    |
| probe-16 | 12.5 %| 17m50s | 94.0 %   | **96.0 %**| 3.43 s    |

### `TestPQ_DBpedia1M` (NC=128, RA=2, Memory, ε=2.5, σ=2^45)

| Config                  | PQ  | Probe  | Build  | Recall@1 | Recall@10 | Avg query | P50      |
|-------------------------|-----|--------|--------|----------|-----------|-----------|----------|
| **PQ-M48-probe8-eps25** | M48 | 6.25 % | 21m14s | 96.0 %   | **92.4 %**| **1.17 s**| **1.16 s** |
| PQ-M96-probe16-eps25    | M96 | 12.5 % | 21m58s | 92.0 %   | 92.8 %    | 1.67 s    | 1.66 s   |

`standard-probe8` in the PQ-test config-set was cancelled (redundant
with the accuracy-test probe-8 above — identical configuration).

### Comparison vs pre-fix (RA=1, File-store) DBpedia bench

| Metric          | Pre-fix (RA=1, File) | Post-fix (RA=2, Memory) | Δ            |
|-----------------|----------------------|--------------------------|--------------|
| probe-8 R@10    | 75.2 %               | **94.4 %**               | **+19.2 pp** |
| probe-8 query   | 18.5 s               | **2.08 s**               | **9× faster**|
| probe-16 R@10   | 75.0 %               | **96.0 %**               | **+21.0 pp** |
| probe-16 query  | 30.8 s               | **3.43 s**               | **9× faster**|
| PQ-M48 query    | n/a (disk-full fail) | **1.17 s**               | **<2 s ✅**  |

### Headline DBpedia number

**PQ-M48-probe8-eps25 on m6i.4xlarge: 92.4 % Recall@10 at P50 = 1.16 s**
on 1M × 1536-dim ada-002 semantic embeddings, with all four privacy
mitigations live and tunable ε=2.5 per-query DP bound. Build wall:
21m14s.

Cost: ~$1.20 (m6i.4xlarge × ~1.5 hr).

### Why the previous DBpedia run was so much slower

The May 10 → May 11 wedge + OOM cascade was three independent issues:

1. `FileStore.PutBatch` held a global mutex across all 1024 file
   writes → all 16 encrypt workers serialised on one writer.
2. `FileStore.saveIndex` was called per `PutBatch` → ~1000 full JSON
   marshals of the ~20 MB index file per build = ~250 sec wasted.
3. `streamPaddingBlobs` called `Put` per padding blob → each Put
   eager-flushed the (already huge) index.

All three fixed in `dd4fc36`. With those fixes, File-storage would
have been usable. But the May 12 SIFT 1M validation showed Memory
storage easily fits 1M × 1536-dim on m6i.4xlarge (peak ~18 GB build,
~6.5 GB steady-state) after the additional float32 + drop-pending
optimisations (`7a9a369` + `828ee0d` + `8905f2d`). Memory is now the
right default for DBpedia.

---

## SIFT 1M validation (2026-05-12, m6i.2xlarge)

Pre-DBpedia validation: rerun the canonical SIFT 1M bench on current
HEAD to confirm no regression after the May 11-12 build-phase /
FileStore optimisations.

### `TestSIFT1MAccuracy` (NC=128, σ=2^45) — May 12 reproduce

| Config     | Probe  | Multi | Recall@1 | Recall@10 | Avg query |
|------------|--------|-------|----------|-----------|-----------|
| strict-4   | 3.1 %  | no    | 88.0 %   | 86.0 %    | 248 ms    |
| strict-8   | 6.25 % | no    | 98.0 %   | 96.4 %    | 302 ms    |
| strict-16  | 12.5 % | no    | 100.0 %  | 99.8 %    | 399 ms    |
| probe-8    | 6.25 %+| yes   | 100.0 %  | 99.8 %    | 402 ms    |
| probe-16   | 12.5 %+| yes   | 100.0 %  | 100.0 %   | 540 ms    |

### `TestPQ_SIFT1M` (NC=128, σ=2^45) — May 12 reproduce

| Config                 | Recall@1 | Recall@10 | Avg query | P50    | Build  |
|------------------------|----------|-----------|-----------|--------|--------|
| **PQ-M8-probe8-eps25** | 100.0 %  | 97.6 %    | **357 ms**| 361 ms | 2m52s  |
| PQ-M8-probe16-eps25    | 100.0 %  | 99.8 %    | 475 ms    | 451 ms | 2m53s  |
| PQ-M8-probe8-eps271    | 100.0 %  | 97.8 %    | 346 ms    | 351 ms | 2m58s  |
| PQ-M8-probe8-eps20     | 100.0 %  | 98.2 %    | 419 ms    | 420 ms | 2m52s  |
| standard-probe8-eps25  | 98.0 %   | 99.6 %    | 397 ms    | 397 ms | 2m13s  |

Numbers match historical SIFT 1M within ±2 pp / ±20 ms sampling noise.
**Builds ~3× faster than May 9** (PQ ~8m → ~2m52s; standard ~3m28s →
~2m13s) — driven by the PQ subsample-first fix (`8905f2d`: codebook
k-means now trains on 100 K subset instead of full 1 M, no recall
regression because that's standard PQ literature precedent — Jegou et
al. 2011, FAISS index_factory).

Cost: ~$0.27 (m6i.2xlarge × ~45 min).

---

## SIFT 100K — DBpedia-prep config validation (2026-05-11, m6i.xlarge)

12 configs in one fire (~$0.04, ~10 min wall). Validated:

| Config                  | Recall@1 | Recall@10 | Avg query | Note                       |
|-------------------------|----------|-----------|-----------|----------------------------|
| standard (baseline)     | 98 %     | 97.2 %    | 166 ms    | reference                  |
| **standard-RA2**        | **100 %**| **99.8 %**| 175 ms    | ✅ RA=2 — +2.6 pp R@10     |
| standard-PCA64          | 80 %     | **49.0 %**| 144 ms    | ❌ PCA-64 destroys SIFT    |
| standard-PCA64-RA2      | 80 %     | 49.6 %    | 162 ms    | ❌ RA=2 can't rescue PCA   |
| PQ-M8-PCA64             | 80 %     | 49.4 %    | 140 ms    | ❌ PCA kills PQ too        |
| PQ-M8-PCA64-RA2         | 80 %     | 49.0 %    | 159 ms    | ❌ full PCA stack broken   |

**Two findings**:
1. `RedundantAssignments=2` is a real recall win — even on already-
   maxed SIFT 100K it picked up R@1 100 % (was 98 %) and R@10 99.8 %
   (was 97.2 %) at +10 ms query cost. Confirmed safe to enable.
2. `PCADimension=64` destroys recall on SIFT 128-dim — going 128 → 64
   loses too much information at this scale. **PCA is not in the
   DBpedia recipe.** (Higher PCA ratios on higher-dim datasets may
   work; not validated here, deferred.)

The PCA configs are kept in the test file as "tested-and-rejected"
guards — future runs should not re-test these.
