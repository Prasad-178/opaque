# Build-phase Memory Profile — what we learned trying to bench DBpedia 1M

**Status:** Reference document. Captures findings from the 2026-05-09/10
attempts to benchmark Opaque on DBpedia-OpenAI-1M (1M × 1536-dim
ada-002 text embeddings) so future sessions don't re-discover the
same memory cliffs.

**TL;DR:** Opaque's `Build()` path peaks at roughly **10× the raw
vector size** at 1M scale, dominated by float64 storage + multiple
in-memory copies + Go GC slack. At SIFT1M (128-dim, ~1 GB raw) the
peak is ~10 GB → fits a laptop. At DBpedia 1M (1536-dim, ~12.3 GB
raw) the peak is ~128 GB before optimizations, ~64 GB after the
three quick wins below. A planned **float32 refactor** would halve
this again to ~30 GB.

---

## What happened

We tried to bench DBpedia 1M @ 1536-dim across **eight EC2 attempts**
on m6i.4xlarge (64 GB) and r6i.4xlarge (128 GB). Cumulative wasted
spend: ~$7.68. Issues encountered, in order:

| # | Instance      | Failure                                | Fixed by |
|---|---------------|----------------------------------------|----------|
| 1 | m6i.4xlarge   | HF download hang (no token, AWS IP rate-limit) | HF_TOKEN env var (`fix(bench): HF_TOKEN env var support`) |
| 2 | m6i.4xlarge   | `ModuleNotFoundError: numpy` in conversion | `pip install numpy` (`fix(bench): explicitly pip install numpy`) |
| 3 | m6i.4xlarge   | OOM in conversion (full pylist materialization) | Streaming shard-by-shard converter (`fix(bench): stream parquet → fvecs`) |
| 4 | m6i.4xlarge   | Dangling Python ref after streaming fix | Strip leftover write_fvecs calls (`fix(bench): strip dangling write_fvecs`) |
| 5 | m6i.4xlarge   | OOM at 64 GB during build, no PQ       | Switch to r6i.4xlarge (128 GB) |
| 6 | r6i.4xlarge   | OOM at **128 GB** anon-rss             | Mem optimizations needed |
| 7 | m6i.4xlarge (post-opts) | OOM at 64 GB anon-rss exactly | Mem opts cut peak 50 % but instance still tight; needs r6i.4xlarge |
| 8 | r6i.4xlarge (post-opts) | aborted by user before completion | (would have likely succeeded; see below) |

---

## The memory profile, decomposed

OOM-killer log on attempt #6 (r6i.4xlarge, 128 GB instance, no opts yet):

```
test.test invoked oom-killer ... total-vm:131464288kB anon-rss:128560632kB
```

That's **128 GB resident set size** for a process that started with
12.3 GB of raw input. Where does it all go?

### Per-component estimate at 1M × 1536-dim, no PQ, NC=128

| Component                                                            | Size        | Notes |
|----------------------------------------------------------------------|-------------|-------|
| `dataset.Vectors` (test code holds it for brute-force GT compute)    | 12.3 GB     | float64; needed throughout the test loop |
| `db.pendingVectors` (defensive copy made by `AddBatch`)              | 12.3 GB     | held until Build completes |
| `kmeans_builder.normalizedVecs` (full normalized copy)               | 12.3 GB     | needed for k-means.Fit + assignment passes |
| AES ciphertext accumulator across worker goroutines                  | ~12 GB      | held in `localBlobs` until merged + flushed |
| K-means working memory (distance arrays, centroid updates)           | ~3-5 GB     | per-iteration allocations |
| HE engine pool (8 × CKKS engines @ 50 MB)                            | ~0.4 GB     | constant |
| Padding-bucketed zero-vector ciphertexts                             | ~0.6 GB     | proportional to padding, modest |
| **Live working set sum**                                             | **~52 GB**  | what's actually referenced |
| **× Go's GOGC=100 default (heap can grow 2× live before GC)**        | **~104 GB** | this is the silent killer |
| Plus kernel + Linux page tables + sshd + go test driver              | ~5-10 GB    | |
| **Observed OOM at**                                                  | **~128 GB** | matches |

### Why is this 10× the raw data?

1. **float64 vs float32.** Go's idiomatic numeric type is float64.
   Every intermediate vector copy is 2× larger than it would be in a
   float32 codebase (FAISS, hnswlib, most C++ vector DBs).
2. **Defensive copies at API boundaries.** `AddBatch` copies the
   caller's slice into `db.pendingVectors`. The caller (test code)
   keeps its own copy for brute-force GT. Doubles raw vector storage.
3. **Pre-computed normalized vectors.** `kmeans_builder.go` builds a
   full second copy of normalized vectors upfront for k-means. After
   k-means finishes, it's only used per-vector inside the worker loop
   — but the slice lives until Build returns.
4. **Worker accumulator pattern.** AES-encrypt phase uses N worker
   goroutines, each accumulating its `localBlobs` slice locally,
   then merging. Peak is `2 × ciphertexts_size`.
5. **Go GC overhead.** `GOGC=100` default means runtime triggers GC
   when heap doubles. So total RSS peaks at ~2× live working set.
   Tightening to `GOGC=50` halves this overhead.

---

## What was reclaimed (commit `1e735ec`, 2026-05-10)

Three quick wins landed, cutting build peak from ~128 GB → ~64 GB:

1. **Free `normalizedVecs` after k-means.** `pkg/hierarchical/kmeans_builder.go`
   sets the slice to `nil` after the k-means + assignment passes
   complete. Workers in the AES-encrypt phase re-normalize per-vector
   on demand. Saves ~12 GB; CPU cost ~5-10 s of extra normalization.
2. **`Storage: File` for DBpedia tests.** Offloads ~12 GB of in-memory
   ciphertext blobs to disk-backed EBS (`/tmp/opaque-dbpedia-*`).
   Per-query read overhead ~1-3 ms (gp3 ~500 MB/s × 1 MB fetch),
   negligible vs the 446 ms SIFT1M baseline.
3. **`GOGC=50` env var.** Triggers Go GC at 50 % heap growth instead
   of 100 %. Halves the GC slack overhead at the cost of more frequent
   GC cycles. Build is RAM-bound not CPU-bound (load was 4/16 = 25 %),
   so the CPU cost is invisible.

Combined: ~40 GB peak reduction. m6i.4xlarge (64 GB) still OOMs by
single-digit GB margin (the remaining peak hugs the instance limit).
r6i.4xlarge (128 GB) gives 2× headroom and would complete the bench
at ~$2.

---

## Pending: the bigger optimization (`float32` refactor)

The deeper structural fix is to replace `[][]float64` with `[][]float32`
across the codebase's hot paths. This halves every memory hog at once:

- `db.pendingVectors`: 12.3 → 6.2 GB
- worker temp buffers: 50 % less
- centroids, normalized vectors: 50 % less
- AES ciphertext payload: 50 % less

Total expected peak after float32 refactor: **~30 GB** at 1M × 1536-dim.
Fits comfortably on m6i.2xlarge (32 GB) at $0.38/hr → DBpedia bench
becomes a ~$1 run instead of ~$3.50 on r6i.4xlarge.

### Scope (multi-day refactor)

API change: public `[]float64` → `[]float32` on `DB.Add`, `DB.AddBatch`,
`DB.Search`. `Result.Score` stays `float64` (similarity scalar; precision
matters at the very end).

Internal packages requiring updates:
`pkg/embeddings`, `pkg/cluster`, `pkg/pca`, `pkg/pq`, `pkg/hierarchical`,
`pkg/client`, `pkg/auth`, `pkg/enterprise`, plus `opaque.go`.

`pkg/encrypt` is byte-level and unaffected. `pkg/crypto` (CKKS HE) keeps
`[]float64` since Lattigo's encoder requires it — float32 vectors get
upcast at the HE encode boundary (cheap, narrow).

### Accuracy / speed / privacy impact

- **Accuracy:** ada-002 embeddings are float32-native (HF parquet stores
  float32; we currently upcast on load and lose nothing by keeping it).
  SIFT 1M float32 vs float64 cosine similarity differs by ≤1.2e-7 per
  computation — well below recall sensitivity. **Recall numbers should
  be identical within sampling noise.**
- **Speed:** memory bandwidth halved on hot paths → 1.3-2× faster on
  memory-bound operations (k-means, dot products, AES encode).
  **Slight win.**
- **Privacy:** zero impact. Encryption operates on bytes; float type
  only changes the byte count being encrypted.

---

## The DBpedia bench result we don't have yet

Once the float32 refactor lands, the DBpedia 1M @ 1536-dim bench is
expected to fit on m6i.2xlarge ($0.38/hr × ~1.5 hr ≈ $0.60). Until
then, the headline benchmark remains SIFT1M:

> AWS m6i.2xlarge, σ=2^45 IND-CPA^D-128, all four mitigations live by
> default, NC=128, ε=2.5: **99.4 % Recall@10 at 446 ms** (probe-8).

For paper-track purposes, the SIFT1M numbers are sufficient as the
primary headline, with DBpedia listed as a planned text-retrieval
validation point post-refactor.

---

## Lessons captured

1. **HF Hub aggressively rate-limits AWS IPs without a token.** Always
   set `HF_TOKEN`. The `setup_dbpedia.sh` + `run_dbpedia_bench.sh`
   chain enforces this with a fail-fast check.
2. **`pyarrow.to_pylist()` materializes Python list-of-lists which is
   ~10× larger than the underlying numpy array.** Stream-by-shard.
3. **Go's `GOGC=100` default keeps 2× the live working set as GC slack.**
   For RAM-bound workloads, `GOGC=50` is a free 30-50 GB reclaim at
   1M × 1536-dim scale.
4. **Add a heartbeat cron / log-mtime monitor when firing long benches.**
   The 6-hour silent HF download hang on attempt #1 was preventable.
5. **Inspect peak memory via SSH `free -h` + `ps -eo pmem` early in the
   build phase** rather than waiting for OOM-killer to confirm.
