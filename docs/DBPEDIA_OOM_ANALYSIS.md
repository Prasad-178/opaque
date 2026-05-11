# DBpedia 1M @ 1536-dim OOM — Root-Cause Analysis

**Status:** Diagnosed 2026-05-11 (after 11 failed bench attempts). Three compounding bugs in `pkg/hierarchical/enterprise_builder.go` `generatePaddingBlobs` + `computePaddingTarget` account for ~12-20 GB of unexplained build-phase memory. Plus an untried Go runtime knob (`GOMEMLIMIT`) for RSS-lag.

---

## Symptoms (recap)

Eleven AWS bench attempts at DBpedia 1M @ 1536-dim, all OOM-killed exactly at instance RAM cap:

| Attempt | Instance | OOM at | Notes |
|---|---|---|---|
| 6 | r6i.4xlarge (128 GB) | 128 GB | pre-optimizations |
| 7 | m6i.4xlarge (64 GB) | 64 GB | post `1e735ec` opts |
| 9 | m6i.2xlarge (32 GB) | 32 GB | post float32 storage tier |
| 10 | m6i.4xlarge (64 GB) | 64 GB | float32 + 4 bench opts |
| 11 | m6i.4xlarge (64 GB) | 64 GB | + test-side 12 GB free |

Mental model predicted **~41 GB peak** with all optimizations stacked. Empirical was **~64 GB**. The ~20 GB gap is explained by the bugs below.

---

## Bug #1: Padding ciphertext size mismatch

`pkg/hierarchical/enterprise_builder.go:83`

```go
ciphertextLen := dimension*8 + 28
```

Real ciphertexts are float32-encoded since commit `7a9a369` (4 bytes/element instead of 8). The padding generator was not updated. Padding blobs are now **2× the size of real blobs**.

**Memory impact at 1M × 1536-dim:** if total padding count is ~1.1M (see Bug #2),
padding occupies `1.1M × (1536*8 + 28) = ~14 GB` instead of `~7 GB`. Difference: ~7 GB.

**Privacy impact:** padding is distinguishable from real on the wire by size. A
server sees `padding_size = 2 × real_size`. This trivially defeats the
PaddingBucketed/PaddingMaxFixed privacy goal of "indistinguishable padding."

**Fix:** match real ciphertext format.

```go
ciphertextLen := dimension*4 + 28
```

---

## Bug #2: PaddingBucketed pads to global-max nextPow2, not per-cluster

`pkg/hierarchical/enterprise_builder.go:51-66`

```go
func computePaddingTarget(counts map[string]int, mode enterprise.PaddingMode) int {
    ...
    maxCount := 0
    for _, c := range counts {
        if c > maxCount { maxCount = c }
    }
    if mode == enterprise.PaddingMaxFixed { return maxCount }
    return nextPow2(maxCount)  // ← Bucketed uses GLOBAL max's nextPow2
}
```

`opaque.go`'s public-API doc explicitly says Bucketed pads each cluster to its
**own** next power-of-2 ("each cluster up to next power-of-2 vector count,
~6-12 % storage waste, closes most volume leak"). The implementation pads
**all** clusters to `nextPow2(global_max)`.

**Memory impact at 1M × 1536-dim, NC=128:**

- Per-cluster avg = 7,812 vectors.
- Real cluster count distribution has σ ≈ 10-20 % of mean for random k-means.
- Max cluster count is typically 9,000-10,000 — **just over the 8,192 nextPow2 threshold**.
- That cliff: `nextPow2(9,000) = 16,384` → every cluster padded to 16,384.
- Total padded blob count = `128 × 16,384 = 2,097,152` ≈ **2.1M blobs**.
- Real-blob count = 1M.
- **Padding count = 2.1M − 1M = ~1.1M padding blobs.**

Padding memory:
- Current (float64 size): 1.1M × 12,316 B ≈ **14 GB padding alone**.
- After Bug #1 fix: 1.1M × 6,172 B ≈ 7 GB.
- After per-cluster fix: each cluster padded to its OWN nextPow2 (~5-12 %
  inflation), total padding ≈ 50K-130K blobs ≈ **0.3-0.8 GB**.

If max cluster count happens to be < 8,192, the global-max-nextPow2 = 8,192
which is close to avg → padding is only ~5 %. Whether OOM hits is then a
**lottery** on a single cluster's seed-dependent count.

This matches the empirical observation: at m6i.4xlarge (64 GB) we OOM
consistently across multiple runs, suggesting the max cluster is reliably ≥
8,192 in the DBpedia dataset.

**Fix:** per-cluster nextPow2.

```go
// computePaddingTarget signature changes to per-cluster — return a map.
// Or: in generatePaddingBlobs, for each bucketKey compute toAdd = nextPow2(count) - count.
```

This matches the documented semantics and gives the promised ~6-12 % storage
waste — vs the current ~100 % waste.

---

## Bug #3: Padding blobs all materialise in RAM before flush

`pkg/hierarchical/enterprise_builder.go:84-101` and
`pkg/hierarchical/kmeans_builder.go:318-321`

```go
var pad []*blob.Blob
for bucketKey, count := range counts {
    for i := 0; i < toAdd; i++ {
        ...
        pad = append(pad, blob.NewBlob(id, bucketKey, ct, dimension))
    }
}
return pad
// ...
blobs = append(blobs, paddingBlobs...)
```

All padding blobs accumulate in `pad` (and then merged into `blobs`) before
`store.PutBatch(ctx, blobs)` writes anything to disk. With `Storage: File`
we'd expect disk-backed offload, but the path materialises 100 % of the
padding in RAM first.

**Memory impact:** at 1.1M padding × 12 KB current = 14 GB in RAM at the
moment of the PutBatch call. Even with Storage:File, this 14 GB peak fires
before any disk write.

**Fix:** stream padding generation directly into the store via
`store.Put()` per-blob inside the loop. Trivial change.

---

## Combined estimate: build-phase peak memory at 1M × 1536-dim

With current code:
- pendingVectors (float32): 6 GB
- normalizedVecs (float32, peak before free): 6 GB
- Real AES ciphertext accumulator: 6 GB
- Padding blob accumulator: **14 GB** ← Bug #1 + #2 + #3
- Go GC slack at GOGC=50: ~+12 GB
- Kernel + sshd + setup: ~5 GB
- **Total: ~49 GB live × 1.5 GC slack ≈ 64-70 GB** ← matches OOM exactly

After fixing Bugs #1 + #2 + #3:
- pendingVectors: 6 GB
- normalizedVecs (peak): 6 GB
- Real AES accumulator: 6 GB
- Padding (per-cluster nextPow2, float32-sized, streaming PutBatch): **< 0.5 GB**
- GC slack: ~+9 GB
- Kernel: ~5 GB
- **Total: ~33 GB peak** — fits comfortably on m6i.4xlarge (64 GB), plausibly on m6i.2xlarge (32 GB).

---

## Bonus: untried Go runtime knob

`GOMEMLIMIT` (Go 1.19+) is a SOFT memory cap. When live + cached memory
approaches it, Go's runtime GC's more aggressively and immediately returns
pages to the OS. Pairs with the existing `GOGC=50`.

**Currently NOT set in `deploy/bench-cpu/run_dbpedia_bench.sh`.** Adding:

```bash
export GOMEMLIMIT=48GiB
export GODEBUG=madvdontneed=1    # eager RSS reclaim instead of lazy MADV_FREE
```

…closes the gap between Go's live heap and OS-visible RSS. Even without the
padding-bug fixes, this knob alone might bring the bench under 64 GB.

---

## Recommended next-session order

1. **Fix Bug #1** (padding size float64 → float32): 1-line change in
   `enterprise_builder.go:83`. Privacy + memory win.
2. **Fix Bug #2** (per-cluster nextPow2 in PaddingBucketed): ~10-line change.
   Big memory win, also matches documented semantics. Add unit test.
3. **Fix Bug #3** (stream padding through `store.Put` rather than
   accumulate): ~5-line change. Memory peak headroom.
4. **Add `GOMEMLIMIT=48GiB`** to the bench runner: env-var change in
   `run_dbpedia_bench.sh`. Belt-and-suspenders.
5. **Re-fire DBpedia bench on m6i.4xlarge** with all of the above. Expected
   to succeed with comfortable headroom; m6i.2xlarge becomes plausible.

All four changes are bounded engineering work (~1-2 hours including tests),
not multi-day refactor.
