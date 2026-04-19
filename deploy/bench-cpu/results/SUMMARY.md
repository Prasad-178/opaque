# SIFT 1M Benchmark Results

Raw logs gitignored. This summary captures per-instance results so they
survive the ephemeral EC2 teardown.

## c6i.2xlarge (8 vCPU, 16 GB, Intel Ice Lake)

Run 2026-04-19. Full log under `20260419_201704-c6i.2xlarge/` (local only).

### TestSIFT1MAccuracy (128 clusters, 50 queries, 8 decoys)

| Config     | Probe | Multi | Recall@1 | Recall@10 | Avg query |
|------------|-------|-------|----------|-----------|-----------|
| strict-4   | 3.1%  | no    | 90.0%    | 87.6%     | 238 ms    |
| strict-8   | 6.2%  | no    | 94.0%    | 93.6%     | 282 ms    |
| strict-16  | 12.5% | no    | 100.0%   | 99.2%     | 349 ms    |
| probe-8    | 6.2%+ | yes   | 100.0%   | 99.8%     | 345 ms    |
| probe-16   | 12.5%+| yes   | 100.0%   | 100.0%    | 452 ms    |

### TestPQ_SIFT1M (128 clusters, 50 queries, 8 decoys)

| Config             | Recall@1 | Recall@10 | Avg query | P50    | Build    |
|--------------------|----------|-----------|-----------|--------|----------|
| standard-strict8   | 96.0%    | 96.0%     | 282 ms    | 277 ms | 3m21s    |
| standard-strict16  | 100.0%   | 99.8%     | 347 ms    | 339 ms | 3m11s    |
| PQ-M8-strict8      | 90.0%    | 92.4%     | 270 ms    | 268 ms | 8m32s    |
| PQ-M8-strict16     | 100.0%   | 98.4%     | 327 ms    | 326 ms | 8m25s    |
| PQ-M8-strict32     | 100.0%   | 99.6%     | 445 ms    | 444 ms | 8m16s    |
| PQ-M8-probe16      | 100.0%   | 99.2%     | 406 ms    | 403 ms | 8m23s    |
| PQ-M8-probe32      | 100.0%   | 100.0%    | 497 ms    | 486 ms | 8m13s    |

### Context

AWS c6i.2xlarge matches Pinecone p1.x1 pod + Qdrant 8-core tier. Opaque on
this hardware achieves 99 %+ Recall@10 at ~350 ms latency on 1 M vectors
with full privacy (CKKS HE + AES-256-GCM + decoy-based access-pattern
hiding). PQ build cost is ~8 min one-time; search latency comparable to
standard at matching recall.

## c6i.4xlarge (16 vCPU)

Pending — paused before completion on 2026-04-19.
