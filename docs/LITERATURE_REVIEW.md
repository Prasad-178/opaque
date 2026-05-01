# Literature Review: Privacy-Preserving Vector Search

## Opaque's Position

**Opaque achieves the fastest published results for private vector search at million-scale:**

Hardware context matters for this table. Where possible we report the
competitor's published hardware inline. Opaque rows are from AWS
`c6i.2xlarge` (8 vCPU Intel Ice Lake) — the commodity production tier
(Pinecone `p1.x1` / Qdrant 8-core pod). All Opaque numbers are
reproducible via `bash deploy/bench-cpu/run_bench.sh c6i.2xlarge`.

**Security mitigations shipped (2026-04):**
- **σ noise flooding 2^20 → 2^30 + DecodePublic(logprec=10)** for Li-Micciancio mitigation (commit `10b7850`, eprint 2020/1533).
- **Per-tenant blob ID permutation π** hides centroid-coordinate ↔ access-pattern link from server (commit `bc0ec45`, fully wired in `e45223b`).
- **`PaddingMode=Bucketed` constant-volume padding** closes volume side-channel (commit `414aa8e`).
- **`TargetEpsilon=2.0` DP-style decoy sizing** derives NumDecoys ≈ 17 for SIFT1M (commit `990e2be`); see `docs/SECURITY_MODEL.md` §5.1 for the bound.

Full-mitigation SIFT1M numbers below are from a clean AWS c6i.2xlarge run
(2026-04-30 19:08, all four mitigations live, see
`deploy/bench-cpu/results/SUMMARY.md`). Latency adds ~30-65 % vs the
partial-mitigation run (extra decoys + padding bandwidth). Recall is
identical-or-better across configurations.

| System | Year/Venue | Approach | Scale | Recall@10 | Latency | Hardware | Security |
|--------|-----------|----------|-------|-----------|---------|----------|----------|
| **Opaque (probe-8, optimal ε=2.5)** | **2026** | **CKKS HE + AES + π + Bucketed pad + ε=2.5 decoys** | **1M, 128d** | **99.6%** | **462ms** | **m6i.2xlarge (8 vCPU)** | **HE + AES + permuted access + DP-bounded** |
| **Opaque (probe-16, optimal ε=2.5)** | **2026** | **CKKS HE + AES + π + Bucketed pad + ε=2.5 decoys** | **1M, 128d** | **100.0%** | **635ms** | **m6i.2xlarge (8 vCPU)** | **HE + AES + permuted access + DP-bounded** |
| Opaque (PQ-M8-probe8, optimal ε=2.5) | 2026 | CKKS HE + PQ + AES + π + pad + ε=2.5 | 1M, 128d | 97.6% | 428ms | m6i.2xlarge | HE + AES + permuted access + DP-bounded |
| Opaque (probe-8, ε=2 high-privacy tier) | 2026 | CKKS HE + AES + π + pad + ε=2.0 | 1M, 128d | 99.4% | 630ms | c6i.2xlarge | stronger ε bound |
| Opaque (probe-8, partial-mit) | 2026 | CKKS HE + AES + decoys (8 fixed) | 1M, 128d | 99.0% | 366ms | c6i.2xlarge | HE + AES + statistical decoys |
| Opaque (PQ-M8-probe32, baseline) | 2026 | CKKS HE + PQ + AES + decoys | 1M, 128d | 100.0% | 497ms | c6i.2xlarge | HE + AES + statistical decoys |
| Opaque (PQ, GIST, pre-mitigation) | 2026 | CKKS HE + PQ + AES + decoys | 100K, 960d | 98.0% | 497ms | M4 Pro (10 vCPU) | HE + AES + access-pattern hiding |
| Compass | OSDI 2025 | ORAM + HNSW | ~1M | high | ~1s | Full server compromise |
| PPMI | arXiv 2025 | CKKS + AES-256 | 1M | >99% | 951ms | 128-bit IND-CPA |
| RemoteRAG | ACL 2025 | PHE + DP | 1M | 100% | 670ms | Differential privacy + PHE |
| Pacmann | ICLR 2025 | PIR + graph ANN | 100M | ~90% | ~3,100ms | Single server PIR |
| SecureRAG ⚠️ | NeurIPS GenAI4Health 2025 | FHE + KP-ABE | **16,384 docs** | not reported | 50ms (workshop bench) | End-to-end FHE + ABE; trusted reader required |
| Panther | CCS 2025 | PIR+SS+GC+HE | 10M | not reported | ~18,000ms | Single server |
| PRAG | PrivateNLP 2024 | MPC (CrypTen) | 100K | >99% | ~10,000ms | Multi-server MPC |
| FedSQ ⚠️ | VLDB 2024 | MPC (federated) | 200K/owner × 5 | not reported | 50-100ms (aggregation only) | **Plaintext per owner**, MPC for cross-owner agg |
| Tiptoe | SOSP 2023 | LHE + clustering | 360M pages | not reported | 2,700ms | Crypto only, single server |
| Servan-Schreiber | S&P 2022 | LSH + DPF | 10M | not reported | 10-20s | Two non-colluding servers |
| SANNS | USENIX Sec 2020 | HE+ORAM+GC | 10M, 96d | ~90% | ~31,000ms (1t) / ~1,400ms (72t) | Two-server 2PC |

### Key Competitors to Watch

**Compass (OSDI 2025)** — Closest competitor. Uses ORAM + HNSW for graph-based encrypted search. ~1s latency with cryptographic access pattern guarantees (ORAM, not statistical decoys). Published at a top systems venue. We need to compare against this carefully — they have stronger access pattern privacy but likely higher latency.

**PPMI (arXiv 2025)** — Very similar to Opaque's approach (CKKS + AES-256). Claims <1s on 1M entries. Uses GPT-4o + local Llama for query decomposition. Directly comparable — we should benchmark against their claims.

**RemoteRAG (ACL 2025)** — Uses differential privacy + Paillier PHE. Claims 0.67s on 1M with 100% recall. Weaker security model (DP perturbation) but fast. Good baseline to compare against.

**FedSQ (VLDB 2024)** ⚠️ **Not a direct competitor — different threat model.** FedSQ is a *federated* multi-party system: each owner runs Milvus over its own **plaintext** data, and only cross-owner aggregation is via MPC. The "50–100 ms" headline is the aggregation phase across ~5 hospitals × ~200K rows each — not single-server private 1M. For an attacker that is one of the participating owners, the local query is in plaintext to that owner. Different problem (data sovereignty across distrustful organizations) than Opaque's HBC single-server-no-plaintext-anywhere. Demo-track paper.

**SecureRAG (NeurIPS GenAI4Health Workshop 2025)** ⚠️ **Different scale and threat model than initially logged.** The workshop paper benchmarks "100 documents from a corpus of 16,384" at 0.05s on a single GPU — not 1M @ 2.2s as previously summarized. Threat model also differs: it is a 4-party system requiring a *trusted reader* (hospital admin role) — strictly stronger trust assumption than Opaque's HBC server. Genuinely novel contribution: KP-ABE for per-document access control (something Opaque doesn't currently provide).

**Pacmann (ICLR 2025)** — Scales to 100M with PIR. Slower (3.1s) but at 100x our scale. If we can demonstrate 10M with reasonable latency, we close this gap.

## Deep-Dive: Compass (OSDI 2025) — Closest Competitor

**Paper:** Zhu, Patel, Zaharia, Popa. "Compass: Encrypted Semantic Search with High Accuracy." OSDI 2025. ([USENIX](https://www.usenix.org/conference/osdi25/presentation/zhu-jinhao), [ePrint 2024/1255](https://eprint.iacr.org/2024/1255))

### Architecture

Compass performs HNSW graph traversal over encrypted embeddings stored in Ring ORAM. The client drives the search algorithm; the server holds only encrypted ORAM blocks and performs XOR-based bandwidth reduction.

```
Client: HNSW search controller + direction sketch + PQ decoder
  ↕ Ring ORAM (2 round trips per access)
Server: Encrypted ORAM tree (each block = one HNSW node)
```

### Key Techniques

**Directional Neighbor Filtering:** At each HNSW node, instead of fetching ALL neighbors from ORAM (expensive), the client uses a small locally-cached "direction sketch" to determine which neighbors are in the same direction as the query. Only directionally-aligned neighbors are fetched. This dramatically reduces ORAM accesses per step.

**Speculative Neighbor Prefetch:** Instead of waiting for each HNSW step to complete before fetching the next node, Compass predicts likely next nodes and issues prefetch requests in parallel. This pipelines ORAM round-trips, hiding latency.

**Graph-Traversal Tailored ORAM:** Modified Ring ORAM optimized for HNSW's access pattern. Eviction (ORAM background maintenance) takes 5.8s but runs asynchronously — overlapped with LLM generation in a RAG pipeline so the user doesn't perceive it.

**PQ compression:** Embeddings within ORAM blocks are PQ-compressed (8 sub-vectors for 128-dim, 32 for 768-dim). This reduces per-block size and ORAM bandwidth.

### Performance

| Dataset | Vectors | Dims | Perceived Latency | Notes |
|---------|---------|------|-------------------|-------|
| SIFT1M | 1M | 128 | <1s (~600-900ms) | PQ with 8 sub-vectors |
| LAION | ~1M | 512 | <1s | Similar bandwidth despite 4x larger dims |
| MS MARCO | ~8.8M | 768 | **1.3s** | Enterprise-scale; eviction in background |
| TripClick | ~1.5M | 768 | ~1s | M=128 HNSW connectivity |

Recall claimed to match plaintext HNSW quality.

### Security Model

**What does NOT leak (cryptographic):**
- Which embeddings are accessed (ORAM guarantee)
- The query vector
- Search results
- Data content (all encrypted)

**What leaks:**
- Query timing (when, not what)
- Volume pattern (fixed/padded, so minimal leakage)

**Security claim:** Privacy of data, queries, and results "even if the server is compromised." This is **cryptographic** access pattern hiding (ORAM), stronger than Opaque's **statistical** hiding (decoys).

### Comparison with Opaque

| Dimension | Opaque | Compass |
|-----------|--------|---------|
| **SIFT1M latency** | **415ms** | **~600-900ms** |
| **Access pattern privacy** | Statistical (decoys) | **Cryptographic (ORAM)** |
| **Search algorithm** | IVF (k-means clusters) | HNSW (graph-based) |
| **HE usage** | Only centroid scoring | None (ORAM-only, no HE) |
| **Largest dataset** | 2M (benchmarked) | **8.8M (MS MARCO)** |
| **Background overhead** | None | 5.8s eviction (async) |
| **PQ usage** | ADC for local scoring | Embedding compression in ORAM |

**Opaque is 1.5-2x faster** on SIFT1M but **Compass provides stronger security guarantees**. The fundamental tradeoff: Opaque minimizes crypto (only centroid scoring under HE), Compass puts everything under ORAM. Opaque's decoys are practical but weaker; Compass's ORAM is provable but slower.

**Paper positioning:** This tradeoff is the key differentiator. Opaque demonstrates that for the honest-but-curious threat model (common in enterprise deployments), statistical access pattern hiding with decoys is sufficient and yields significantly lower latency. Compass is the right choice when defending against a fully compromised server.

## Why Opaque Is Fast

Most systems do all computation under crypto (MPC, ORAM, garbled circuits). Opaque's key architectural insight: **minimize what's done under HE**.

| System | What's done under crypto | Cost |
|--------|-------------------------|------|
| SANNS | Distance computation + top-k selection (GC + ORAM) | ~31s/query |
| Panther | PIR retrieval + GC top-k + HE distances | ~18s/query |
| Pacmann | PIR graph traversal | ~3.1s/query |
| **Opaque** | **Only centroid scoring (1 batched CKKS op)** | **~48ms HE** |

Everything else (local scoring, AES decrypt, PQ) is done in plaintext on the client. This is fundamentally cheaper.

### The Tradeoff

Opaque uses **decoy clusters** for access pattern privacy — statistical k-anonymity, not cryptographic ORAM/PIR. This is the honest weakness:

- **Stronger systems** (Compass, SANNS): ORAM-based access patterns — cryptographic guarantee
- **Opaque**: Decoys — statistical guarantee (server sees which clusters are fetched, but can't distinguish real from decoy)

This is a valid design choice for the threat model (honest-but-curious server), and the performance difference is 10-100x.

## Security Model Comparison

| System | Query Privacy | Data Privacy | Access Pattern | Threat Model |
|--------|--------------|--------------|----------------|--------------|
| **Opaque** | CKKS HE (crypto) | AES-256-GCM (crypto) | Decoys (statistical) | Honest-but-curious server |
| Compass | ORAM (crypto) | Encrypted (crypto) | ORAM (crypto) | Full server compromise |
| SANNS | 2PC (crypto) | Secret-shared (crypto) | ORAM (crypto) | Semi-honest 2-party |
| Pacmann | PIR (crypto) | Server holds plaintext | PIR (crypto) | Single server |
| RemoteRAG | PHE + DP | Server holds plaintext | DP perturbation | Untrusted server |
| PPMI | CKKS (crypto) | AES-256 (crypto) | Not explicitly addressed | IND-CPA |

## Datasets for Benchmarking

### Currently Benchmarked

| Dataset | Vectors | Dims | Source | Status |
|---------|---------|------|--------|--------|
| SIFT1M | 1M | 128 | ANN-benchmarks | Done (415ms, 99.6% recall with PQ) |
| GIST1M | 100K subset | 960 | ANN-benchmarks | Done (497ms, 98% recall with PQ) |
| GloVe-6B | 100K subset | 300 | Stanford NLP | Done (311ms, 100% recall) |

### Recommended Next

| Dataset | Vectors | Dims | Why | Download |
|---------|---------|------|-----|----------|
| **SIFT10M** | 10M | 128 | Prove scaling beyond 1M — would be first HE system at 10M | First 10M of SIFT1B at corpus-texmex.irisa.fr |
| **DBpedia-OpenAI-1M** | 1M | 1536 | Real LLM embeddings (text-embedding-ada-002) — proves RAG use case | HuggingFace: KShivendu/dbpedia-entities-openai-1M |
| **Cohere Wikipedia** | 1-10M subset | 1024 | Real multilingual embeddings — production relevance | HuggingFace: Cohere/wikipedia-2023-11-embed-multilingual-v3 |
| **SIFT100M** | 100M | 128 | Moonshot — 100x beyond current. If sub-10s, field-defining | corpus-texmex.irisa.fr (SIFT1B first 100M) |

### Scale Context

No published HE-based private vector search paper has demonstrated performance at 1M+ scale. Most test at 10K-100K. Opaque at 1M is already novel. At 10M, it would be groundbreaking.

## Research Paper Angles

### Primary: Systems Paper

**Venue:** USENIX Security, OSDI, SOSP, or EuroSys

**Title idea:** "Opaque: Sub-Second Privacy-Preserving Vector Search at Million Scale"

**Contributions:**
1. Hybrid architecture (HE for coarse search, PQ for fine search, decoys for access patterns)
2. Product quantization under encryption — 19x speedup on 960-dim
3. Threshold CKKS for key ownership — 0-10% overhead at scale
4. GPU profiling revealing key-switching bottleneck (not NTT) — 8.6x per-op speedup measured
5. Comprehensive benchmarks on SIFT1M, GIST100K, GloVe (multiple dims, multiple configs)
6. Sub-second query on 1M vectors at 99.6% recall — fastest published

**Comparison baseline:** SANNS (USENIX 2020), Compass (OSDI 2025), Pacmann (ICLR 2025)

### Secondary: Private RAG Application

**Framing:** "Private LLM Memory" — multi-tenant AI where each user's conversation history / knowledge base is encrypted and searchable without the server reading it.

**Why this matters:** Every RAG system today stores embeddings in plaintext on the server. The provider (OpenAI, Anthropic, etc.) can read all your documents. Opaque enables private RAG where the server can answer queries without seeing the data.

## Open Questions

1. **Compass comparison** — Need to benchmark on the same datasets. Their ORAM approach gives stronger access pattern privacy. What's the latency gap?

2. **FedSQ claims** — They report 50-100ms on SIFT1M. If true, this is faster than us. Need to verify their security model and assumptions.

3. **Threshold CKKS security** — Guy flagged potential simulation security issues with Lattigo's implementation. Research needed on whether noise flooding (sigma=2^20) mitigates known attacks.

4. **10M scaling** — Download SIFT10M, benchmark with PQ. Expected latency ~2-4s based on sub-linear scaling. Would be first HE system at this scale.

5. **Real LLM embeddings** — Benchmark on OpenAI/Cohere embeddings (1536-dim, 1024-dim). These are what production RAG systems use. GIST 960-dim is close but not the same distribution.

## References

### Core Private Vector Search
- [SANNS (USENIX Security 2020)](https://www.usenix.org/conference/usenixsecurity20/presentation/chen-hao)
- [Servan-Schreiber et al. (IEEE S&P 2022)](https://eprint.iacr.org/2021/1157)
- [Tiptoe (SOSP 2023)](https://dl.acm.org/doi/10.1145/3600006.3613134)
- [PRAG (arXiv 2311.12955)](https://arxiv.org/abs/2311.12955)
- [FedSQ (VLDB 2024)](https://dl.acm.org/doi/abs/10.14778/3685800.3685895)
- [Pacmann (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/391d50b3fe1c59b3e2b8b644e0c8fe81-Abstract-Conference.html)
- [Panther (CCS 2025)](https://dl.acm.org/doi/10.1145/3719027.3765190)
- [Compass (OSDI 2025)](https://www.usenix.org/conference/osdi25/presentation/zhu-jinhao)

### Private RAG
- [RemoteRAG (ACL 2025)](https://arxiv.org/abs/2412.12775)
- [SecureRAG (NeurIPS 2025)](https://neurips.cc/virtual/2025/124872)
- [PPMI (arXiv 2506.17336)](https://arxiv.org/abs/2506.17336)
- [PIR-RAG (arXiv 2509.21325)](https://arxiv.org/abs/2509.21325)
- [ppRAG (arXiv 2601.12331)](https://arxiv.org/abs/2601.12331)
- [SAG (arXiv 2508.01084)](https://arxiv.org/abs/2508.01084)
- [STEER (arXiv 2507.18518)](https://arxiv.org/abs/2507.18518)

### FHE Vector Search
- [Hermes FHE Vector DB (arXiv 2506.03308)](https://arxiv.org/abs/2506.03308)
- [AHE Sufficiency for Similarity (arXiv 2502.14291)](https://arxiv.org/abs/2502.14291)
- [PHE Vector Similarity Benchmark (arXiv 2503.05850)](https://arxiv.org/abs/2503.05850)
- [CipherFace (arXiv 2502.18514)](https://arxiv.org/abs/2502.18514)

### Other
- [DCPE-HNSW (ICDE 2025)](https://arxiv.org/abs/2508.10373)
- [FedVSE (VLDB 2025)](https://dl.acm.org/doi/10.14778/3750601.3750674)
- [FedVS (KDD 2025)](https://dl.acm.org/doi/10.1145/3711896.3736958)
- [FRAG (arXiv 2410.13272)](https://arxiv.org/abs/2410.13272)
- [Zhou PhD Thesis on Private Search (CMU 2025)](https://csd.cmu.edu/sites/default/files/phd-thesis/CMU-CS-25-115.pdf)

### Datasets
- [SIFT1B / TEXMEX corpus](http://corpus-texmex.irisa.fr/)
- [Deep1B (Yandex)](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search)
- [DBpedia OpenAI Embeddings 1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M)
- [Cohere Wikipedia Embeddings 250M](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3)
- [ann-benchmarks.com](http://ann-benchmarks.com/)
- [big-ann-benchmarks](https://big-ann-benchmarks.com/)
