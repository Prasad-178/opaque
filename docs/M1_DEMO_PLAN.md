# Opaque M1 Demo — Private Symptom-to-Condition Search

A live, interactive demo showcasing Opaque's privacy-preserving vector search applied to medical literature. A visitor types symptoms in plain English; the query is encrypted client-side, sent to an Opaque-indexed PubMed-derived embedding database, and the server returns top-K matching conditions with citations — all without ever decrypting the query or learning which records matched.

Hosted as an embedded widget on `prasadjs.me/opaque` and as a standalone page at `opaque-demo.<host>`.

---

## 1. Why M1 (Symptoms → Conditions)

- **Visceral**: visitors viscerally feel "I just told a server my symptoms — and it can't see them." Privacy story lands in 5 seconds.
- **Cold-demo friendly**: doesn't require visitors to bring their own data. Anyone can try it immediately.
- **Real domain**: medical privacy is the canonical example for HIPAA / health-data sensitivity, so the use case feels grounded, not contrived.
- **Showcases Opaque's actual benchmarks**: 99.8% Recall@10 at 345ms on 1M vectors → "ask about your symptoms in under half a second, fully encrypted."
- **Shareable**: friction-free demo links well on Twitter, HN, privacy newsletters.

---

## 2. User Flow

1. Visitor lands on `/opaque/demo` (or scrolls to embedded widget on `/opaque`).
2. Single dark-themed input field: "Describe your symptoms (e.g., 'persistent dry cough, fatigue, low-grade fever for 2 weeks')".
3. Visitor types, hits enter.
4. Client-side: text is embedded locally (via WebLLM-loaded MiniLM or BGE-small in-browser), then encrypted with CKKS via WASM.
5. Encrypted vector sent to Opaque server.
6. Server runs threshold-CKKS homomorphic similarity search over encrypted PubMed-derived embedding DB.
7. Returns encrypted top-K result IDs.
8. Client decrypts; UI renders top-K matched conditions with PubMed citations.
9. Side panel: "What just happened" — live trace showing data was never seen by the server.

---

## 3. Data Source

- **PubMed Central** open-access subset.
- For each condition (e.g., from MeSH descriptors or ICD-10 codes), gather top symptom-bearing abstracts.
- Generate embeddings using BGE-small or MiniLM-L6 (matching what we can run in-browser for query side).
- Index ~100k condition–symptom embeddings as v1 dataset (well within Opaque's 1M-vector benchmark range).

Disclaimer: educational demo only, not medical advice. Loud disclaimer in UI + footer.

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────┐
│  Browser (visitor)                                   │
│  - Loads embedding model (MiniLM-L6, ~25MB) via WebLLM/Transformers.js │
│  - User types symptoms                               │
│  - Local embedding generation                        │
│  - CKKS encryption of embedding via WASM             │
│  - Sends encrypted vector to Opaque server           │
│  - Receives encrypted top-K, decrypts locally        │
│  - Renders matched conditions + PubMed links         │
└─────────────────────────────────────────────────────┘
                              ↑↓ HTTPS
┌─────────────────────────────────────────────────────┐
│  Opaque server (existing Go service)                 │
│  - Encrypted vector DB: 100k condition–symptom embs  │
│  - Threshold-CKKS distributed key committee (3-of-5) │
│  - Homomorphic similarity search                     │
│  - Returns encrypted top-K record IDs                │
│  - Zero query/data/result leakage                    │
└─────────────────────────────────────────────────────┘
```

### Reused Opaque components

- `internal/ckks` — CKKS scheme params, keygen, homomorphic ops.
- `internal/threshold` — 3-of-5 distributed key committee.
- `internal/index` — encrypted vector index + similarity search.
- `http-server` — gRPC/HTTP front-end (extend with new `/v1/symptom-search` endpoint).
- `services/key-committee` — committee node infra.

### New components for this demo

- **Browser embedder**: transformers.js loading MiniLM-L6 (~25MB initial download, cached IndexedDB on subsequent visits).
- **WASM CKKS client**: existing `pkg/client` Go code compiled to WASM via TinyGo.
- **Web UI**: dark-themed single-page React app, matches `prasadjs.me` aesthetic.
- **PubMed ingestion pipeline**: one-time script in `scripts/ingest-pubmed.go` that fetches PubMed abstracts, generates embeddings server-side (using same MiniLM-L6 to keep model parity), encrypts with collection-public-key, indexes in Opaque.

---

## 5. UI Spec

### Layout

- Single-column, max-width 720px, centered.
- Input field, prominent, mono-font for typing feel.
- Submit button: `🔐 search privately`.
- Below: results area (initially empty, shows skeleton during search).
- Right panel (or below on mobile): "what just happened" trace — collapsed by default, expandable.

### Trace panel content

Live-updating during a query:

```
[client] embedded "<query>" → 384-dim vector locally
[client] CKKS-encrypted vector → 8.2 KB ciphertext
[client] sent ciphertext to opaque-demo.host
[server] received ciphertext (cannot decrypt)
[server] running 100k homomorphic similarity comparisons
[server] returned encrypted top-10 result IDs (256 bytes)
[client] decrypted result IDs locally → rendered conditions
elapsed: 412ms
```

### Result rendering

- Top-K = 10.
- Each result: condition name + 1-line summary + PubMed citation link.
- Caveat banner above results: "Educational demo. Not medical advice. See a real doctor."

### Theme

- Match `prasadjs.me`: `#1a1a1a` background, `#e5e5e5` text, Space Grotesk font, white accents on hover.
- Subtle animation on lock-icon during encryption step (300ms ease).

---

## 6. Performance Targets

| Metric | Target | Source |
|--------|--------|--------|
| Cold load (first visit) | < 5 sec | Embedding model download dominates |
| Warm load (return visit) | < 500 ms | Model cached in IndexedDB |
| Query end-to-end | < 700 ms | Includes embed + encrypt + RPC + decrypt |
| Server homomorphic search | < 350 ms | Already benchmarked: 345ms on 1M vectors, 100k will be faster |
| Recall@10 | > 99% | Already benchmarked: 99.8% |
| Concurrent QPS | > 20 | On 8-vCPU instance |

---

## 7. Security & Privacy

- **No query logging**: server config explicitly disables access logs; nginx/caddy in front configured to drop request bodies.
- **Threshold-CKKS**: 3-of-5 key committee — no single party can decrypt even server-side data.
- **Public auditability**: all server config, infra, deploy scripts published in same repo (`opaque/deploy/`); claim of zero-leak verifiable.
- **Strong client-side**: embedding model runs in-browser, query never leaves client unencrypted.
- **Educational disclaimer**: prominent banner; no medical claims; PubMed links only.

---

## 8. Build Phases

### Phase 1 — Data ingestion (3 days)
- Write `scripts/ingest-pubmed.go` — pulls PubMed abstracts via E-utilities API, filters to condition–symptom subset, generates MiniLM-L6 embeddings, formats for Opaque indexer.
- Index 100k records into Opaque.

### Phase 2 — Browser embedder (2 days)
- Set up transformers.js with MiniLM-L6.
- Verify embedding parity with server-side embeddings (cosine sim > 0.99 on test pairs).
- Cache model in IndexedDB.

### Phase 3 — WASM client crypto (3 days)
- Compile existing Go CKKS client code to WASM via TinyGo.
- Validate keygen, encrypt, decrypt round-trip in browser.
- Benchmark in-browser encryption time.

### Phase 4 — Server endpoint (2 days)
- Add `/v1/symptom-search` to `http-server/`.
- Wires existing Opaque index + threshold committee.
- Handles batched query if needed.

### Phase 5 — Web UI (3 days)
- React + Vite + TypeScript single-page app.
- Dark theme matching prasadjs.me.
- Trace panel.
- Submit + results loop.

### Phase 6 — Embed widget (2 days)
- Iframe-able variant for portfolio embed.
- Configurable theme tokens for embedding sites.

### Phase 7 — Polish + launch (1 week)
- Loading states, error handling, mobile responsiveness.
- Disclaimer copy review.
- Launch blog post on prasadjs.me explaining cryptography.
- Twitter thread + HN post.

---

## 9. Tech Decisions

- **Embedding model**: MiniLM-L6 (384-dim, ~25MB), small enough for browser, strong enough for symptom-condition matching.
- **In-browser ML**: transformers.js (Hugging Face) — battle-tested, IndexedDB caching built-in.
- **WASM toolchain**: TinyGo for Go-to-WASM (existing CKKS client is in Go).
- **Web framework**: React 19 + Vite + TypeScript (matches portfolio stack).
- **Server**: existing `http-server/` extended with new endpoint.
- **Hosting**: server on Fly.io or Hetzner (8-vCPU instance for homomorphic ops); static frontend on Cloudflare Pages.
- **Domain**: `opaque-demo.<your-domain>` and embedded at `prasadjs.me/opaque`.

---

## 10. Open Decisions

- Stick with PubMed as source, or also include something like NHS Conditions / WebMD-style structured data for cleaner condition labels?
- Result rendering: list of conditions, or richer card with snippet from abstract?
- Should the demo include a "show me the ciphertext" mode that displays the raw 8KB ciphertext so visitors can see it's truly opaque?
- Whether to include a side-by-side "vs unencrypted search" comparison (toggle button) — strong educational value but doubles complexity.
- Embed-widget: how to handle theme inheritance from host site?

---

## 11. Hard Constraints

- Realfy must NEVER appear in any artifact related to this demo.
- All PubMed data is public domain — no licensing concerns.
- Disclaimers must make clear this is educational, not medical advice.
- No logging of queries, results, or user data anywhere — verifiable in published config.
- Embed widget must work without breaking host-site CSP (sandbox iframe properly).

---

## 12. Success Criteria

- Live demo at `prasadjs.me/opaque/demo` and standalone domain.
- Cold-load → first result in < 5 sec; warm-load < 1 sec.
- 1k+ unique visitors in first month post-launch.
- Embed picked up by at least one external privacy/security blog or newsletter.
- Performance numbers visible in trace panel match published benchmarks (validates Opaque's claims to anyone curious enough to look).
- Becomes the canonical "show me what privacy-preserving vector search actually looks like" demo on the internet.

---

## 13. fhe-attest — Verifiable FHE Compute (Opaque Extension, Sister Project)

A separate but related research project that closes a real trust gap in Opaque (and in the M1 demo above).

### The gap

In M1 today, the visitor must trust that the server actually ran the homomorphic similarity search circuit they were promised — rather than (a) running a different circuit that selectively reveals query info via crafted noise, or (b) substituting a planted database that maps suspicious queries to attacker-chosen results, or (c) silently degrading recall to harvest re-query signals over time. CKKS hides the data, but does not prove the computation was the one declared.

### The idea

`fhe-attest` produces a **succinct ZK proof that a homomorphic evaluation was performed correctly** against a published circuit and a committed database. Server returns: encrypted result + ZK proof. Client verifies the proof in milliseconds before trusting the result.

Concretely:
- Database root is committed publicly (Merkle / Verkle / KZG over encrypted vectors).
- Circuit (similarity-search-and-top-K homomorphic program) compiled to a verifiable IR.
- Each query produces a SNARK / STARK proof: "I ran circuit `C` over database with commitment `D` and input ciphertext `Q`, producing output ciphertext `R`."
- Client verifies in < 50ms before decrypting.

### Why it's worth building

- **Frontier research**: verifiable FHE is an open problem with active interest (Marlin/Plonky3 over arithmetic FHE traces; recent zkFHE proposals from Zama, Fhenix research). No production-grade library exists. Real paper-worthy gap.
- **Closes Opaque's last trust assumption**: Opaque already proves data doesn't leak. fhe-attest proves *the right computation happened*. Together: end-to-end verifiable private compute.
- **Reuses Opaque's Go infra + lattigo stack**: same FHE library, additive build.
- **Universal beyond Opaque**: any FHE-based product (private RAG, private inference, private analytics) needs this primitive. Could be the standard library for the space.

### Cryptographic approach (sketch)

Three viable paths:

1. **SNARK over the FHE evaluation trace** — represent each homomorphic op (rotation, addition, multiplication, key-switch) as constraints in a SNARK system (Plonky3 / Halo2 / Nova). Generates a proof that the trace is consistent with the declared circuit and inputs. **High effort, most general.**
2. **Sumcheck / GKR over arithmetic circuit** — FHE evals are giant arithmetic circuits over a polynomial ring. GKR-style sumcheck protocols (Gemini, HyperPlonk variants) can prove correctness with sub-linear proof size. Likely best perf trade-off for the ciphertext sizes Opaque uses. **Medium effort, paper-friendly.**
3. **TEE attestation hybrid (escape hatch)** — run FHE eval in an SGX/TDX/Nitro enclave, attestation acts as proof of correct execution. Weaker security model (trust the TEE vendor) but ships in weeks. **Low effort, weaker guarantees.**

Recommend: ship a TEE-attested v0 in 2-3 weeks for the M1 demo (immediate trust improvement); pursue sumcheck-based v1 over 3-6 months as the research artifact.

### Architecture changes vs Opaque today

```
┌────────────────────────────────┐
│  Opaque server (existing)       │
│  + circuit registry (declared)  │
│  + db commitment publisher       │
│  + proof generator (new)         │
└────────────────────────────────┘
              ↓
        Enc(R) + π
              ↓
┌────────────────────────────────┐
│  Client                          │
│  + verifier (WASM, < 50ms)       │
│  + fetches db_commitment + circuit│
│  + verifies π before decrypting  │
└────────────────────────────────┘
```

### Build phases (independent of M1)

1. **Phase 1 — TEE-attested v0** (2 weeks): wrap existing Opaque homomorphic search in AWS Nitro enclave, ship Nitro attestation alongside results. Write client-side verifier.
2. **Phase 2 — Circuit registry + db commitment** (1 week): publish hash of declared homomorphic circuit + Merkle root of encrypted DB. Anyone can audit.
3. **Phase 3 — Sumcheck prover for FHE traces** (2-3 months): research-grade. Implement GKR-style protocol over CKKS evaluation traces. Benchmark proof size + verify time.
4. **Phase 4 — Paper draft** (1 month): write up for IACR ePrint / submit to PETS / RWC / TCC.
5. **Phase 5 — Wire into M1 demo** (1 week): visitor sees "✓ proof of correct execution verified" in trace panel. Real differentiator.

### Open questions

- TEE acceptable for v0 trust model, or skip directly to cryptographic proof?
- Which proof system best fits CKKS arithmetic (Plonky3, Nova, sumcheck-based)?
- Is the right output a paper, an OSS library, or both?
- Does it warrant a separate folder (`fhe-attest/`) at the projects root, or stay as Opaque extension?

### Hard constraints

- Realfy must NEVER appear in any artifact, paper, or repo related to this work.
- All research artifacts (paper draft, code) OSS from day 1 for trust/verifiability.
- TEE v0 must be marked "weak attestation, trust the vendor" — no overclaiming.

---

## 14. Future Variants (post-M1, post-fhe-attest)

- **M4 — Private journaling app**: visitor brings own data, search across encrypted personal notes. Standalone product, deeper engagement, return visits. Defer to v2.
- **R1 — Private email search**: same primitive, different dataset. Could be its own consumer product.
- **Encrypted Slack search**: B2B variant.

These all share Opaque infra; M1 establishes the live-demo pattern, others extend it.
