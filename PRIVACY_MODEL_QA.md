# Project Opaque: Privacy Model Q&A

This document explains the privacy model of Project Opaque, common misconceptions, and what problems it solves (and doesn't solve).

> **Note**: This document covers all tiers. For Tier 2.5 (Hierarchical Private Search) architecture details, see [docs/TIER_2_5_ARCHITECTURE.md](docs/TIER_2_5_ARCHITECTURE.md).

---

## Table of Contents

1. [What Problem Does This Project Solve?](#what-problem-does-this-project-solve)
2. [Privacy Tiers Overview](#privacy-tiers-overview)
3. [Common Misconceptions](#common-misconceptions)
4. [The Two-Stage Search Explained (Tier 1)](#the-two-stage-search-explained-tier-1)
5. [The Three-Level Search Explained (Tier 2.5)](#the-three-level-search-explained-tier-25)
6. [What the Server Sees vs Doesn't See](#what-the-server-sees-vs-doesnt-see)
7. [Privacy Tradeoffs and Tunable Parameters](#privacy-tradeoffs-and-tunable-parameters)
8. [Threat Model](#threat-model)
9. [Use Cases: Good Fit vs Bad Fit](#use-cases-good-fit-vs-bad-fit)
10. [Technical Deep Dive: Encryption Used](#technical-deep-dive-encryption-used)
11. [FAQ](#faq)

---

## What Problem Does This Project Solve?

### The Problem: Private Query on a Shared Database

You want to search a database that **someone else owns** without revealing **what you're searching for**.

**Example:**
- A hospital has a medical research database (100,000 papers)
- You want to search for "HIV treatment side effects"
- You don't want the hospital to know you searched for HIV-related topics

**Without Opaque:**
```
You → "Search for HIV treatment" → Hospital
Hospital → Returns results
Hospital also logs: "User X searched for HIV treatment"
```

**With Opaque:**
```
You → [Encrypted query hospital can't read] → Hospital
Hospital → [Computes on encrypted data] → [Encrypted results]
You → [Decrypt locally to see results]
Hospital only knows: "User X searched for something in the medical treatment region"
```

### What It Does NOT Solve

This project does **not** solve the problem of storing your own private data on an untrusted server. If you want to:
- Upload YOUR confidential documents to a cloud server
- Keep them secret from the server operator
- Search them privately

That requires a **different architecture** (see "Fully Encrypted Database" section below).

---

## Privacy Tiers Overview

Project Opaque offers multiple privacy tiers to match different use cases:

| Tier | Name | Query Privacy | Data Privacy | Best For |
|------|------|---------------|--------------|----------|
| **Tier 1** | Query-Private | Yes (HE) | No | Searching shared databases privately |
| **Tier 2** | Data-Private | Yes (local) | Yes (AES) | Storing your own encrypted data |
| **Tier 2.5** | Hierarchical Private | Yes (HE) | Yes (AES) | Maximum crypto privacy for both |

### Tier 1: Query-Private

- Your query is encrypted with homomorphic encryption
- Server computes on encrypted query, never sees it
- Server's vectors are in plaintext (server owns them)
- Fast: ~66ms for 100K vectors

### Tier 2: Data-Private

- Vectors are encrypted with AES-256-GCM before storage
- Query is computed locally (never sent to server)
- Server only stores encrypted blobs, can't read them
- Medium: ~50-200ms depending on bucket size

### Tier 2.5: Hierarchical Private (Recommended)

- **Level 1**: HE scoring on 64 centroids (server can't see query OR which buckets selected)
- **Level 2**: Decoy-based bucket fetch (server can't distinguish real from fake requests)
- **Level 3**: Local AES decrypt + scoring (all computation client-side)
- ~700ms for 100K vectors, maximum privacy

---

## Common Misconceptions

### Misconception 1: "The 100K vectors should be encrypted"

**Reality:** The vectors belong to the SERVER. It's their database. They already know what's in it.

Think of it like Google Search:
- Google owns the index of web pages
- You search Google
- The privacy question is: "Can Google see what I searched for?"
- NOT: "Can Google see their own web index?"

### Misconception 2: "If the server sees the top 10 vectors, they know my search"

**Reality:** The server sees WHICH vectors you're interested in, but:
- They don't see your EXACT query
- They don't see the similarity SCORES
- They don't see which result was BEST for you

It's like a librarian knowing you browsed the "medical" section, but not knowing which specific disease you were researching.

### Misconception 3: "An attacker can decode vectors back to text"

**Reality:** Yes, if you have an embedding model, you can understand what vectors represent. But the server ALREADY knows what those vectors represent—they created them!

The privacy isn't about hiding the database contents. It's about hiding YOUR interest in specific items.

### Misconception 4: "More candidates (200 → 500) means less privacy"

**Reality:** It's nuanced.
- More candidates = server sees a LARGER region of your interest
- But also = more noise, harder to pinpoint your EXACT interest
- Fewer candidates = narrower, more precise leak of your interest

The real privacy lever is **LSH bits**, not candidate count.

### Misconception 5: "This provides complete anonymity"

**Reality:** No. The server learns:
- Approximate region of your query (via LSH hash)
- Which candidates were considered
- Timing information

It's REDUCED information leakage, not ZERO leakage.

---

## The Two-Stage Search Explained (Tier 1)

### Why Two Stages?

Homomorphic encryption is slow (~33ms per vector). For 100K vectors:
```
100,000 × 33ms = 55 minutes per query (unusable)
```

So we filter first, then encrypt-compute on fewer vectors.

### Stage 1: Coarse Filtering (LSH)

**Tech:** Locality-Sensitive Hashing (random hyperplanes)

1. Client computes a "fingerprint" (hash) of their query locally
2. Client masks the hash with a session key (privacy protection)
3. Server finds vectors with similar fingerprints
4. Returns 200 candidate IDs

**Time:** ~5ms
**Server learns:** Approximate region of query

### Stage 2: Fine Scoring (Homomorphic Encryption)

**Tech:** BFV Homomorphic Encryption (Lattigo)

1. Server takes top 10 candidates (by fingerprint similarity)
2. Server computes encrypted dot product: E(query) · vector = E(score)
3. Server returns 10 encrypted score blobs
4. Client decrypts locally, ranks results

**Time:** ~56ms (parallelized)
**Server learns:** Nothing (scores are encrypted)

### The Funnel

```
100,000 vectors
     │
     ▼ LSH fingerprint matching (~5ms)
   200 candidates
     │
     ▼ Hamming distance ranking (~0ms)
    10 candidates
     │
     ▼ Homomorphic encryption scoring (~56ms)
    10 encrypted scores
     │
     ▼ Client decryption + ranking (~3ms)
  Final results
```

**Total: ~66ms instead of 55 minutes**

---

## The Three-Level Search Explained (Tier 2.5)

### Why Three Levels?

Tier 2.5 provides **both query privacy AND data privacy**. The three levels achieve this:

1. **Level 1 (HE)**: Hide the query AND which buckets were selected
2. **Level 2 (Decoys)**: Hide which buckets you're actually interested in
3. **Level 3 (AES)**: Hide the vector contents from storage

### Level 1: HE Centroid Scoring

**Tech:** BFV Homomorphic Encryption (same as Tier 1)

1. Vectors are organized into 64 super-buckets, each with a centroid (mean vector)
2. Client encrypts query with HE: `HE(query)`
3. Server computes `HE(query) · centroid[i]` for ALL 64 centroids
4. Server returns 64 encrypted scores
5. Client decrypts locally, selects top-K buckets

**Key insight:** Server computes ALL scores blindly. It never knows which buckets the client selected!

**Time:** ~640ms (dominated by 64 HE dot products)

### Level 2: Decoy-Based Bucket Fetch

**Tech:** Decoys + Shuffling (optional: PIR)

1. Client determines real buckets to fetch (from Level 1 selection)
2. Client generates decoy bucket IDs from non-selected super-buckets
3. Client shuffles all bucket IDs together
4. Client requests all buckets from server
5. Server returns encrypted blobs for all requested buckets

**Key insight:** Server sees bucket IDs but can't tell which are real vs decoy.

**Time:** ~10-50ms (network-dependent)

### Level 3: Local AES Decrypt + Scoring

**Tech:** AES-256-GCM symmetric encryption

1. All vectors are encrypted at rest with AES-256-GCM
2. Client decrypts fetched vectors locally
3. Client computes cosine similarity locally
4. Client returns top-K results

**Key insight:** Server never sees decrypted vectors or final scores.

**Time:** ~15-30ms for ~1500 vectors

### The Funnel (Tier 2.5)

```
100,000 vectors in 64 super-buckets
     │
     ▼ HE scoring on 64 centroids (~640ms)
     │ Server computes blindly, client decrypts privately
     │
   64 encrypted scores
     │
     ▼ Client selects top 8 super-buckets (SERVER NEVER SEES THIS)
     │
     ▼ Add 8 decoy buckets, shuffle
     │
  16 bucket requests (8 real + 8 decoy)
     │
     ▼ Fetch encrypted blobs (~10-50ms)
     │
 ~12,000 encrypted vectors
     │
     ▼ AES decrypt + local scoring (~20ms)
     │
  Final top-K results

Total: ~700ms for 100K vectors
```

### Why HE for Centroids? Why Not Just Use AES?

Great question! HE is essential here because:

| Without HE | With HE |
|------------|---------|
| Client sends plaintext query to server | Client sends HE(query) to server |
| Server sees query, knows what you're searching | Server computes blindly on encrypted query |
| Server sees which buckets score highest | Server returns encrypted scores, can't decrypt |
| Server knows which buckets you'll select | Client decrypts privately, server never knows |

**AES can only encrypt/decrypt.** It cannot compute on encrypted data. HE allows the server to compute `score = query · centroid` without seeing the query or the score.

---

## What the Server Sees vs Doesn't See

### Tier 1: Server SEES (Information Leakage)

| What | How Much Leakage |
|------|------------------|
| LSH hash (masked) | Approximate query region |
| 200 candidate IDs | Vectors you might be interested in |
| 10 scored IDs | Vectors you're more likely interested in |
| Timing | When you search, how long it takes |
| Session patterns | Multiple queries can be correlated |

### Tier 1: Server Does NOT See

| What | Why Protected |
|------|---------------|
| Exact query vector | Encrypted with BFV, never sent in plaintext |
| Similarity scores | Computed on encrypted data |
| Final ranking | Client decrypts and ranks locally |
| Which result you used | All processing after decryption is local |

### Tier 2.5: Server SEES (Information Leakage)

| What | How Much Leakage |
|------|------------------|
| HE(query) | Encrypted blob, cannot decrypt |
| Bucket IDs fetched | But cannot distinguish real from decoy |
| Encrypted blobs | AES ciphertext, cannot read |
| Timing | When you search, how long it takes |

### Tier 2.5: Server Does NOT See

| What | Why Protected |
|------|---------------|
| Query vector | HE encrypted, server computes blindly |
| Centroid scores | HE encrypted, only client decrypts |
| Which super-buckets selected | Client-side decryption |
| Which buckets are real vs decoy | Shuffled together |
| Vector contents | AES-256-GCM encrypted |
| Similarity scores | Computed locally |
| Final ranking | Computed locally |

### Tier 2.5: Per-Enterprise LSH (Planned Enhancement)

With per-enterprise LSH, the server additionally cannot:

| What | Why Protected |
|------|---------------|
| Bucket-to-region mapping | LSH hyperplanes are enterprise secrets |
| Cross-enterprise correlation | Different enterprises have different LSH |

---

## Privacy Tradeoffs and Tunable Parameters

### Parameters That Affect Privacy

| Parameter | Default | More Privacy | Less Privacy |
|-----------|---------|--------------|--------------|
| LSH bits | 128 | Decrease to 64 (wider buckets) | Increase to 256 (narrow buckets) |
| Session key rotation | Manual | Rotate frequently | Never rotate |
| Session TTL | 24 hours | Shorter (1 hour) | Longer (7 days) |

### Parameters That Affect Speed

| Parameter | Default | Faster | Slower |
|-----------|---------|--------|--------|
| LSH candidates | 200 | Decrease to 100 | Increase to 500 |
| HE candidates | 10 | Decrease to 5 | Increase to 20 |
| Parallel workers | 12 | More CPU cores | Fewer cores |

### Parameters That Affect Accuracy

| Parameter | Default | More Accurate | Less Accurate |
|-----------|---------|---------------|---------------|
| LSH bits | 128 | Increase to 256 | Decrease to 64 |
| LSH candidates | 200 | Increase to 500 | Decrease to 100 |
| HE candidates | 10 | Increase to 20 | Decrease to 5 |

### Recommended Presets

| Preset | LSH Bits | Candidates | HE Scored | Latency | Privacy |
|--------|----------|------------|-----------|---------|---------|
| High Privacy | 64 | 100 | 5 | ~40ms | Excellent |
| Balanced | 128 | 200 | 10 | ~66ms | Good |
| High Accuracy | 256 | 500 | 20 | ~150ms | Lower |

---

## Threat Model

### What We Protect Against

**Honest-but-Curious Server:**
- Server follows the protocol correctly
- Server tries to learn your queries by analyzing data
- Protection: Query encryption, score encryption, hash masking

**Network Eavesdropper:**
- Attacker intercepts network traffic
- Protection: TLS encryption + HE means traffic is double-encrypted

**Database Breach (of encrypted data):**
- Attacker gets encrypted queries/scores
- Protection: Without secret key, ciphertexts are meaningless

### What We Do NOT Protect Against

**Malicious Server:**
- Server returns fake/manipulated results
- No protection (would need verifiable computation)

**Compromised Client:**
- If attacker has your secret key, all bets are off
- No protection (secure your keys!)

**Traffic Analysis:**
- Attacker observes query timing, sizes
- Partial protection (could add padding, delays)

**Long-term Statistical Analysis:**
- Attacker collects many queries, builds statistical model
- Partial protection (key rotation, session limits)

---

## Use Cases: Good Fit vs Bad Fit

### Good Fit (This Project Helps)

| Use Case | Why It Works |
|----------|--------------|
| Search medical literature without revealing symptoms | Hospital's data, your private query |
| Search legal database without revealing strategy | Law firm's data, your private query |
| Search market data without revealing trading signals | Data provider's data, your private query |
| Search job database without revealing hiring plans | LinkedIn's data, your private query |

### Bad Fit (Need Different Solution)

| Use Case | Why It Doesn't Work | What You Need |
|----------|---------------------|---------------|
| Store YOUR private docs on untrusted cloud | Vectors aren't encrypted at rest | Encrypted database |
| Complete anonymity from server | LSH leaks approximate region | PIR or ORAM |
| Prevent server from lying about results | No verification | Verifiable computation |
| Decentralized storage (blockchain) | Public ledger exposes everything | Client-side encryption + specialized protocols |

---

## Technical Deep Dive: Encryption Used

### Encryption Scheme: BFV (Brakerski-Fan-Vercauteren)

| Property | Value |
|----------|-------|
| Type | Somewhat Homomorphic / Leveled FHE |
| Security Level | 128-bit |
| Based On | Ring Learning With Errors (RLWE) |
| Quantum Resistant | Yes |
| Library | Lattigo v5 |

### What BFV Allows

- Addition on encrypted data: E(a) + E(b) = E(a+b)
- Multiplication on encrypted data: E(a) × b = E(a×b)
- Limited depth (enough for dot products)

### Why Not "Full" FHE?

Full FHE requires "bootstrapping" to refresh ciphertexts for unlimited operations. This adds 10-100x overhead. For dot products (just multiply + sum), BFV is sufficient and much faster.

### Security Guarantees

| Guarantee | Explanation |
|-----------|-------------|
| Semantic Security | Ciphertexts reveal nothing about plaintexts |
| IND-CPA | Attacker can't distinguish encryptions of different messages |
| 128-bit Security | Would take 2^128 operations to break |

---

## FAQ

### Q: Is the server's database encrypted?

**A:** No. The server owns its database and can see its contents. The encryption protects YOUR QUERY, not the server's data.

### Q: Can the server reverse-engineer my query from the LSH hash?

**A:** Partially. The LSH hash reveals the approximate "region" of your query, but not the exact vector. Hash masking with session keys prevents correlation across queries.

### Q: What if someone hacks the server and gets all the vectors?

**A:** They'd have the database contents, which the server already had access to. Your QUERIES (which were encrypted) would still be protected. Your query history would not be exposed.

### Q: How is this different from just using HTTPS?

**A:** HTTPS encrypts data in transit, but the server decrypts it to process. With Opaque:
- Server NEVER decrypts your query
- Server computes on ENCRYPTED data
- Server NEVER sees similarity scores

### Q: Can I use this to store my private embeddings?

**A:** This system is designed for querying someone else's database privately, not for storing your own encrypted data. For that use case, you'd need a different architecture (see fully encrypted database approaches).

### Q: What's the performance cost of privacy?

**A:** Depends on the tier:
- **Tier 1**: ~66ms (60x slower than unencrypted)
- **Tier 2**: ~50-200ms (depends on bucket size)
- **Tier 2.5**: ~700ms (700x slower, but maximum privacy)

Still fast enough for real-time applications in all cases.

### Q: Why does Tier 2.5 use both HE and AES?

**A:** They serve different purposes:
- **HE**: Allows server to compute on encrypted query without seeing it
- **AES**: Protects vectors at rest (storage never sees plaintext)

HE is for computation privacy. AES is for storage privacy. Together they provide maximum protection.

### Q: What is PIR and do I need it?

**A:** PIR (Private Information Retrieval) lets you fetch data from a server without the server knowing which data you fetched.

In Tier 2.5, we use **decoys** instead of PIR:
- Decoys: Request real buckets + fake buckets, server can't tell which are real
- PIR: Cryptographically hide which buckets you want (slower, more complex)

For most use cases, decoys provide sufficient privacy with better performance. PIR can be added as an optional enhancement for high-security scenarios.

### Q: What is per-enterprise LSH?

**A:** Currently, LSH hyperplanes are generated from a known seed. An attacker who knows the seed can map bucket IDs to query regions.

With per-enterprise LSH (planned):
- Each enterprise gets a secret, random LSH seed
- Only authenticated users receive the hyperplanes
- Server cannot map bucket IDs to query regions without the secret

This prevents the server from learning anything about query intent from bucket access patterns.

---

## Summary

### Tier 1 (Query-Private)

| Question | Answer |
|----------|--------|
| What does this protect? | Your search query and result preferences |
| What doesn't it protect? | The database contents (server owns them) |
| Is it 100% private? | No, LSH leaks approximate query region |
| Is it useful? | Yes, for searching shared databases privately |
| What's the latency? | ~66ms for 100K vectors |
| Is it quantum-safe? | Yes (lattice-based cryptography) |

### Tier 2.5 (Hierarchical Private)

| Question | Answer |
|----------|--------|
| What does this protect? | Query, data, bucket selection, results |
| What doesn't it protect? | Bucket access patterns (mitigated with decoys) |
| Is it 100% private? | Nearly - decoys + per-enterprise LSH provide strong guarantees |
| Is it useful? | Yes, for maximum privacy on both query AND data |
| What's the latency? | ~700ms for 100K vectors |
| Is it quantum-safe? | Yes (BFV is lattice-based) |

### Which Tier Should I Use?

| Use Case | Recommended Tier |
|----------|-----------------|
| Search a shared database privately | Tier 1 |
| Store your own encrypted data | Tier 2 |
| Maximum privacy (query + data) | Tier 2.5 |
| Enterprise with strict compliance | Tier 2.5 + per-enterprise LSH |
| Sub-100ms latency required | Tier 1 or Tier 2 |
