# Tier 2.5: Hierarchical Private Search Architecture

> **Version**: 2.0
> **Status**: Design Finalized
> **Last Updated**: 2026-02-06

This document describes the complete privacy architecture for Tier 2.5 Hierarchical Private Search, including the rationale for each design decision.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Privacy Goals](#privacy-goals)
3. [Architecture Overview](#architecture-overview)
4. [Component Deep Dive](#component-deep-dive)
   - [Level 1: HE Centroid Scoring](#level-1-he-centroid-scoring)
   - [Level 2: PIR or Decoy-Based Bucket Fetch](#level-2-pir-or-decoy-based-bucket-fetch)
   - [Level 3: Local AES Decrypt + Scoring](#level-3-local-aes-decrypt--scoring)
5. [PIR Analysis](#pir-analysis)
6. [Per-Enterprise LSH](#per-enterprise-lsh)
7. [Authentication Model](#authentication-model)
8. [What Each Party Sees](#what-each-party-sees)
9. [Why HE for Centroid Scoring](#why-he-for-centroid-scoring)
10. [Security Analysis](#security-analysis)
11. [Performance Characteristics](#performance-characteristics)
12. [Configuration Options](#configuration-options)
13. [Future Enhancements](#future-enhancements)

---

## Executive Summary

Tier 2.5 provides **both query privacy and data privacy** through a three-level hierarchical approach:

| Level | Purpose | Technology | Privacy Guarantee |
|-------|---------|------------|-------------------|
| **Level 1** | Find relevant super-buckets | Homomorphic Encryption | Server can't see query or which buckets were selected |
| **Level 2** | Fetch encrypted vectors | Decoys + Shuffling (PIR optional) | Server sees bucket access but can't distinguish real vs decoy |
| **Level 3** | Score and rank | AES-256-GCM + Local compute | All decryption and scoring happens client-side |

**Key Design Decisions**:
- **HE for centroid scoring**: Essential for hiding query from server during computation
- **Per-enterprise LSH**: Each enterprise has secret LSH hyperplanes, preventing bucket-to-query mapping attacks
- **Decoys over PIR**: Simpler, faster, sufficient for k-anonymity (PIR available as optional upgrade)
- **Option B authentication**: Token-based key distribution with rotation support

---

## Privacy Goals

The system is designed to achieve these specific privacy properties:

### Must Have

| Goal | Description | Achieved By |
|------|-------------|-------------|
| **Query Privacy** | Server cannot see what user is searching for | HE encryption of query |
| **Data Privacy** | Server/storage cannot see vector contents | AES-256-GCM encryption |
| **Selection Privacy** | Server cannot see which super-buckets were selected | Client-side decryption of HE scores |
| **Result Privacy** | Server cannot see final search results | Local scoring and ranking |

### Nice to Have

| Goal | Description | Achieved By |
|------|-------------|-------------|
| **Access Pattern Obfuscation** | Make it hard to correlate bucket access with query intent | Per-enterprise LSH + Decoys |
| **Temporal Unlinkability** | Queries from same user can't be correlated over time | Session key rotation |

### Acceptable Leakage

| Leakage | Justification |
|---------|---------------|
| **Bucket access patterns** | Buckets contain ~1500+ vectors, providing k-anonymity |
| **Timing information** | Can be mitigated with padding if needed |
| **Metadata (bucket sizes, access frequency)** | Doesn't reveal query or data contents |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TIER 2.5 HIERARCHICAL PRIVATE SEARCH                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     AUTHENTICATION SERVICE                          │   │
│  │                                                                     │   │
│  │  Per-Enterprise Secrets (distributed to authenticated users):      │   │
│  │    - AES-256 key (decrypt vectors)                                 │   │
│  │    - LSH hyperplanes (compute bucket mapping)                      │   │
│  │    - Centroids (64 vectors for HE scoring)                         │   │
│  │    - HE public parameters (for query encryption)                   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                     ┌───────────────┴───────────────┐                       │
│                     ▼                               ▼                       │
│  ┌──────────────────────────────┐  ┌──────────────────────────────────┐    │
│  │          CLIENT              │  │           SERVER                 │    │
│  │                              │  │                                  │    │
│  │  Has:                        │  │  Has:                            │    │
│  │  - Enterprise secrets        │  │  - HE public key                 │    │
│  │  - HE keypair (secret key)   │  │  - Plaintext centroids           │    │
│  │  - Query vector              │  │  - Encrypted vector blobs        │    │
│  │                              │  │                                  │    │
│  └──────────────────────────────┘  └──────────────────────────────────┘    │
│                                                                             │
│  LEVEL 1: HE CENTROID SCORING (~640ms for 64 centroids)                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Client                           Server                            │   │
│  │    │                                │                               │   │
│  │    │─── HE(query) ─────────────────►│                               │   │
│  │    │                                │ Compute: HE(query) · centroid │   │
│  │    │                                │          for ALL 64 centroids │   │
│  │    │◄── HE(score_0..63) ───────────│                               │   │
│  │    │                                                                │   │
│  │  Decrypt locally                                                    │   │
│  │  Select top-K super-buckets                                         │   │
│  │  SERVER NEVER SEES SELECTION                                        │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LEVEL 2: BUCKET FETCH (Decoys or PIR)                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Client computes:                                                   │   │
│  │    - LSH(query) → sub-bucket ID (using secret enterprise LSH)      │   │
│  │    - Generate decoy bucket IDs from non-selected super-buckets     │   │
│  │    - Shuffle all bucket IDs                                         │   │
│  │                                                                     │   │
│  │  Request: [bucket_07, bucket_23, bucket_45, bucket_12, ...]        │   │
│  │           (real and decoy mixed, shuffled)                          │   │
│  │                                                                     │   │
│  │  Server returns: encrypted blobs for all requested buckets         │   │
│  │  Server cannot distinguish real from decoy                          │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LEVEL 3: LOCAL DECRYPT + SCORING (~15-30ms)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Client:                                                            │   │
│  │    1. AES-256-GCM decrypt all fetched vectors                      │   │
│  │    2. Compute cosine similarity with query                         │   │
│  │    3. Sort by score, return top-K                                  │   │
│  │                                                                     │   │
│  │  All computation is LOCAL - server sees nothing                    │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### Level 1: HE Centroid Scoring

#### Purpose
Find which of the 64 super-buckets contain vectors most similar to the query, **without the server learning the query or which buckets were selected**.

#### How It Works

1. **Client encrypts query** using BFV homomorphic encryption
   - Security: 128-bit (RLWE assumption, quantum-resistant)
   - Parameters: LogN=14, LogQP=438

2. **Server computes HE dot products** with ALL centroids
   - Operation: `HE(query) · centroid[i] = HE(score[i])`
   - The server sees encrypted query, computes on it blindly
   - Server returns all 64 encrypted scores

3. **Client decrypts privately**
   - Only client has the HE secret key
   - Client sees all 64 scores in plaintext
   - Client selects top-K (e.g., top 8) super-buckets

4. **Privacy guarantee**: Server computed on encrypted data, never saw:
   - The query vector
   - The similarity scores
   - Which super-buckets were selected

#### Why HE Here? (Not AES)

This is a critical design decision. HE is essential because:

| Approach | Privacy | Why It Works/Fails |
|----------|---------|-------------------|
| **HE (chosen)** | Query hidden | Server computes `HE(query) · centroid = HE(score)` without seeing query |
| **Send plaintext query** | Query exposed | Server sees exactly what you're searching for |
| **Download centroids, compute locally** | Query hidden | Works for 64 centroids, but doesn't scale to 10,000+ |

**HE becomes essential when**:
- Centroids are numerous (can't download all)
- Centroids are confidential (shouldn't leave server)
- You need auditability (prove server computed correctly)

For 64 centroids, local computation is viable, but **HE future-proofs** the architecture.

### Level 2: PIR or Decoy-Based Bucket Fetch

#### Current: Decoy-Based Approach

After Level 1, the client knows which super-buckets to fetch, but must request buckets from the server without revealing which are "real".

**Decoy Strategy**:
```
1. Real buckets: top-K super-buckets (e.g., 8)
2. Decoy buckets: random buckets from non-selected super-buckets (e.g., 8)
3. Combine and shuffle all bucket requests
4. Server returns encrypted blobs for all buckets
5. Server cannot distinguish real from decoy
```

**Privacy guarantee**: Server sees 16 bucket requests (8 real + 8 decoy), shuffled. Without knowing which are decoys, server can only guess with 1/C(16,8) = 1/12870 probability.

#### Optional: PIR Enhancement

PIR (Private Information Retrieval) can replace or augment the decoy approach.

See [PIR Analysis](#pir-analysis) section for detailed comparison.

### Level 3: Local AES Decrypt + Scoring

All fetched vectors are encrypted with AES-256-GCM. The client:

1. **Decrypts** using the enterprise AES key (from authentication)
2. **Computes cosine similarity** locally
3. **Ranks and returns** top-K results

**Privacy guarantee**: The server never sees:
- Decrypted vectors
- Similarity scores
- Which vectors matched best
- Final results

---

## PIR Analysis

### What is PIR?

**Private Information Retrieval (PIR)** allows a client to fetch an item from a database without the server knowing which item was fetched.

```
Traditional Fetch:               PIR:
┌────────┐  "bucket 7"  ┌────┐   ┌────────┐  encrypted(7)  ┌────┐
│ Client │─────────────►│ DB │   │ Client │───────────────►│ DB │
└────────┘              └────┘   └────────┘                └────┘
                          │                                   │
Server knows: bucket 7    │      Server knows: NOTHING        │
was requested             │      (processes all buckets)      │
```

### Types of PIR

| Type | Description | Pros | Cons |
|------|-------------|------|------|
| **Computational PIR (CPIR)** | Single server, based on HE/lattices | No trust assumptions | Slow (server processes ALL data) |
| **Information-Theoretic PIR (ITPIR)** | Multiple non-colluding servers | Fast | Requires 2+ servers that don't collude |
| **Symmetric PIR (SPIR)** | Also hides database size | Maximum privacy | Very slow |

### PIR vs Decoys: Comparison

| Aspect | Decoys | Computational PIR | IT-PIR (2 servers) |
|--------|--------|-------------------|-------------------|
| **Privacy level** | k-anonymity (k = decoys) | Cryptographic | Information-theoretic |
| **Server trust** | Semi-honest | Semi-honest | Non-colluding |
| **Latency overhead** | ~2x (fetch extra buckets) | ~10-100x (HE on all buckets) | ~2-3x |
| **Bandwidth overhead** | ~2x | ~1x (server-side computation) | ~2x |
| **Implementation complexity** | Simple | Complex (HE integration) | Complex (multi-server) |
| **Scales with database** | O(1) | O(N) server work | O(N) total work split |

### Recommendation

**Start with decoys, add PIR as optional enhancement**:

1. **Decoys** provide sufficient privacy for most use cases:
   - With 8 real + 8 decoy buckets, attacker guessing probability is ~0.008%
   - Each bucket has ~1500 vectors, providing k-anonymity with k=1500

2. **PIR** is valuable when:
   - Regulatory requirements demand cryptographic (not probabilistic) guarantees
   - Bucket access patterns themselves are sensitive
   - Willing to accept 10-100x latency increase (CPIR) or multi-server setup (ITPIR)

3. **Implementation approach**:
   - Phase 1: Decoys (current)
   - Phase 2: Optional ITPIR (if 2-server deployment is acceptable)
   - Phase 3: Optional CPIR (when performance improves, or for low-frequency queries)

### PIR Integration Points

If PIR is added later, it integrates at Level 2:

```
LEVEL 2 (with PIR):

Option A: Replace decoys entirely
  - Client uses PIR to fetch exact buckets needed
  - Server cannot learn which buckets
  - Higher latency, maximum privacy

Option B: Hybrid (PIR + decoys)
  - Use PIR for some buckets (high-sensitivity queries)
  - Use decoys for others (performance-sensitive)
  - Runtime decision based on query sensitivity
```

---

## Per-Enterprise LSH

### The Problem

In the current implementation, LSH hyperplanes are generated from a known seed:

```go
// Current: Seed is fixed and known
LSHSuperSeed: 42
LSHSubSeed:   137
```

An attacker who knows the seed can:
1. Reconstruct the hyperplanes
2. For any bucket ID, determine what query region it represents
3. Correlate bucket access with query intent

### The Solution: Per-Enterprise Secret LSH

Each enterprise has its own **secret** LSH configuration:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ENTERPRISE CONFIGURATION                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Enterprise "Acme Corp":                                                │
│    LSH Seed:        0x7a3b9c4d... (cryptographically random, secret)   │
│    AES Key:         0xe4f2a1... (256-bit, secret)                      │
│    Centroids:       [64 vectors] (can be stored on server)             │
│                                                                         │
│  Enterprise "Beta Inc":                                                 │
│    LSH Seed:        0x2f8e1b7c... (different random seed)              │
│    AES Key:         0xb3c7d9... (different key)                        │
│    Centroids:       [64 vectors]                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Security Properties

| Property | Description |
|----------|-------------|
| **Bucket opacity** | Without LSH seed, server cannot map bucket ID to query region |
| **Cross-enterprise isolation** | Bucket "07" in Acme ≠ Bucket "07" in Beta |
| **Key rotation** | LSH can be regenerated, data re-bucketed (offline process) |

### Data Flow with Per-Enterprise LSH

```
INDEX TIME (Enterprise Admin):
┌────────────────────────────────────────────────────────────────┐
│  1. Generate enterprise secrets:                               │
│     - LSH seed (cryptographically random)                      │
│     - AES-256 key                                              │
│                                                                │
│  2. Generate LSH hyperplanes from seed                         │
│                                                                │
│  3. For each vector:                                           │
│     - Compute bucket_id = LSH(vector)                          │
│     - Encrypt: AES(vector, key)                                │
│     - Store: (bucket_id, encrypted_vector)                     │
│                                                                │
│  4. Compute centroids for each super-bucket                    │
│                                                                │
│  5. Store enterprise config in secure vault:                   │
│     - LSH seed                                                 │
│     - AES key                                                  │
│     - Centroids (or keep on server)                            │
└────────────────────────────────────────────────────────────────┘

QUERY TIME (Authenticated User):
┌────────────────────────────────────────────────────────────────┐
│  1. User authenticates → receives:                             │
│     - LSH hyperplanes (derived from secret seed)               │
│     - AES key                                                  │
│     - Centroids (cached locally)                               │
│     - Session token                                            │
│                                                                │
│  2. User computes locally:                                     │
│     - bucket_id = LSH(query) using secret hyperplanes         │
│     - HE(query) for centroid scoring                           │
│                                                                │
│  3. Server processes:                                          │
│     - HE scoring (sees encrypted query only)                   │
│     - Bucket fetch (sees bucket IDs, not their meaning)        │
│                                                                │
│  4. User decrypts and scores locally                           │
└────────────────────────────────────────────────────────────────┘
```

### What the Server Sees (Post Per-Enterprise LSH)

| Information | Visible? | Notes |
|-------------|----------|-------|
| Bucket IDs (e.g., "07", "23") | Yes | But can't map to query region without LSH seed |
| Encrypted vectors | Yes | Can't decrypt without AES key |
| HE(query) | Yes | Can't decrypt, can only compute on it |
| HE(scores) | Yes | Can't decrypt |
| Which buckets accessed | Yes | But meaning is opaque without LSH |

---

## Authentication Model

### Option B: Token-Based Key Distribution

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     AUTHENTICATION FLOW                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐         ┌──────────────┐         ┌──────────────────┐     │
│  │  User   │         │  Auth Server │         │  Key Vault       │     │
│  │ Client  │         │  (IdP)       │         │  (Secrets)       │     │
│  └────┬────┘         └──────┬───────┘         └────────┬─────────┘     │
│       │                     │                          │               │
│       │ 1. Login (OAuth/SSO)│                          │               │
│       │────────────────────►│                          │               │
│       │                     │                          │               │
│       │ 2. Verify identity  │                          │               │
│       │                     │                          │               │
│       │                     │ 3. Fetch enterprise keys │               │
│       │                     │─────────────────────────►│               │
│       │                     │                          │               │
│       │                     │◄─────────────────────────│               │
│       │                     │   AES key, LSH seed,     │               │
│       │                     │   centroids              │               │
│       │                     │                          │               │
│       │◄────────────────────│                          │               │
│       │  4. Return:         │                          │               │
│       │   - Session token   │                          │               │
│       │   - AES key (encrypted to user)                │               │
│       │   - LSH hyperplanes (derived from seed)        │               │
│       │   - Centroids (for local caching)              │               │
│       │   - Token TTL (e.g., 1 hour)                   │               │
│       │                                                │               │
│       │ 5. Store in secure local storage               │               │
│       │                                                                 │
└───────┴─────────────────────────────────────────────────────────────────┘
```

### Key Properties

| Property | Implementation |
|----------|----------------|
| **Short-lived tokens** | 1-hour TTL, refresh before expiry |
| **Key rotation** | Enterprise admin can rotate AES/LSH keys |
| **Secure distribution** | Keys encrypted in transit (TLS) and at rest |
| **Revocation** | Invalidate tokens immediately on logout/breach |
| **Audit logging** | Track who accessed which enterprise keys |

### Secrets Managed

| Secret | Purpose | Rotation Frequency |
|--------|---------|-------------------|
| **AES-256 key** | Decrypt vectors | Quarterly or on breach |
| **LSH seed** | Compute bucket mapping | Rarely (requires re-indexing) |
| **HE keypair** | Query encryption | Per-session or daily |
| **Session token** | API authentication | Hourly |

### Client-Side Key Storage

```
┌─────────────────────────────────────────────────────────────────┐
│                   CLIENT KEY STORAGE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Secure Storage (OS Keychain / Hardware Security Module):       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  enterprise_id: "acme-corp"                               │ │
│  │  aes_key: [encrypted, 256 bits]                           │ │
│  │  lsh_hyperplanes: [8 × 128D vectors]                      │ │
│  │  centroids: [64 × 128D vectors]                           │ │
│  │  session_token: "eyJ..."                                  │ │
│  │  token_expiry: 2026-02-06T15:00:00Z                       │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  On Token Expiry:                                               │
│  1. Refresh token with auth server                             │
│  2. Optionally fetch updated keys (if rotated)                 │
│  3. Continue operations                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## What Each Party Sees

### Summary Table

| Information | Client | Compute Server | Storage Server | External Attacker |
|-------------|--------|----------------|----------------|-------------------|
| Query vector (plaintext) | Yes | No | No | No |
| Query vector (HE encrypted) | Yes | Yes | No | No |
| Centroid scores (plaintext) | Yes | No | No | No |
| Centroid scores (HE encrypted) | Yes | Yes | No | No |
| Super-bucket selection | Yes | No | No | No |
| Bucket IDs accessed | Yes | Yes | Yes | No (TLS) |
| Bucket → query region mapping | Yes | No* | No* | No |
| Vector contents (plaintext) | Yes | No | No | No |
| Vector contents (AES encrypted) | Yes | Yes | Yes | No |
| Final search results | Yes | No | No | No |
| Timing information | Yes | Yes | Partial | No |

*With per-enterprise LSH, the server cannot map bucket IDs to query regions without the secret LSH seed.

### Detailed Breakdown

#### What the Compute Server Sees

```
Compute Server's View:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  SEES:                                                          │
│  ├─ HE(query) - encrypted blob, cannot decrypt                 │
│  ├─ Centroids (plaintext) - needed for HE computation          │
│  ├─ HE(scores) - computed blindly, cannot decrypt              │
│  └─ Bucket fetch requests - IDs like "07", "23"                │
│                                                                 │
│  CANNOT DETERMINE:                                              │
│  ├─ What the query vector is                                   │
│  ├─ Which centroids are most similar                           │
│  ├─ Which super-buckets client selected                        │
│  ├─ What bucket "07" represents (no LSH seed)                  │
│  └─ Final search results                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### What the Storage Server Sees

```
Storage Server's View:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  SEES:                                                          │
│  ├─ Encrypted blobs (AES-256-GCM ciphertext)                   │
│  ├─ Bucket IDs each blob belongs to                            │
│  ├─ Which buckets were fetched                                 │
│  └─ Access timestamps                                           │
│                                                                 │
│  CANNOT DETERMINE:                                              │
│  ├─ Vector contents (no AES key)                               │
│  ├─ What bucket "07" means (no LSH seed)                       │
│  ├─ Query vector (never sent to storage)                       │
│  ├─ Which fetched buckets were real vs decoy                   │
│  └─ Search results                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why HE for Centroid Scoring

### The Core Question

> "If we can compute locally, why use expensive HE?"

### The Answer: HE Enables Server-Side Computation on Encrypted Data

| Scenario | Without HE | With HE |
|----------|------------|---------|
| **64 centroids** | Download centroids, compute locally (~0.1ms) | Server computes on HE(query) (~640ms) |
| **10,000 centroids** | Download 10K vectors (~10MB), compute locally | Server computes, returns 10K encrypted scores |
| **Confidential centroids** | Can't download (server won't share) | Server keeps centroids, computes blindly |
| **Audit trail** | No proof of computation | Server proves it computed on all centroids |

### When HE is Essential

1. **Scalability**: As centroids grow beyond ~1000, download becomes impractical
2. **Confidentiality**: Centroids themselves may be proprietary (server's IP)
3. **Compliance**: Regulations may require server-side processing with encryption
4. **Auditability**: Verifiable computation (server can't lie about computing all centroids)

### Current Architecture Decision

**Keep HE for centroid scoring** because:
- Future-proofs for larger centroid sets
- Centroids can remain on server (better security)
- Enables audit/verification if needed
- ~640ms is acceptable for most use cases

**If latency is critical** (sub-100ms required):
- Switch to local centroid scoring
- Accept tradeoff: centroids must be distributed to clients

---

## Security Analysis

### Threat Model

| Threat | Protection | Residual Risk |
|--------|------------|---------------|
| **Honest-but-curious server** | HE, AES, per-enterprise LSH | Server learns bucket access patterns |
| **Malicious server** | Not protected | Server could return wrong results (need verifiable computation) |
| **Compromised credentials** | Token rotation, HSM | If AES/LSH keys leak, re-encryption needed |
| **Network eavesdropper** | TLS + HE/AES | None (encrypted end-to-end) |
| **Database breach** | AES-256-GCM | Attacker gets ciphertext only |

### Cryptographic Assumptions

| Component | Assumption | Security Level |
|-----------|------------|----------------|
| **BFV (HE)** | Ring-LWE hardness | 128-bit, quantum-resistant |
| **AES-256-GCM** | AES security | 256-bit |
| **LSH** | Randomness of hyperplanes | Information-theoretic (if seed is secret) |

### Attack Vectors and Mitigations

#### Attack 1: Bucket Frequency Analysis

**Attack**: Over many queries, observe which buckets are accessed most. Correlate with known popular queries.

**Mitigation**:
- Decoy buckets add noise
- Per-enterprise LSH means attacker can't know what buckets represent
- Session key rotation prevents cross-session correlation

#### Attack 2: Timing Side-Channels

**Attack**: Measure how long AES decryption takes for different buckets. Infer bucket sizes.

**Mitigation**:
- Constant-time decryption (AES-GCM is constant-time)
- Optional: Add random delays to obfuscate

#### Attack 3: Known-Plaintext (if attacker controls some vectors)

**Attack**: Attacker inserts known vectors, observes their bucket assignments, infers LSH mapping.

**Mitigation**:
- Per-enterprise isolation (attacker in Enterprise A can't learn about B)
- Rate limiting on insertions
- Audit logging of all inserts

---

## Performance Characteristics

### Latency Breakdown (100K vectors, 128D, 64 super-buckets)

| Phase | Time | Notes |
|-------|------|-------|
| **Level 1: HE Scoring** | | |
| HE encrypt query | ~5ms | One-time per query |
| HE dot products (64) | ~640ms | 4 parallel workers |
| HE decrypt scores | ~3ms | 64 scalar decryptions |
| **Level 2: Bucket Fetch** | | |
| Bucket selection | ~2ms | Local computation |
| Fetch blobs | ~5-50ms | Network-dependent |
| **Level 3: Local Scoring** | | |
| AES decrypt | ~10-20ms | ~1500 vectors |
| Cosine similarity | ~5ms | Local computation |
| **Total** | **~670-720ms** | |

### Scalability

| Vectors | Super-buckets | Vectors/bucket | Query Time |
|---------|---------------|----------------|------------|
| 10K | 32 | ~312 | ~400ms |
| 100K | 64 | ~1562 | ~700ms |
| 1M | 128 | ~7812 | ~900ms |
| 10M | 256 | ~39062 | ~1200ms |

Note: HE time scales with super-buckets, not total vectors.

---

## Configuration Options

### Privacy vs Performance Presets

```go
// High Privacy: Fewer buckets, more decoys
HighPrivacyConfig{
    NumSuperBuckets:    32,      // Larger buckets (~3125 vectors each)
    TopSuperBuckets:    6,       // Select fewer
    NumDecoys:          12,      // More decoys
}

// Balanced (Default)
DefaultConfig{
    NumSuperBuckets:    64,      // ~1562 vectors each
    TopSuperBuckets:    8,
    NumDecoys:          8,
}

// High Recall: More granular buckets
HighRecallConfig{
    NumSuperBuckets:    64,
    NumSubBuckets:      4,       // Sub-bucket subdivision
    TopSuperBuckets:    12,
    SubBucketsPerSuper: 3,
    NumDecoys:          8,
}
```

### Tuning Guidelines

| Goal | Adjust | Direction |
|------|--------|-----------|
| More privacy | NumDecoys | Increase |
| More privacy | NumSuperBuckets | Decrease (larger buckets) |
| Better recall | TopSuperBuckets | Increase |
| Better recall | SubBucketsPerSuper | Increase |
| Lower latency | NumSuperBuckets | Decrease (fewer HE ops) |
| Lower latency | TopSuperBuckets | Decrease (fewer buckets to fetch) |

---

## Future Enhancements

### Phase 2: Optional PIR

- Add IT-PIR support for 2-server deployments
- Add CPIR for single-server high-security mode
- Runtime selection based on query sensitivity

### Phase 3: Verifiable Computation

- Server proves it computed on all centroids
- Prevents malicious server from returning selective results
- Uses SNARKs or commitments

### Phase 4: Advanced Key Management

- HSM integration for key storage
- Multi-party key generation
- Threshold decryption (require M of N parties)

### Phase 5: Additional Privacy Features

- Differential privacy on bucket access patterns
- Onion routing for query submission
- Decentralized key distribution

---

## References

1. **BFV Scheme**: Fan, J., & Vercauteren, F. (2012). Somewhat Practical Fully Homomorphic Encryption.
2. **Lattigo**: https://github.com/tuneinsight/lattigo
3. **PIR Survey**: Chor, B., et al. (1998). Private Information Retrieval.
4. **LSH**: Indyk, P., & Motwani, R. (1998). Approximate Nearest Neighbors.
5. **AES-GCM**: NIST SP 800-38D.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2026-02-06 | Complete redesign: per-enterprise LSH, PIR analysis, auth model |
| 1.0 | 2026-02-05 | Initial hierarchical architecture |
