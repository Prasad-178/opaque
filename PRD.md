# PRD: Project Opaque (MVP)

**Version:** 0.1 (Draft)
**Type:** Engineering & Research
**Core Philosophy:** "Speed via Approximation, Privacy via Math."

---

## 1. Executive Summary

**Project Opaque** is a privacy-preserving vector search engine. It allows a client to query a remote vector database such that the **server never sees the raw query vector** and the **client never downloads the full database**.

We solve the "FHE Latency Bottleneck" by using a two-stage funnel:

1. **Coarse Stage:** Fast, approximate filtering using Locality Sensitive Hashing (LSH) on obfuscated data.
2. **Fine Stage:** Cryptographically secure scoring using Partial Homomorphic Encryption (PHE) on the top candidates.

---

## 2. Problem Statement

- **The Problem:** Current vector search (Pinecone, Milvus) requires the server to see the query vector (Q) in plaintext to compute similarity. This prevents the usage of RAG (Retrieval Augmented Generation) in highly sensitive fields (Healthcare, Legal, Defense).

- **The Opportunity:** True FHE is too slow (hours). A hybrid PHE approach can achieve "good enough" privacy with sub-second latency.

---

## 3. Scope & Goals

### 3.1. In-Scope (The MVP)

- **Client-Side:** Embedding generation, LSH hashing, Paillier Key Generation, Encryption/Decryption.
- **Server-Side:** In-memory storage of vectors, LSH Indexing, PHE Scalar Multiplication.
- **Protocol:** A custom request/response flow handling the "Two-Step" search.
- **Scale:** Support for 100,000 vectors with < 3 second retrieval latency.

### 3.2. Out-of-Scope (For now)

- Fully Homomorphic Encryption (TFHE/Zama) – *Relegated to "Future Research"*.
- Disk-based storage (Everything in RAM for speed).
- CRUD operations (Read-only database for the MVP).

---

## 4. System Architecture

The system relies on a **"Funnel Strategy"** to reduce the cryptographic workload.

### The Data Structures

1. **The Public/Obfuscated Index (LSH):**
   - Used for: Rapid candidate generation.
   - Data: Vectors are **Rotated** or **Noised** (Obfuscated) so they aren't raw plaintext, but preserve locality.
   - Technique: `Faiss` or custom LSH implementation.

2. **The Secure Store:**
   - Used for: Exact scoring.
   - Data: Plaintext vectors (stored securely on server) OR Encrypted vectors (depending on threat model).
   - *Decision for MVP:* **Private Query Model.** Server has Plaintext Data. User has Private Query.

### The Request Lifecycle

1. **Setup:**
   - Client generates Paillier Public/Private Keys.
   - Client sends Public Key to Server.

2. **Stage 1: The Bucket Lookup (Coarse)**
   - Client generates Query Vector Q.
   - Client applies **Obfuscation Function** -> Q'.
   - Client sends Q' to Server.
   - Server queries LSH Index with Q'.
   - Server retrieves **Top 1,000 Candidate IDs**.

3. **Stage 2: The Blind Scoring (Fine)**
   - Client encrypts Q using Paillier -> E(Q).
   - Client sends E(Q) to Server.
   - Server pulls the actual vectors for the 1,000 Candidates.
   - Server computes Dot Product in PHE: E(Score_i) = Σ(E(Q_j) × DatabaseVector_ij).
   - Result: Server now has 1,000 Encrypted Scores. It does not know which is highest.

4. **Stage 3: The Reveal**
   - Server sends all 1,000 Encrypted Scores back to Client.
   - Client decrypts them.
   - Client performs `ArgMax` (sorts them) to find the top result.

---

## 5. Technical Specification

### 5.1. Tech Stack

- **Language:** Python 3.10+ (Prototype), Rust (Future Production).
- **Cryptography:** `LightPHE` (Library).
- **Algorithm:** Paillier (Additive Homomorphism).
- **Indexing:** `NumPy` (initially) or `Faiss` (later) for LSH.
- **API:** `FastAPI` (for Client-Server communication).

### 5.2. Key Algorithms

**The PHE Dot Product (Client-Side Encrypted Query):** If Q is encrypted and V is plaintext:

```
E(Q · V) = Π(E(q_i)^v_i)
```

*(Note: In Paillier, "multiplication by scalar" is actually exponentiation of the ciphertext.)*

### 5.3. Latency Budget (Target: < 3s)

| Step | Operation | Estimated Time |
|------|-----------|----------------|
| 1 | LSH Lookup (100k DB) | 50ms |
| 2 | Encrypt Query (Client) | 100ms |
| 3 | Network (Upload E(Q)) | 50ms |
| 4 | **PHE Dot Product (1k vecs)** | **2000ms** (The bottleneck) |
| 5 | Network (Download Scores) | 50ms |
| 6 | Decrypt & Sort | 50ms |

---

## 6. Implementation Roadmap

### Phase 1: The "Hello World"

**Goal:** Verify `LightPHE` performance.

**Tasks:**
- Write a script to encrypt a vector of size 1536 (OpenAI embedding size).
- Perform 1,000 dot products against random plaintext vectors.
- **Stop/Go Decision:** If this takes >10 seconds, we must reduce embedding dimensionality (e.g., use PCA to reduce 1536 -> 256).

### Phase 2: The "Blind Search"

**Goal:** End-to-end flow without LSH.

**Tasks:**
- Build a mock database of 100 vectors.
- Implement Client script (Encrypt -> Send).
- Implement Server script (Compute -> Return).
- Verify accuracy (Does the encrypted search return the same result as a plaintext search?).

### Phase 3: The "Funnel"

**Goal:** Add LSH to support 100k vectors.

**Tasks:**
- Implement LSH (Random Projection).
- Integrate the "Coarse -> Fine" workflow.

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Ciphertext Blowup** | High | Paillier ciphertexts are huge (2048 bits). Sending 1,000 scores back = 2MB-4MB of data. **Mitigation:** Strict limit on "Candidate Set" size (e.g., max 500). |
| **Dimensionality** | High | 1536 dimensions (OpenAI) is too slow for PHE. **Mitigation:** Use PCA to compress vectors to 128 or 256 dims before encryption. |
| **Accuracy Loss** | Medium | LSH is approximate. We might miss the true "best" match in the Coarse stage. **Mitigation:** Tune LSH recall parameters (overlap buckets). |

---

## 8. Success Metrics

1. **Accuracy:** Recall@10 must be > 90% (compared to standard plaintext search).
2. **Privacy:** Zero leakage of the Query Vector to the server (mathematically guaranteed by Paillier).
3. **Speed:** P95 Latency under 3 seconds for a 10k vector dataset.

---

## 9. Technical Deep Dive: LightPHE Usage

### Key Generation & Basic Operations

```python
from lightphe import LightPHE

# Initialize Paillier cryptosystem
cs = LightPHE(algorithm_name="Paillier", key_size=1024, precision=5)

# Export keys for client-server separation
cs.export_keys("private_keys.json", public=False)
cs.export_keys("public_keys.json", public=True)

# Encrypt a vector
query_vector = [0.1, 0.2, 0.3, ...]  # 128-dim
encrypted_query = cs.encrypt(query_vector)

# Server performs dot product (encrypted × plaintext)
db_vector = [0.5, 0.6, 0.7, ...]  # 128-dim plaintext
encrypted_score = encrypted_query @ db_vector  # Returns encrypted result

# Client decrypts
score = cs.decrypt(encrypted_score)[0]
```

### Vector Operations in LightPHE

- **Encrypted Vector + Encrypted Vector:** Supported (additive homomorphism)
- **Encrypted Vector × Plaintext Scalar:** Supported
- **Encrypted Vector × Plaintext Vector (element-wise):** Supported
- **Encrypted Vector @ Plaintext Vector (dot product):** Supported
- **Encrypted × Encrypted Multiplication:** NOT supported (Paillier limitation)

---

## 10. Technical Deep Dive: LSH Implementation

### Using Faiss for LSH

```python
import faiss
import numpy as np

# Configuration
d = 128  # dimensionality (after PCA)
nbits = 256  # hash bits

# Create LSH index
index = faiss.IndexLSH(d, nbits)

# Add vectors
vectors = np.random.randn(100000, d).astype('float32')
index.add(vectors)

# Query for top-k candidates
query = np.random.randn(1, d).astype('float32')
distances, indices = index.search(query, k=1000)
```

### Custom Random Projection LSH

```python
import numpy as np

class RandomProjectionLSH:
    def __init__(self, dim, nbits):
        self.dim = dim
        self.nbits = nbits
        self.planes = np.random.randn(nbits, dim)
        self.planes /= np.linalg.norm(self.planes, axis=1, keepdims=True)

    def hash(self, vectors):
        """Convert vectors to binary hashes"""
        projections = vectors @ self.planes.T
        return (projections > 0).astype(np.uint8)

    def build_index(self, vectors):
        """Build hash buckets"""
        self.hashes = self.hash(vectors)
        self.buckets = {}
        for idx, h in enumerate(self.hashes):
            key = tuple(h)
            if key not in self.buckets:
                self.buckets[key] = []
            self.buckets[key].append(idx)

    def query(self, vector, k=1000):
        """Find candidates with similar hashes"""
        query_hash = self.hash(vector.reshape(1, -1))[0]
        # Return exact matches + hamming-distance neighbors
        candidates = set()
        candidates.update(self.buckets.get(tuple(query_hash), []))
        return list(candidates)[:k]
```

---

## 11. Project Structure (Proposed)

```
opaque/
├── PRD.md
├── pyproject.toml
├── src/
│   └── opaque/
│       ├── __init__.py
│       ├── client/
│       │   ├── __init__.py
│       │   ├── crypto.py      # Paillier key gen, encrypt, decrypt
│       │   ├── embedding.py   # Embedding generation (optional)
│       │   └── search.py      # Client-side search orchestration
│       ├── server/
│       │   ├── __init__.py
│       │   ├── index.py       # LSH index management
│       │   ├── compute.py     # PHE dot product computation
│       │   └── api.py         # FastAPI server
│       └── shared/
│           ├── __init__.py
│           ├── protocol.py    # Request/Response models
│           └── utils.py       # Common utilities
├── scripts/
│   ├── benchmark_phe.py       # Phase 1: Test LightPHE performance
│   ├── demo_blind_search.py   # Phase 2: End-to-end without LSH
│   └── demo_full_pipeline.py  # Phase 3: Full coarse->fine workflow
└── tests/
    ├── test_crypto.py
    ├── test_lsh.py
    └── test_integration.py
```
