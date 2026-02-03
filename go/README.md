# Opaque Go

Privacy-preserving vector search using homomorphic encryption, implemented in Go.

## Overview

Opaque Go allows clients to search encrypted queries against a vector database without revealing query content to the server. It uses:

- **Lattigo BFV** for homomorphic encryption (128-bit security)
- **Locality-Sensitive Hashing (LSH)** for fast candidate retrieval
- **gRPC** for efficient client-server communication

## Project Structure

```
go/
├── cmd/
│   ├── search-service/  # gRPC server
│   └── cli/             # CLI testing tool
├── pkg/
│   ├── crypto/          # Homomorphic encryption (Lattigo BFV)
│   ├── lsh/             # Locality-sensitive hashing
│   └── client/          # Client SDK
├── internal/
│   ├── service/         # Search service implementation
│   ├── session/         # Session management
│   └── store/           # Vector storage (in-memory, Milvus)
├── api/
│   └── proto/           # Protocol buffer definitions
└── examples/
    └── search/          # Example usage
```

## Quick Start

### Build

```bash
go build ./...
```

### Run Tests

```bash
go test ./... -v
```

### Run Example

```bash
go run ./examples/search/main.go
```

### Run CLI

```bash
go run ./cmd/cli/main.go -bench
```

### Run Server

```bash
go run ./cmd/search-service/main.go -demo-vectors=1000
```

Health endpoints:
- `http://localhost:8080/healthz` - Health check
- `http://localhost:8080/readyz` - Readiness check

## Performance

### Go vs Python Comparison (128-dim vectors)

| Operation | Go (Lattigo) | Python (LightPHE) | Speedup |
|-----------|--------------|-------------------|---------|
| Encryption | 5.2 ms | 1,912 ms | **367x** |
| Decryption | 0.8 ms | 140 ms | **175x** |
| Dot Product | 33 ms | 91 ms | **2.8x** |
| LSH Hash | 0.006 ms | ~1 ms | **166x** |

### Parallel Execution (Apple M4 Pro, 12 cores)

| Operation | Sequential | Parallel | Speedup |
|-----------|------------|----------|---------|
| 20× Dot Products | 665 ms | 128 ms | **5.2x** |
| 20× Decryptions | 18 ms | 5.4 ms | **3.3x** |

### End-to-End Latency

| Scenario | Go Parallel | Python | Improvement |
|----------|-------------|--------|-------------|
| Full Search (20 results) | **188 ms** | 6,592 ms | **35x faster** |

See [BENCHMARKS.md](BENCHMARKS.md) for detailed benchmark results and methodology.

## Architecture

### Search Flow

1. **Client** computes LSH hash of query vector (locally)
2. **Server** returns candidate IDs by LSH similarity
3. **Client** encrypts query vector with homomorphic encryption
4. **Server** computes encrypted similarity scores
5. **Client** decrypts scores and returns top-k results

### Privacy Guarantees

- Server **never** sees actual query vectors
- Server **never** sees similarity scores
- LSH hash reveals only approximate bucket (acceptable leakage)
- Client **never** downloads full database

## Configuration

### Client Config

```go
client.Config{
    Dimension:     128,    // Vector dimension
    LSHBits:       64,     // LSH hash bits
    MaxCandidates: 100,    // Max LSH candidates
    DecryptTopN:   20,     // Scores to decrypt
}
```

### Server Config

```go
service.Config{
    LSHNumBits:          128,
    LSHDimension:        128,
    LSHSeed:             42,
    MaxSessionTTL:       24 * time.Hour,
    MaxConcurrentScores: 16,
}
```

## Dependencies

- [Lattigo v5](https://github.com/tuneinsight/lattigo) - Homomorphic encryption
- [gRPC-Go](https://github.com/grpc/grpc-go) - RPC framework

## TODO

- [ ] Generate protobuf code
- [ ] Implement full gRPC service
- [ ] Add Milvus vector store integration
- [ ] Add Redis session store
- [ ] Add observability (metrics, tracing)
- [ ] Implement proper evaluation key exchange for rotations
- [ ] Add end-to-end integration tests

## License

MIT
