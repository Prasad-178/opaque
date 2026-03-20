# SDK Plan

Turn Opaque from an internal project into a proper Go SDK that anyone can install and use to build privacy-preserving vector search applications.

## Goal

```bash
go get github.com/opaque-db/opaque
```

```go
import "github.com/opaque-db/opaque"

db, _ := opaque.NewDB(opaque.Config{Dimension: 128})
db.Add(ctx, "doc-1", embedding)
db.Build(ctx)
db.Save("my-index/")

// Later...
db, _ = opaque.Load("my-index/")
results, _ := db.Search(ctx, query, 10)
```

Someone clones the repo, reads the README, and has a working encrypted vector DB in 10 lines of code.

---

## What's Missing (vs. what a usable SDK needs)

### 1. Persistence — Save/Load a Built Index
**Priority: Critical**

Right now, once you build an index it lives in memory. If the process dies, everything is gone. An SDK must support:

- `db.Save(path string) error` — serialize the full built state (centroids, AES key, config, blob store metadata) to a directory
- `opaque.Load(path string) (*DB, error)` — reconstruct a DB from saved state, ready for Search immediately
- Format: directory with structured files (config.json + centroids.bin + blob store)
- The AES key must be saved securely (or the user provides it)

### 2. Delete & Update Vectors
**Priority: High**

- `db.Delete(ctx, id string) error` — remove a vector by ID
- `db.Update(ctx, id string, vector []float64) error` — replace a vector's data
- Soft delete initially (mark as deleted, filter during search), with compaction via Rebuild
- Update = Delete + Add under the hood

### 3. Metadata & Filtered Search
**Priority: High**

Users want to attach metadata to vectors and filter at search time:

```go
db.Add(ctx, "doc-1", vector, opaque.Metadata{"category": "tech", "year": 2024})

results, _ := db.Search(ctx, query, 10, opaque.Filter{
    Where: map[string]any{"category": "tech"},
})
```

- Metadata stored alongside encrypted blobs (encrypted too, for privacy)
- Filtering happens client-side after AES decryption (preserves privacy model)
- Start simple: exact-match filters on string/int fields

### 4. Vector Count & Info APIs
**Priority: Medium**

```go
db.Size()                    // total vectors (exists already)
db.Count()                   // indexed vectors only (excluding pending)
db.Has(ctx, "doc-1")         // check if ID exists
db.Get(ctx, "doc-1")         // retrieve vector by ID
db.List(ctx, offset, limit)  // paginated listing of IDs
db.Config()                  // return current config
db.Stats()                   // cluster stats + storage stats + vector count
```

### 5. Proper Go Module Path
**Priority: Critical**

Current path is `github.com/opaque/opaque/go` — this isn't a real importable path. Need to:

- Pick a real GitHub org/repo (e.g., `github.com/opaque-db/opaque`)
- Or use a vanity import path
- Update `go.mod` and all internal imports
- This should be done last (after everything else works)

### 6. Examples Directory
**Priority: High**

Working, runnable examples that show real use cases:

```
examples/
├── basic/              # NewDB → Add → Build → Search (10 lines)
├── persistence/        # Save → Load → Search
├── metadata-filter/    # Add with metadata, filtered search
├── large-scale/        # 100K vectors, tuning NumClusters
├── file-storage/       # Using file-backed blob store
└── embedding-model/    # Integration with a real embedding API
```

Each example: `main.go` + short README, runnable with `go run .`

### 7. Better Error Types
**Priority: Medium**

Replace string errors with typed errors for programmatic handling:

```go
var (
    ErrNotBuilt       = errors.New("opaque: index not built")
    ErrAlreadyBuilt   = errors.New("opaque: index already built")
    ErrDimensionMismatch = errors.New("opaque: dimension mismatch")
    ErrNotFound       = errors.New("opaque: vector not found")
    ErrEmptyID        = errors.New("opaque: empty vector ID")
)
```

Users can then do `if errors.Is(err, opaque.ErrNotBuilt) { ... }`

### 8. Callback Hooks / Events
**Priority: Low**

Let users observe the pipeline:

```go
db, _ := opaque.NewDB(opaque.Config{
    Dimension: 128,
    OnBuildProgress: func(phase string, pct float64) {
        fmt.Printf("[%s] %.0f%%\n", phase, pct*100)
    },
})
```

Phases: `"clustering"`, `"encrypting"`, `"indexing"`

---

## Implementation Order

| Phase | What | Why First |
|-------|------|-----------|
| **Phase 1** | Typed errors + `Get`/`Has`/`List`/`Count` APIs | Small, foundational, unblocks everything else |
| **Phase 2** | Save/Load persistence | Most critical missing feature for any real use |
| **Phase 3** | Delete & Update | Core CRUD completeness |
| **Phase 4** | Metadata & filtered search | The feature users will ask for most |
| **Phase 5** | Examples directory | Makes it actually adoptable |
| **Phase 6** | Module path + publish | Ship it |

---

## Non-Goals (for now)

- **Client/server mode** — already exists via gRPC, not the SDK focus
- **Multi-tenancy** — enterprise isolation already works internally
- **Python/JS bindings** — Go SDK first, bindings later
- **Embedding generation** — SDK takes vectors in, embedding is the user's job
- **Distributed/sharded** — single-node SDK first

---

## Design Principles

1. **Simple by default, tunable when needed** — `NewDB(Config{Dimension: 128})` should just work with good defaults
2. **Privacy is not optional** — every vector is always encrypted, every search always uses HE + decoys
3. **No footguns** — lifecycle states prevent misuse (can't Search before Build, etc.)
4. **Standard Go patterns** — `io.Closer`, `context.Context`, `error` wrapping, `json.Marshaler`
5. **Zero external services** — the SDK is a library, not a platform. No Docker, no Redis, no cloud deps.
