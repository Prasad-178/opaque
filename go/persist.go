package opaque

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
	"github.com/opaque/opaque/go/pkg/pca"
)

const (
	metadataFile    = "metadata.json"
	enterpriseFile  = "enterprise.gob"
	pcaFile         = "pca.gob"
	blobsDir        = "blobs"
	persistVersion  = 1
)

// saveMetadata is the JSON-serializable metadata for a saved DB.
type saveMetadata struct {
	Version      int          `json:"version"`
	SavedAt      time.Time    `json:"saved_at"`
	Config       Config       `json:"config"`
	ClusterStats ClusterStats `json:"cluster_stats"`
	HasPCA       bool         `json:"has_pca"`
}

// Save persists a built DB to the given directory path.
//
// The directory must not already contain a saved DB (no metadata.json).
// After Save, the DB can be restored with [Load] in a new process.
//
// Save is safe for concurrent use with [DB.Search] — it acquires a read lock.
func (db *DB) Save(path string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.state != stateReady {
		return ErrNotReady
	}

	// Create directory.
	if err := os.MkdirAll(path, 0o700); err != nil {
		return fmt.Errorf("opaque: failed to create save directory: %w", err)
	}

	// Check for existing save.
	metaPath := filepath.Join(path, metadataFile)
	if _, err := os.Stat(metaPath); err == nil {
		return fmt.Errorf("opaque: save directory already contains a database; remove it first or use a different path")
	}

	// Save enterprise config.
	ecfgPath := filepath.Join(path, enterpriseFile)
	if err := db.enterpriseCfg.SaveToFile(ecfgPath); err != nil {
		os.RemoveAll(path)
		return fmt.Errorf("opaque: failed to save enterprise config: %w", err)
	}

	// Save PCA model if present.
	hasPCA := db.pcaModel != nil
	if hasPCA {
		pcaPath := filepath.Join(path, pcaFile)
		f, err := os.Create(pcaPath)
		if err != nil {
			os.RemoveAll(path)
			return fmt.Errorf("opaque: failed to create PCA file: %w", err)
		}
		if err := db.pcaModel.Save(f); err != nil {
			f.Close()
			os.RemoveAll(path)
			return fmt.Errorf("opaque: failed to save PCA model: %w", err)
		}
		f.Close()
	}

	// Export blobs to a FileStore at <path>/blobs/.
	blobsPath := filepath.Join(path, blobsDir)
	destStore, err := blob.NewFileStore(blobsPath)
	if err != nil {
		os.RemoveAll(path)
		return fmt.Errorf("opaque: failed to create blob store: %w", err)
	}

	ctx := context.Background()
	buckets, err := db.blobStore.ListBuckets(ctx)
	if err != nil {
		destStore.Close()
		os.RemoveAll(path)
		return fmt.Errorf("opaque: failed to list buckets: %w", err)
	}

	for _, bucket := range buckets {
		blobs, err := db.blobStore.GetBucket(ctx, bucket)
		if err != nil {
			destStore.Close()
			os.RemoveAll(path)
			return fmt.Errorf("opaque: failed to get bucket %q: %w", bucket, err)
		}
		if len(blobs) > 0 {
			if err := destStore.PutBatch(ctx, blobs); err != nil {
				destStore.Close()
				os.RemoveAll(path)
				return fmt.Errorf("opaque: failed to write bucket %q: %w", bucket, err)
			}
		}
	}
	destStore.Close()

	// Write metadata last (acts as commit marker).
	meta := saveMetadata{
		Version:      persistVersion,
		SavedAt:      time.Now(),
		Config:       db.cfg,
		ClusterStats: ClusterStats{
			NumClusters:   db.clusterStats.NumClusters,
			MinSize:       db.clusterStats.MinSize,
			MaxSize:       db.clusterStats.MaxSize,
			AvgSize:       db.clusterStats.AvgSize,
			EmptyClusters: db.clusterStats.EmptyClusters,
			Iterations:    db.clusterStats.Iterations,
		},
		HasPCA:       hasPCA,
	}
	metaData, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		os.RemoveAll(path)
		return fmt.Errorf("opaque: failed to marshal metadata: %w", err)
	}
	if err := os.WriteFile(metaPath, metaData, 0o600); err != nil {
		os.RemoveAll(path)
		return fmt.Errorf("opaque: failed to write metadata: %w", err)
	}

	return nil
}

// Load restores a DB from a directory previously created by [DB.Save].
//
// The returned DB is immediately ready for [DB.Search] — no Build is needed.
// The blob store is opened in file mode from the saved directory.
//
// To add new vectors after loading, use [DB.Add] followed by [DB.Rebuild].
func Load(path string) (*DB, error) {
	// Read and parse metadata.
	metaPath := filepath.Join(path, metadataFile)
	metaData, err := os.ReadFile(metaPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("opaque: no saved database found at %q", path)
		}
		return nil, fmt.Errorf("opaque: failed to read metadata: %w", err)
	}

	var meta saveMetadata
	if err := json.Unmarshal(metaData, &meta); err != nil {
		return nil, fmt.Errorf("opaque: failed to parse metadata: %w", err)
	}
	if meta.Version != persistVersion {
		return nil, fmt.Errorf("opaque: unsupported save version %d (expected %d)", meta.Version, persistVersion)
	}

	// Load enterprise config.
	ecfgPath := filepath.Join(path, enterpriseFile)
	enterpriseCfg, err := enterprise.LoadConfigFromFile(ecfgPath)
	if err != nil {
		return nil, fmt.Errorf("opaque: failed to load enterprise config: %w", err)
	}

	// Open blob store.
	blobsPath := filepath.Join(path, blobsDir)
	store, err := blob.NewFileStore(blobsPath)
	if err != nil {
		return nil, fmt.Errorf("opaque: failed to open blob store: %w", err)
	}

	// Load PCA model if saved.
	var pcaModel *pca.PCA
	if meta.HasPCA {
		pcaPath := filepath.Join(path, pcaFile)
		f, err := os.Open(pcaPath)
		if err != nil {
			store.Close()
			return nil, fmt.Errorf("opaque: failed to open PCA file: %w", err)
		}
		pcaModel, err = pca.LoadPCA(f)
		f.Close()
		if err != nil {
			store.Close()
			return nil, fmt.Errorf("opaque: failed to load PCA model: %w", err)
		}
	}

	// Reconstruct config with defaults applied.
	cfg := meta.Config
	applyDefaults(&cfg)
	// The loaded DB always uses file storage from the save directory.
	cfg.Storage = File
	cfg.StoragePath = blobsPath

	// Build credentials and search config.
	creds := makeCredentials(enterpriseCfg)

	effectiveDim := cfg.Dimension
	if cfg.PCADimension > 0 {
		effectiveDim = cfg.PCADimension
	}

	db := &DB{
		cfg:           cfg,
		state:         stateReady,
		enterpriseCfg: enterpriseCfg,
		blobStore:     store,
		pcaModel:      pcaModel,
		clusterStats: hierarchical.ClusterStats{
			NumClusters:   meta.ClusterStats.NumClusters,
			MinSize:       meta.ClusterStats.MinSize,
			MaxSize:       meta.ClusterStats.MaxSize,
			AvgSize:       meta.ClusterStats.AvgSize,
			EmptyClusters: meta.ClusterStats.EmptyClusters,
			Iterations:    meta.ClusterStats.Iterations,
		},
	}

	searchCfg := db.makeSearchConfig(enterpriseCfg, effectiveDim)

	searchClient, err := client.NewEnterpriseHierarchicalClientWithPoolSize(
		searchCfg, creds, store, cfg.WorkerPoolSize,
	)
	if err != nil {
		store.Close()
		return nil, fmt.Errorf("opaque: failed to create search client: %w", err)
	}
	db.searchClient = searchClient

	return db, nil
}
