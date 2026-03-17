// Command search-service runs the Opaque SDK-backed search service.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Prasad-178/opaque"
)

var (
	httpPort            = flag.Int("http-port", envInt("OPAQUE_HTTP_PORT", 8080), "HTTP API/health port")
	dimension           = flag.Int("dimension", envInt("OPAQUE_DIMENSION", 128), "Vector dimension")
	numClusters         = flag.Int("num-clusters", envInt("OPAQUE_NUM_CLUSTERS", 64), "Number of k-means clusters")
	topClusters         = flag.Int("top-clusters", envInt("OPAQUE_TOP_CLUSTERS", 0), "Top clusters probed during search (0=default)")
	numDecoys           = flag.Int("num-decoys", envInt("OPAQUE_NUM_DECOYS", 8), "Decoy clusters per search")
	workerPoolSize      = flag.Int("worker-pool-size", envInt("OPAQUE_WORKER_POOL_SIZE", 0), "HE worker pool size (0=auto)")
	dbPath              = flag.String("db-path", envString("OPAQUE_DB_PATH", "./data/db"), "Opaque DB snapshot directory")
	bootstrapVectors    = flag.String("bootstrap-vectors", envString("OPAQUE_BOOTSTRAP_VECTORS", ""), "Optional JSON file containing vectors to preload")
	autoIndexEnabled    = flag.Bool("auto-index-enabled", envBool("OPAQUE_AUTO_INDEX_ENABLED", true), "Enable autonomous background Build/Rebuild")
	autoIndexInterval   = flag.Duration("auto-index-interval", envDuration("OPAQUE_AUTO_INDEX_INTERVAL", 5*time.Second), "Auto-index check interval")
	autoIndexMinChanges = flag.Int("auto-index-min-changes", envInt("OPAQUE_AUTO_INDEX_MIN_CHANGES", 1), "Minimum pending mutations before auto-index runs")
	autoIndexTimeout    = flag.Duration("auto-index-timeout", envDuration("OPAQUE_AUTO_INDEX_TIMEOUT", 15*time.Minute), "Timeout per auto Build/Rebuild")
)

type bootstrapVector struct {
	ID       string         `json:"id"`
	Values   []float64      `json:"values"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

type serviceRuntime struct {
	db     *opaque.DB
	dbPath string
}

type addBatchRequest struct {
	Vectors []bootstrapVector `json:"vectors"`
}

type searchRequest struct {
	Vector []float64      `json:"vector"`
	TopK   int            `json:"top_k"`
	Filter *opaque.Filter `json:"filter,omitempty"`
}

type updateRequest struct {
	Values   []float64      `json:"values"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

type saveRequest struct {
	Path string `json:"path"`
}

func main() {
	flag.Parse()

	log.Println("Starting Opaque SDK Search Service...")

	runtime, err := newServiceRuntime()
	if err != nil {
		log.Fatalf("Failed to initialize runtime: %v", err)
	}
	defer runtime.db.Close()

	if err := runtime.seedVectors(); err != nil {
		log.Fatalf("Failed to seed vectors: %v", err)
	}

	httpMux := http.NewServeMux()
	runtime.registerRoutes(httpMux)

	httpServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", *httpPort),
		Handler: httpMux,
	}

	go func() {
		log.Printf("HTTP server listening on :%d", *httpPort)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("HTTP server failed: %v", err)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("Shutting down...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(ctx)
	log.Println("Shutdown complete")
}

func newServiceRuntime() (*serviceRuntime, error) {
	absPath, err := filepath.Abs(*dbPath)
	if err != nil {
		return nil, fmt.Errorf("resolve db path: %w", err)
	}

	db, err := openOrCreateDB(absPath)
	if err != nil {
		return nil, err
	}

	db.ConfigureAutoIndex(
		*autoIndexEnabled,
		*autoIndexInterval,
		*autoIndexMinChanges,
		*autoIndexTimeout,
		func(err error) { log.Printf("auto-index error: %v", err) },
	)

	return &serviceRuntime{db: db, dbPath: absPath}, nil
}

func openOrCreateDB(path string) (*opaque.DB, error) {
	if _, err := os.Stat(filepath.Join(path, "metadata.json")); err == nil {
		db, loadErr := opaque.Load(path)
		if loadErr != nil {
			return nil, fmt.Errorf("load db: %w", loadErr)
		}
		log.Printf("Loaded DB snapshot from %s", path)
		return db, nil
	}

	cfg := opaque.Config{
		Dimension:             *dimension,
		NumClusters:           *numClusters,
		TopClusters:           *topClusters,
		NumDecoys:             *numDecoys,
		WorkerPoolSize:        *workerPoolSize,
		AutoIndexEnabled:      *autoIndexEnabled,
		AutoIndexInterval:     *autoIndexInterval,
		AutoIndexMinChanges:   *autoIndexMinChanges,
		AutoIndexBuildTimeout: *autoIndexTimeout,
		OnAutoIndexError:      func(err error) { log.Printf("auto-index error: %v", err) },
	}

	db, err := opaque.NewDB(cfg)
	if err != nil {
		return nil, fmt.Errorf("new db: %w", err)
	}
	log.Printf("Created new DB (dim=%d, clusters=%d)", *dimension, *numClusters)
	return db, nil
}

func (rt *serviceRuntime) seedVectors() error {
	if *bootstrapVectors == "" {
		return nil
	}

	data, err := os.ReadFile(*bootstrapVectors)
	if err != nil {
		return fmt.Errorf("read bootstrap file: %w", err)
	}

	var records []bootstrapVector
	if err := json.Unmarshal(data, &records); err != nil {
		return fmt.Errorf("parse bootstrap file: %w", err)
	}
	if len(records) == 0 {
		return nil
	}

	ctx := context.Background()
	for i, rec := range records {
		if rec.ID == "" {
			return fmt.Errorf("bootstrap vector %d has empty id", i)
		}
		if len(rec.Values) != *dimension {
			return fmt.Errorf("bootstrap vector %q has dimension %d, expected %d", rec.ID, len(rec.Values), *dimension)
		}
		if rec.Metadata != nil {
			if err := rt.db.AddWithMetadata(ctx, rec.ID, rec.Values, rec.Metadata); err != nil {
				return fmt.Errorf("add bootstrap vector %q: %w", rec.ID, err)
			}
		} else {
			if err := rt.db.Add(ctx, rec.ID, rec.Values); err != nil {
				return fmt.Errorf("add bootstrap vector %q: %w", rec.ID, err)
			}
		}
	}

	log.Printf("Queued %d bootstrap vectors from %s", len(records), *bootstrapVectors)
	return nil
}

func (rt *serviceRuntime) registerRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/healthz", rt.handleHealth)
	mux.HandleFunc("/readyz", rt.handleReady)

	mux.HandleFunc("POST /v1/vectors/batch", rt.handleAddBatch)
	mux.HandleFunc("PUT /v1/vectors/{id}", rt.handleUpdate)
	mux.HandleFunc("DELETE /v1/vectors/{id}", rt.handleDelete)
	mux.HandleFunc("POST /v1/search", rt.handleSearch)
	mux.HandleFunc("GET /v1/stats", rt.handleStats)

	mux.HandleFunc("POST /v1/admin/build", rt.handleBuild)
	mux.HandleFunc("POST /v1/admin/save", rt.handleSave)
}

func (rt *serviceRuntime) handleHealth(w http.ResponseWriter, r *http.Request) {
	stats := rt.db.Stats(r.Context())
	writeJSON(w, http.StatusOK, map[string]any{
		"status":          "healthy",
		"is_ready":        stats.IsReady,
		"indexed_vectors": stats.IndexedVectors,
	})
}

func (rt *serviceRuntime) handleReady(w http.ResponseWriter, r *http.Request) {
	if !rt.db.IsReady() {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"status": "not-ready"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ready"})
}

func (rt *serviceRuntime) handleAddBatch(w http.ResponseWriter, r *http.Request) {
	var req addBatchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if len(req.Vectors) == 0 {
		writeError(w, http.StatusBadRequest, "vectors must not be empty")
		return
	}

	ctx := r.Context()
	for i, v := range req.Vectors {
		if v.Metadata != nil {
			if err := rt.db.AddWithMetadata(ctx, v.ID, v.Values, v.Metadata); err != nil {
				writeError(w, http.StatusBadRequest, fmt.Sprintf("vector %d: %v", i, err))
				return
			}
		} else {
			if err := rt.db.Add(ctx, v.ID, v.Values); err != nil {
				writeError(w, http.StatusBadRequest, fmt.Sprintf("vector %d: %v", i, err))
				return
			}
		}
	}

	writeJSON(w, http.StatusAccepted, map[string]any{
		"queued":       len(req.Vectors),
		"auto_index":   *autoIndexEnabled,
		"next_rebuild": *autoIndexInterval,
	})
}

func (rt *serviceRuntime) handleUpdate(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "missing id")
		return
	}

	var req updateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	ctx := r.Context()
	if req.Metadata == nil {
		if err := rt.db.Update(ctx, id, req.Values); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
	} else {
		if err := rt.db.Delete(ctx, id); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		if err := rt.db.AddWithMetadata(ctx, id, req.Values, req.Metadata); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
	}

	writeJSON(w, http.StatusAccepted, map[string]any{"status": "queued"})
}

func (rt *serviceRuntime) handleDelete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := rt.db.Delete(r.Context(), id); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"status": "queued"})
}

func (rt *serviceRuntime) handleSearch(w http.ResponseWriter, r *http.Request) {
	var req searchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.TopK <= 0 {
		req.TopK = 10
	}

	var (
		results []opaque.Result
		err     error
	)
	if req.Filter != nil {
		results, err = rt.db.SearchWithFilter(r.Context(), req.Vector, req.TopK, *req.Filter)
	} else {
		results, err = rt.db.Search(r.Context(), req.Vector, req.TopK)
	}
	if err != nil {
		if errors.Is(err, opaque.ErrNotBuilt) || errors.Is(err, opaque.ErrNoVectors) {
			writeError(w, http.StatusServiceUnavailable, "index not ready yet")
			return
		}
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, results)
}

func (rt *serviceRuntime) handleStats(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, rt.db.Stats(r.Context()))
}

func (rt *serviceRuntime) handleBuild(w http.ResponseWriter, r *http.Request) {
	if err := rt.db.Build(r.Context()); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "built"})
}

func (rt *serviceRuntime) handleSave(w http.ResponseWriter, r *http.Request) {
	var req saveRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	path := req.Path
	if path == "" {
		path = rt.dbPath
	}
	if err := rt.db.Save(path); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "saved", "path": path})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func envString(key, fallback string) string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	return v
}

func envInt(key string, fallback int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(v)
	if err != nil {
		log.Printf("Invalid %s=%q, using default %d", key, v, fallback)
		return fallback
	}
	return parsed
}

func envBool(key string, fallback bool) bool {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	parsed, err := strconv.ParseBool(v)
	if err != nil {
		log.Printf("Invalid %s=%q, using default %t", key, v, fallback)
		return fallback
	}
	return parsed
}

func envDuration(key string, fallback time.Duration) time.Duration {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(v)
	if err != nil {
		log.Printf("Invalid %s=%q, using default %s", key, v, fallback)
		return fallback
	}
	return parsed
}
