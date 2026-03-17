package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Prasad-178/opaque"
)

func TestEnvHelpers(t *testing.T) {
	const key = "OPAQUE_TEST_VALUE"

	t.Setenv(key, "")
	if got := envString(key, "fallback"); got != "fallback" {
		t.Fatalf("envString empty = %q, want fallback", got)
	}

	t.Setenv(key, "12")
	if got := envInt(key, 7); got != 12 {
		t.Fatalf("envInt = %d, want 12", got)
	}

	t.Setenv(key, "true")
	if got := envBool(key, false); !got {
		t.Fatal("envBool should parse true")
	}

	t.Setenv(key, "bad")
	if got := envInt(key, 7); got != 7 {
		t.Fatalf("envInt fallback = %d, want 7", got)
	}

	t.Setenv(key, "1s")
	if got := envDuration(key, time.Minute); got != time.Second {
		t.Fatalf("envDuration = %s, want 1s", got)
	}
}

func TestOpenOrCreateDB_NewAndLoad(t *testing.T) {
	dir := t.TempDir()

	oldPath := *dbPath
	oldDim := *dimension
	oldClusters := *numClusters
	t.Cleanup(func() {
		*dbPath = oldPath
		*dimension = oldDim
		*numClusters = oldClusters
	})
	*dbPath = dir
	*dimension = 4
	*numClusters = 2

	db, err := openOrCreateDB(dir)
	if err != nil {
		t.Fatalf("openOrCreateDB(new): %v", err)
	}

	if err := db.Add(context.Background(), "v1", []float64{1, 0, 0, 0}); err != nil {
		t.Fatalf("Add v1: %v", err)
	}
	if err := db.Add(context.Background(), "v2", []float64{0, 1, 0, 0}); err != nil {
		t.Fatalf("Add v2: %v", err)
	}
	if err := db.Build(context.Background()); err != nil {
		t.Fatalf("Build: %v", err)
	}
	if err := db.Save(dir); err != nil {
		t.Fatalf("Save: %v", err)
	}
	db.Close()

	loaded, err := openOrCreateDB(dir)
	if err != nil {
		t.Fatalf("openOrCreateDB(load): %v", err)
	}
	defer loaded.Close()

	if !loaded.IsReady() {
		t.Fatal("loaded DB should be ready")
	}
}

func TestHandleAddBatch(t *testing.T) {
	db, err := opaque.NewDB(opaque.Config{Dimension: 4, NumClusters: 2})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	rt := &serviceRuntime{db: db, dbPath: t.TempDir()}

	reqBody := addBatchRequest{Vectors: []bootstrapVector{{ID: "v1", Values: []float64{1, 0, 0, 0}}}}
	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/vectors/batch", bytes.NewReader(body))
	w := httptest.NewRecorder()

	rt.handleAddBatch(w, req)
	if w.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want %d body=%s", w.Code, http.StatusAccepted, w.Body.String())
	}

	if !db.Has(context.Background(), "v1") {
		t.Fatal("vector v1 should be present after add")
	}
}

func TestSeedVectors(t *testing.T) {
	db, err := opaque.NewDB(opaque.Config{Dimension: 4, NumClusters: 2})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	rt := &serviceRuntime{db: db, dbPath: t.TempDir()}

	file := filepath.Join(t.TempDir(), "bootstrap.json")
	data, _ := json.Marshal([]bootstrapVector{{ID: "v1", Values: []float64{1, 0, 0, 0}}})
	if err := os.WriteFile(file, data, 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	oldBootstrap := *bootstrapVectors
	oldDim := *dimension
	t.Cleanup(func() {
		*bootstrapVectors = oldBootstrap
		*dimension = oldDim
	})
	*bootstrapVectors = file
	*dimension = 4

	if err := rt.seedVectors(); err != nil {
		t.Fatalf("seedVectors: %v", err)
	}
	if !db.Has(context.Background(), "v1") {
		t.Fatal("bootstrap vector not queued")
	}
}
