package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/Prasad-178/opaque/internal/service"
	"github.com/Prasad-178/opaque/internal/store"
)

func newTestService(t *testing.T, dim int) *service.SearchService {
	t.Helper()
	svc, err := service.NewSearchService(service.Config{
		LSHNumBits:          16,
		LSHDimension:        dim,
		LSHSeed:             42,
		MaxSessionTTL:       time.Minute,
		MaxConcurrentScores: 2,
	}, store.NewMemoryStore())
	if err != nil {
		t.Fatalf("NewSearchService: %v", err)
	}
	return svc
}

func TestCreateStore_InvalidBackend(t *testing.T) {
	old := *storageBackend
	*storageBackend = "bad"
	t.Cleanup(func() { *storageBackend = old })

	if _, err := createStore(); err == nil {
		t.Fatal("expected error for unknown storage backend")
	}
}

func TestCreateStore_FileBackend(t *testing.T) {
	oldBackend := *storageBackend
	oldPath := *storagePath
	*storageBackend = "file"
	*storagePath = filepath.Join(t.TempDir(), "vectors.json")
	t.Cleanup(func() {
		*storageBackend = oldBackend
		*storagePath = oldPath
	})

	s, err := createStore()
	if err != nil {
		t.Fatalf("createStore: %v", err)
	}
	defer s.Close()

	if _, ok := s.(*store.FileStore); !ok {
		t.Fatalf("expected FileStore, got %T", s)
	}
}

func TestSeedVectors_RequiresUnsafeFlagForDemoData(t *testing.T) {
	svc := newTestService(t, 8)

	oldDemo := *demoVectors
	oldUnsafe := *allowUnsafeDemoData
	*demoVectors = 10
	*allowUnsafeDemoData = false
	t.Cleanup(func() {
		*demoVectors = oldDemo
		*allowUnsafeDemoData = oldUnsafe
	})

	if err := seedVectors(svc); err == nil {
		t.Fatal("expected error when demo vectors are enabled without unsafe flag")
	}
}

func TestLoadBootstrapVectors(t *testing.T) {
	svc := newTestService(t, 4)

	path := filepath.Join(t.TempDir(), "bootstrap.json")
	payload := []bootstrapVector{
		{ID: "v1", Values: []float64{1, 0, 0, 0}},
		{ID: "v2", Values: []float64{0, 1, 0, 0}},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	oldDim := *dimension
	*dimension = 4
	t.Cleanup(func() { *dimension = oldDim })

	if err := loadBootstrapVectors(svc, path); err != nil {
		t.Fatalf("loadBootstrapVectors: %v", err)
	}

	count, err := svc.GetVectorCount(context.Background())
	if err != nil {
		t.Fatalf("GetVectorCount: %v", err)
	}
	if count != 2 {
		t.Fatalf("count = %d, want 2", count)
	}
}

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
	if got := envInt64(key, 7); got != 12 {
		t.Fatalf("envInt64 = %d, want 12", got)
	}

	t.Setenv(key, "true")
	if got := envBool(key, false); !got {
		t.Fatal("envBool should parse true")
	}

	t.Setenv(key, "bad")
	if got := envInt(key, 7); got != 7 {
		t.Fatalf("envInt fallback = %d, want 7", got)
	}
	if got := envInt64(key, 7); got != 7 {
		t.Fatalf("envInt64 fallback = %d, want 7", got)
	}

	t.Setenv(key, strconv.FormatBool(false))
	if got := envBool(key, true); got {
		t.Fatal("envBool should parse false")
	}
}
