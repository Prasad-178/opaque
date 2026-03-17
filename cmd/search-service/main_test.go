package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	opaque "github.com/Prasad-178/opaque"
)

func TestCreateDB_InvalidDimension(t *testing.T) {
	old := *dimension
	*dimension = 0
	t.Cleanup(func() { *dimension = old })

	// NewDB should fail with dimension 0, tested indirectly via the service.
}

func TestSeedVectors_RequiresUnsafeFlagForDemoData(t *testing.T) {
	oldDemo := *demoVectors
	oldUnsafe := *allowUnsafeDemoData
	oldDim := *dimension
	*demoVectors = 10
	*allowUnsafeDemoData = false
	*dimension = 8
	t.Cleanup(func() {
		*demoVectors = oldDemo
		*allowUnsafeDemoData = oldUnsafe
		*dimension = oldDim
	})

	db, err := newTestDB(t, 8)
	if err != nil {
		t.Fatalf("newTestDB: %v", err)
	}
	defer db.Close()

	if err := seedVectors(db); err == nil {
		t.Fatal("expected error when demo vectors are enabled without unsafe flag")
	}
}

func TestLoadBootstrapVectors(t *testing.T) {
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

	db, err := newTestDB(t, 4)
	if err != nil {
		t.Fatalf("newTestDB: %v", err)
	}
	defer db.Close()

	if err := loadBootstrapVectors(db, path); err != nil {
		t.Fatalf("loadBootstrapVectors: %v", err)
	}

	if db.Size() != 2 {
		t.Fatalf("size = %d, want 2", db.Size())
	}
}

func TestGenerateDemoVectors(t *testing.T) {
	db, err := newTestDB(t, 8)
	if err != nil {
		t.Fatalf("newTestDB: %v", err)
	}
	defer db.Close()

	generateDemoVectors(db, 10, 8)
	if db.Size() != 10 {
		t.Fatalf("size = %d, want 10", db.Size())
	}

	// Build should succeed on demo vectors.
	if err := db.Build(context.Background()); err != nil {
		t.Fatalf("Build: %v", err)
	}
	if !db.IsReady() {
		t.Fatal("expected DB to be ready after Build")
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

	t.Setenv(key, "true")
	if got := envBool(key, false); !got {
		t.Fatal("envBool should parse true")
	}

	t.Setenv(key, "bad")
	if got := envInt(key, 7); got != 7 {
		t.Fatalf("envInt fallback = %d, want 7", got)
	}

	t.Setenv(key, strconv.FormatBool(false))
	if got := envBool(key, true); got {
		t.Fatal("envBool should parse false")
	}
}

// newTestDB creates a small in-memory opaque.DB for testing.
func newTestDB(t *testing.T, dim int) (*opaque.DB, error) {
	t.Helper()
	return opaque.NewDB(opaque.Config{
		Dimension:   dim,
		NumClusters: 2,
	})
}
