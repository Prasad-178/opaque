package opaque

import (
	"context"
	"testing"
	"time"
)

func waitFor(t *testing.T, timeout time.Duration, fn func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if fn() {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatal("condition was not met before timeout")
}

func containsResult(results []Result, id string) bool {
	for _, r := range results {
		if r.ID == id {
			return true
		}
	}
	return false
}

func TestAutoIndex_BuildAndRebuild(t *testing.T) {
	ctx := context.Background()

	db, err := NewDB(Config{
		Dimension:             4,
		NumClusters:           2,
		AutoIndexEnabled:      true,
		AutoIndexInterval:     100 * time.Millisecond,
		AutoIndexMinChanges:   1,
		AutoIndexBuildTimeout: 30 * time.Second,
	})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	if err := db.Add(ctx, "a", []float64{1, 0, 0, 0}); err != nil {
		t.Fatalf("Add a: %v", err)
	}
	if err := db.Add(ctx, "b", []float64{0, 1, 0, 0}); err != nil {
		t.Fatalf("Add b: %v", err)
	}

	waitFor(t, 20*time.Second, db.IsReady)

	newVec := []float64{0, 0, 1, 0}
	if err := db.Add(ctx, "new", newVec); err != nil {
		t.Fatalf("Add new: %v", err)
	}

	waitFor(t, 20*time.Second, func() bool {
		results, err := db.Search(ctx, newVec, 5)
		if err != nil {
			return false
		}
		return containsResult(results, "new")
	})
}

func TestAutoIndex_DisabledByDefault(t *testing.T) {
	ctx := context.Background()

	db, err := NewDB(Config{Dimension: 4, NumClusters: 2})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	if err := db.Add(ctx, "a", []float64{1, 0, 0, 0}); err != nil {
		t.Fatalf("Add: %v", err)
	}

	time.Sleep(300 * time.Millisecond)
	if db.IsReady() {
		t.Fatal("database should not auto-build when AutoIndexEnabled is false")
	}
}
