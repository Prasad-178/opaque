package cache

import (
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func TestCentroidCache(t *testing.T) {
	// Create HE parameters and encoder
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,
		LogQ:            []int{60, 45, 45, 45, 45, 45, 45, 45},
		LogP:            []int{61, 61},
		LogDefaultScale: 45,
	})
	if err != nil {
		t.Fatalf("Failed to create params: %v", err)
	}

	encoder := hefloat.NewEncoder(params)
	cache := NewCentroidCache(params, encoder)

	// Test centroids
	centroids := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}

	// Load centroids
	err = cache.LoadCentroids(centroids, params.MaxLevel())
	if err != nil {
		t.Fatalf("Failed to load centroids: %v", err)
	}

	// Verify size
	if cache.Size() != 3 {
		t.Errorf("Expected 3 cached centroids, got %d", cache.Size())
	}

	// Verify Get returns non-nil
	for i := 0; i < 3; i++ {
		pt := cache.Get(i)
		if pt == nil {
			t.Errorf("Get(%d) returned nil", i)
		}
	}

	// Verify GetAll
	all := cache.GetAll(3)
	if len(all) != 3 {
		t.Errorf("GetAll returned %d, expected 3", len(all))
	}
	for i, pt := range all {
		if pt == nil {
			t.Errorf("GetAll[%d] is nil", i)
		}
	}

	// Verify Version incremented
	if cache.Version() != 1 {
		t.Errorf("Expected version 1, got %d", cache.Version())
	}

	// Reload should increment version
	cache.LoadCentroids(centroids, params.MaxLevel())
	if cache.Version() != 2 {
		t.Errorf("Expected version 2, got %d", cache.Version())
	}

	// Test NeedsRefresh
	if cache.NeedsRefresh(centroids) {
		t.Error("NeedsRefresh should return false for unchanged centroids")
	}

	// Change a centroid
	changedCentroids := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.1}, // Changed
		{7.0, 8.0, 9.0},
	}
	if !cache.NeedsRefresh(changedCentroids) {
		t.Error("NeedsRefresh should return true for changed centroids")
	}

	// Test IsStale
	if cache.IsStale(1 * time.Hour) {
		t.Error("Cache should not be stale immediately after loading")
	}

	// Test Clear
	cache.Clear()
	if cache.Size() != 0 {
		t.Errorf("After Clear, size should be 0, got %d", cache.Size())
	}
}

func TestCentroidCacheEmpty(t *testing.T) {
	params, _ := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,
		LogQ:            []int{60, 45, 45, 45, 45, 45, 45, 45},
		LogP:            []int{61, 61},
		LogDefaultScale: 45,
	})

	encoder := hefloat.NewEncoder(params)
	cache := NewCentroidCache(params, encoder)

	// Get on empty cache
	pt := cache.Get(0)
	if pt != nil {
		t.Error("Get on empty cache should return nil")
	}

	// GetAll on empty cache
	all := cache.GetAll(3)
	for _, v := range all {
		if v != nil {
			t.Error("GetAll on empty cache should return nil entries")
		}
	}
}
