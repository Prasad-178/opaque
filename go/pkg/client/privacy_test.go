package client

import (
	"context"
	"testing"
	"time"
)

func TestTimingObfuscator_Obfuscate(t *testing.T) {
	obfuscator := NewTimingObfuscator(50*time.Millisecond, 10*time.Millisecond)

	start := time.Now()
	done := obfuscator.Obfuscate()

	// Simulate some work
	time.Sleep(10 * time.Millisecond)

	done()
	elapsed := time.Since(start)

	// Should take at least minLatency (50ms)
	if elapsed < 50*time.Millisecond {
		t.Errorf("expected at least 50ms, got %v", elapsed)
	}

	// Should not take more than minLatency + jitterRange + some buffer
	if elapsed > 80*time.Millisecond {
		t.Errorf("expected at most ~80ms, got %v", elapsed)
	}
}

func TestTimingObfuscator_ObfuscateContext(t *testing.T) {
	obfuscator := NewTimingObfuscator(100*time.Millisecond, 0)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	start := time.Now()
	done := obfuscator.ObfuscateContext(ctx)

	// Simulate some work
	time.Sleep(5 * time.Millisecond)

	done()
	elapsed := time.Since(start)

	// Should be cancelled before minLatency due to context timeout
	if elapsed > 50*time.Millisecond {
		t.Errorf("context should have cancelled delay, got %v", elapsed)
	}
}

func TestTimingObfuscator_ZeroJitter(t *testing.T) {
	obfuscator := NewTimingObfuscator(20*time.Millisecond, 0)

	start := time.Now()
	done := obfuscator.Obfuscate()
	done()
	elapsed := time.Since(start)

	// Should be close to minLatency with no jitter
	if elapsed < 20*time.Millisecond {
		t.Errorf("expected at least 20ms, got %v", elapsed)
	}
	if elapsed > 30*time.Millisecond {
		t.Errorf("expected at most ~30ms with no jitter, got %v", elapsed)
	}
}

func TestPrivacyConfig_Defaults(t *testing.T) {
	cfg := DefaultPrivacyConfig()

	if cfg.MinLatency != 50*time.Millisecond {
		t.Errorf("default MinLatency should be 50ms, got %v", cfg.MinLatency)
	}
	if cfg.DecoyBuckets != 3 {
		t.Errorf("default DecoyBuckets should be 3, got %d", cfg.DecoyBuckets)
	}
	if !cfg.ShuffleBeforeProcess {
		t.Error("default ShuffleBeforeProcess should be true")
	}
}

func TestPrivacyConfig_HighPrivacy(t *testing.T) {
	cfg := HighPrivacyConfig()

	if cfg.MinLatency != 100*time.Millisecond {
		t.Errorf("high privacy MinLatency should be 100ms, got %v", cfg.MinLatency)
	}
	if cfg.DecoyBuckets != 10 {
		t.Errorf("high privacy DecoyBuckets should be 10, got %d", cfg.DecoyBuckets)
	}
	if !cfg.EnableDummyQueries {
		t.Error("high privacy should enable dummy queries")
	}
}

func TestPrivacyConfig_LowLatency(t *testing.T) {
	cfg := LowLatencyConfig()

	if cfg.MinLatency != 0 {
		t.Errorf("low latency MinLatency should be 0, got %v", cfg.MinLatency)
	}
	if cfg.DecoyBuckets != 0 {
		t.Errorf("low latency DecoyBuckets should be 0, got %d", cfg.DecoyBuckets)
	}
	if cfg.ShuffleBeforeProcess {
		t.Error("low latency ShuffleBeforeProcess should be false")
	}
}

func TestPrivacyMetrics_RecordQuery(t *testing.T) {
	metrics := &PrivacyMetrics{}

	// Record normal query
	metrics.RecordQuery(false, 3, 50*time.Millisecond)
	if metrics.TotalQueries != 1 {
		t.Errorf("expected 1 total query, got %d", metrics.TotalQueries)
	}
	if metrics.DummyQueries != 0 {
		t.Errorf("expected 0 dummy queries, got %d", metrics.DummyQueries)
	}
	if metrics.DecoyBucketsFetched != 3 {
		t.Errorf("expected 3 decoy buckets, got %d", metrics.DecoyBucketsFetched)
	}

	// Record dummy query
	metrics.RecordQuery(true, 5, 100*time.Millisecond)
	if metrics.TotalQueries != 2 {
		t.Errorf("expected 2 total queries, got %d", metrics.TotalQueries)
	}
	if metrics.DummyQueries != 1 {
		t.Errorf("expected 1 dummy query, got %d", metrics.DummyQueries)
	}
	if metrics.DecoyBucketsFetched != 8 {
		t.Errorf("expected 8 decoy buckets, got %d", metrics.DecoyBucketsFetched)
	}
}

func TestPrivacyMetrics_GetMetrics(t *testing.T) {
	metrics := &PrivacyMetrics{}
	metrics.RecordQuery(false, 2, 30*time.Millisecond)
	metrics.RecordQuery(true, 3, 40*time.Millisecond)

	snapshot := metrics.GetMetrics()

	// Snapshot should be a copy
	if snapshot.TotalQueries != 2 {
		t.Errorf("expected 2 total queries in snapshot, got %d", snapshot.TotalQueries)
	}

	// Modifying original shouldn't affect snapshot
	metrics.RecordQuery(false, 1, 20*time.Millisecond)
	if snapshot.TotalQueries != 2 {
		t.Errorf("snapshot should not be affected by new recordings")
	}
}

func TestShuffleSlice(t *testing.T) {
	// Create a slice with known values
	original := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	shuffled := make([]int, len(original))
	copy(shuffled, original)

	shuffleSlice(shuffled)

	// Check same length
	if len(shuffled) != len(original) {
		t.Errorf("shuffled length should match original")
	}

	// Check all elements still present
	seen := make(map[int]bool)
	for _, v := range shuffled {
		seen[v] = true
	}
	for _, v := range original {
		if !seen[v] {
			t.Errorf("element %d missing from shuffled slice", v)
		}
	}

	// Check order changed (statistical - could rarely fail)
	// With 10 elements, probability of same order is 1/10! ≈ 2.8e-7
	sameOrder := true
	for i := range original {
		if original[i] != shuffled[i] {
			sameOrder = false
			break
		}
	}
	if sameOrder {
		t.Log("Warning: shuffle resulted in same order (very rare, but possible)")
	}
}

func TestShuffleSlice_Empty(t *testing.T) {
	var empty []int
	shuffleSlice(empty)
	if len(empty) != 0 {
		t.Error("empty slice should remain empty")
	}
}

func TestShuffleSlice_SingleElement(t *testing.T) {
	single := []string{"only"}
	shuffleSlice(single)
	if len(single) != 1 || single[0] != "only" {
		t.Error("single element slice should be unchanged")
	}
}
