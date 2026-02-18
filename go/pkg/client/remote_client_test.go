package client

import (
	"testing"
)

func TestDefaultRemoteClientConfig(t *testing.T) {
	cfg := DefaultRemoteClientConfig()

	if cfg.NumDecoys != 8 {
		t.Errorf("expected NumDecoys=8, got %d", cfg.NumDecoys)
	}
	if cfg.ProbeThreshold != 0.95 {
		t.Errorf("expected ProbeThreshold=0.95, got %f", cfg.ProbeThreshold)
	}
	if cfg.MaxProbeClusters != 48 {
		t.Errorf("expected MaxProbeClusters=48, got %d", cfg.MaxProbeClusters)
	}
	if cfg.TopSelect != 16 {
		t.Errorf("expected TopSelect=16, got %d", cfg.TopSelect)
	}
	if cfg.ServerURL != "http://localhost:8080" {
		t.Errorf("expected ServerURL=http://localhost:8080, got %s", cfg.ServerURL)
	}
}

func TestGenerateDecoySupers(t *testing.T) {
	t.Run("basic generation", func(t *testing.T) {
		selected := []int{0, 1, 2}
		numTotal := 64
		numDecoys := 8

		decoys := generateDecoySupers(selected, numTotal, numDecoys)

		if len(decoys) != numDecoys {
			t.Errorf("expected %d decoys, got %d", numDecoys, len(decoys))
		}

		// Verify no overlap with selected
		selectedSet := make(map[int]bool)
		for _, s := range selected {
			selectedSet[s] = true
		}
		for _, d := range decoys {
			if selectedSet[d] {
				t.Errorf("decoy %d overlaps with selected", d)
			}
		}

		// Verify all decoys are in valid range
		for _, d := range decoys {
			if d < 0 || d >= numTotal {
				t.Errorf("decoy %d out of range [0, %d)", d, numTotal)
			}
		}

		// Verify no duplicate decoys
		seen := make(map[int]bool)
		for _, d := range decoys {
			if seen[d] {
				t.Errorf("duplicate decoy %d", d)
			}
			seen[d] = true
		}
	})

	t.Run("zero decoys", func(t *testing.T) {
		selected := []int{0, 1, 2}
		decoys := generateDecoySupers(selected, 64, 0)
		if decoys != nil {
			t.Errorf("expected nil for 0 decoys, got %v", decoys)
		}
	})

	t.Run("negative decoys", func(t *testing.T) {
		selected := []int{0, 1, 2}
		decoys := generateDecoySupers(selected, 64, -1)
		if decoys != nil {
			t.Errorf("expected nil for negative decoys, got %v", decoys)
		}
	})

	t.Run("more decoys than available", func(t *testing.T) {
		selected := []int{0, 1, 2, 3, 4}
		numTotal := 8
		numDecoys := 10

		decoys := generateDecoySupers(selected, numTotal, numDecoys)

		// Should get at most numTotal - len(selected) = 3 decoys
		maxDecoys := numTotal - len(selected)
		if len(decoys) > maxDecoys {
			t.Errorf("expected at most %d decoys, got %d", maxDecoys, len(decoys))
		}
	})

	t.Run("all selected", func(t *testing.T) {
		selected := []int{0, 1, 2, 3}
		decoys := generateDecoySupers(selected, 4, 5)
		if len(decoys) != 0 {
			t.Errorf("expected 0 decoys when all selected, got %d", len(decoys))
		}
	})

	t.Run("uses full range", func(t *testing.T) {
		// With 10 non-selected, requesting 5 decoys should give 5
		selected := []int{0, 1, 2}
		numTotal := 13
		decoys := generateDecoySupers(selected, numTotal, 5)

		if len(decoys) != 5 {
			t.Errorf("expected 5 decoys, got %d", len(decoys))
		}

		// All should be from non-selected range
		selectedSet := make(map[int]bool)
		for _, s := range selected {
			selectedSet[s] = true
		}
		for _, d := range decoys {
			if selectedSet[d] {
				t.Errorf("decoy %d overlaps with selected", d)
			}
			if d < 0 || d >= numTotal {
				t.Errorf("decoy %d out of range [0, %d)", d, numTotal)
			}
		}
	})
}

func TestRemoteClient_DecoyIntegration(t *testing.T) {
	// Test that the decoy + shuffle logic produces correct output structure
	t.Run("allSupers contains real plus decoys", func(t *testing.T) {
		topSupers := []int{0, 5, 10, 15}
		numClusters := 64
		numDecoys := 8

		decoySupers := generateDecoySupers(topSupers, numClusters, numDecoys)
		allSupers := append(append([]int{}, topSupers...), decoySupers...)

		expectedLen := len(topSupers) + len(decoySupers)
		if len(allSupers) != expectedLen {
			t.Errorf("expected allSupers length %d, got %d", expectedLen, len(allSupers))
		}

		// Verify all real buckets are present
		allSet := make(map[int]bool)
		for _, s := range allSupers {
			allSet[s] = true
		}
		for _, r := range topSupers {
			if !allSet[r] {
				t.Errorf("real bucket %d missing from allSupers", r)
			}
		}
	})

	t.Run("shuffle changes order", func(t *testing.T) {
		// With enough elements, shuffle should change order at least sometimes
		original := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
		for attempt := 0; attempt < 10; attempt++ {
			shuffled := make([]int, len(original))
			copy(shuffled, original)
			shuffleInts(shuffled)

			different := false
			for i := range original {
				if original[i] != shuffled[i] {
					different = true
					break
				}
			}
			if different {
				return // Success - shuffle changed the order
			}
		}
		t.Error("shuffle did not change order in 10 attempts (extremely unlikely)")
	})
}
