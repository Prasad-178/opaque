package client

import (
	"math"
	"testing"
)

// TestGenerateDecoySupersBernoulli_BasicShape covers the core invariants of
// the per-cluster Bernoulli decoy sampler:
//   - never returns a selected cluster
//   - empirical inclusion rate matches p within sampling tolerance
//   - p≤0 returns nil; p≥1 includes every non-selected cluster.
func TestGenerateDecoySupersBernoulli_BasicShape(t *testing.T) {
	const (
		numTotal = 128
		trials   = 500
	)
	selected := []int{0, 1, 2, 3, 4, 5, 6, 7} // K_real = 8 → pool = 120

	t.Run("p=0 returns nil", func(t *testing.T) {
		got := generateDecoySupersBernoulli(selected, numTotal, 0)
		if got != nil {
			t.Errorf("p=0: expected nil, got %v", got)
		}
	})

	t.Run("p=-1 returns nil", func(t *testing.T) {
		got := generateDecoySupersBernoulli(selected, numTotal, -0.5)
		if got != nil {
			t.Errorf("p<0: expected nil, got %v", got)
		}
	})

	t.Run("never returns selected cluster", func(t *testing.T) {
		selectedSet := make(map[int]bool, len(selected))
		for _, s := range selected {
			selectedSet[s] = true
		}
		for trial := 0; trial < 50; trial++ {
			got := generateDecoySupersBernoulli(selected, numTotal, 0.1)
			for _, c := range got {
				if selectedSet[c] {
					t.Errorf("trial %d: returned selected cluster %d", trial, c)
				}
				if c < 0 || c >= numTotal {
					t.Errorf("trial %d: returned out-of-range cluster %d", trial, c)
				}
			}
		}
	})

	t.Run("empirical rate matches p", func(t *testing.T) {
		const p = 0.1 // expect ~12 decoys per call
		var totalIncluded int
		for trial := 0; trial < trials; trial++ {
			got := generateDecoySupersBernoulli(selected, numTotal, p)
			totalIncluded += len(got)
		}
		// 500 trials × 120-cluster pool × p = 6000 expected inclusions.
		// Binomial std dev ≈ sqrt(500 × 120 × p × (1-p)) ≈ sqrt(5400) ≈ 73.5.
		// Allow ±5 std dev tolerance.
		expected := float64(trials) * float64(numTotal-len(selected)) * p
		stdDev := math.Sqrt(float64(trials) * float64(numTotal-len(selected)) * p * (1 - p))
		if math.Abs(float64(totalIncluded)-expected) > 5*stdDev {
			t.Errorf("empirical rate off: got %d total inclusions, expected ~%.0f ± %.1f (5σ)",
				totalIncluded, expected, 5*stdDev)
		}
	})
}

// TestGenerateDecoySupersBernoulli_VarianceVsUniform contrasts Bernoulli K
// (binomial, varies) with uniform-K (deterministic count).
func TestGenerateDecoySupersBernoulli_VarianceVsUniform(t *testing.T) {
	const (
		numTotal = 128
		trials   = 200
		p        = 0.1
	)
	selected := []int{0, 1, 2, 3, 4, 5, 6, 7}
	expectedNumDecoys := int(float64(numTotal-len(selected)) * p) // ~12

	uniformVar := 0
	bernoulliKValues := make([]int, 0, trials)
	for trial := 0; trial < trials; trial++ {
		// Uniform: always exactly expectedNumDecoys (when pool is large enough).
		uniformGot := generateDecoySupers(selected, numTotal, expectedNumDecoys)
		if len(uniformGot) != expectedNumDecoys {
			uniformVar++
		}

		// Bernoulli: variable K.
		bernoulliGot := generateDecoySupersBernoulli(selected, numTotal, p)
		bernoulliKValues = append(bernoulliKValues, len(bernoulliGot))
	}

	if uniformVar != 0 {
		t.Errorf("uniform-K should be deterministic, but %d/%d trials produced wrong count", uniformVar, trials)
	}

	// Verify Bernoulli K shows real variance.
	mean := 0.0
	for _, k := range bernoulliKValues {
		mean += float64(k)
	}
	mean /= float64(len(bernoulliKValues))
	variance := 0.0
	for _, k := range bernoulliKValues {
		variance += (float64(k) - mean) * (float64(k) - mean)
	}
	variance /= float64(len(bernoulliKValues))

	if variance < 1.0 {
		t.Errorf("Bernoulli K should have non-trivial variance, got %.2f over %d trials", variance, trials)
	}

	t.Logf("Bernoulli K stats over %d trials: mean=%.2f, variance=%.2f, expected_mean=%.2f, expected_variance=%.2f",
		trials, mean, variance, float64(numTotal-len(selected))*p, float64(numTotal-len(selected))*p*(1-p))
}
