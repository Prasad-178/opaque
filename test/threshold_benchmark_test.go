//go:build integration

package test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/Prasad-178/opaque/pkg/crypto/threshold"
)

// TestThresholdBenchmarkComparison runs a side-by-side comparison of
// direct mode vs threshold CKKS mode across key operations.
func TestThresholdBenchmarkComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping threshold benchmark in short mode")
	}

	rand.Seed(42)
	dim := 128

	fmt.Println("============================================================")
	fmt.Println("Direct vs Threshold CKKS Benchmark Comparison")
	fmt.Println("============================================================")

	// --- Setup ---

	fmt.Println("\n[Setup] Creating providers...")

	start := time.Now()
	direct, err := crypto.NewDirectHEProvider(1)
	if err != nil {
		t.Fatalf("NewDirectHEProvider failed: %v", err)
	}
	defer direct.Close()
	directSetup := time.Since(start)
	fmt.Printf("  Direct provider setup: %v\n", directSetup)

	thresholdConfigs := []struct {
		n, threshold int
	}{
		{3, 2},
		{5, 3},
		{7, 5},
	}

	for _, tc := range thresholdConfigs {
		start = time.Now()
		committee, err := threshold.NewCommittee(tc.n, tc.threshold)
		if err != nil {
			t.Fatalf("NewCommittee(%d,%d) failed: %v", tc.n, tc.threshold, err)
		}
		threshProv, err := crypto.NewThresholdHEProvider(committee, 1)
		if err != nil {
			t.Fatalf("NewThresholdHEProvider failed: %v", err)
		}
		setupTime := time.Since(start)
		fmt.Printf("  Threshold %d-of-%d setup: %v\n", tc.threshold, tc.n, setupTime)

		runProviderBenchmark(t, "Direct", direct, dim)
		runProviderBenchmark(t, fmt.Sprintf("Threshold_%d-of-%d", tc.threshold, tc.n), threshProv, dim)

		threshProv.Close()
	}
}

func runProviderBenchmark(t *testing.T, name string, provider crypto.HEProvider, dim int) {
	t.Helper()

	numOps := 5
	query := generateVector(dim)
	centroids := generateVectors(numOps, provider.GetParams().MaxSlots())

	fmt.Printf("\n--- %s ---\n", name)

	// Encryption
	start := time.Now()
	var cts []*interface{}
	for i := 0; i < numOps; i++ {
		_, err := provider.EncryptVector(query)
		if err != nil {
			t.Fatalf("EncryptVector failed: %v", err)
		}
	}
	encTime := time.Since(start)
	fmt.Printf("  Encrypt (%d ops): %v avg = %.2fms\n", numOps, encTime/time.Duration(numOps), float64(encTime.Milliseconds())/float64(numOps))
	_ = cts

	// HE Dot Product (acquire/release per op to avoid pool deadlock with Encrypt)
	encQuery, _ := provider.EncryptVector(query)

	start = time.Now()
	for i := 0; i < numOps; i++ {
		engine := provider.Acquire()
		_, err := engine.HomomorphicDotProduct(encQuery, centroids[i])
		provider.Release(engine)
		if err != nil {
			t.Fatalf("HomomorphicDotProduct failed: %v", err)
		}
	}
	dotTime := time.Since(start)
	fmt.Printf("  HE dot product (%d ops): %v avg = %.2fms\n", numOps, dotTime/time.Duration(numOps), float64(dotTime.Milliseconds())/float64(numOps))

	// Decryption (scalar)
	engine := provider.Acquire()
	dotResult, _ := engine.HomomorphicDotProduct(encQuery, centroids[0])
	provider.Release(engine)

	start = time.Now()
	for i := 0; i < numOps; i++ {
		_, err := provider.DecryptScalar(dotResult)
		if err != nil {
			t.Fatalf("DecryptScalar failed: %v", err)
		}
	}
	decTime := time.Since(start)
	fmt.Printf("  Decrypt scalar (%d ops): %v avg = %.2fms\n", numOps, decTime/time.Duration(numOps), float64(decTime.Milliseconds())/float64(numOps))

	// Full cycle: encrypt → dot product → decrypt (sequential, no pool contention)
	start = time.Now()
	for i := 0; i < numOps; i++ {
		enc, _ := provider.EncryptVector(query)
		eng := provider.Acquire()
		dot, _ := eng.HomomorphicDotProduct(enc, centroids[i])
		provider.Release(eng)
		_, _ = provider.DecryptScalar(dot)
	}
	fullTime := time.Since(start)
	fmt.Printf("  Full cycle (%d ops): %v avg = %.2fms\n", numOps, fullTime/time.Duration(numOps), float64(fullTime.Milliseconds())/float64(numOps))

	// Precision check
	expected := 0.0
	for i := 0; i < dim; i++ {
		expected += query[i] * centroids[0][i]
	}
	got, _ := provider.DecryptScalar(dotResult)
	fmt.Printf("  Precision: expected=%.6f, got=%.6f, diff=%.2e\n", expected, got, math.Abs(got-expected))
}

// BenchmarkDirectFullCycle128D benchmarks the full encrypt→dot→decrypt cycle in direct mode.
func BenchmarkDirectFullCycle128D(b *testing.B) {
	provider, _ := crypto.NewDirectHEProvider(1)
	defer provider.Close()
	benchFullCycle(b, provider)
}

// BenchmarkThreshold3of5FullCycle128D benchmarks the full cycle in 3-of-5 threshold mode.
func BenchmarkThreshold3of5FullCycle128D(b *testing.B) {
	committee, _ := threshold.NewCommittee(5, 3)
	provider, _ := crypto.NewThresholdHEProvider(committee, 1)
	defer provider.Close()
	benchFullCycle(b, provider)
}

func benchFullCycle(b *testing.B, provider crypto.HEProvider) {
	b.Helper()
	query := generateVector(128)
	centroid := make([]float64, provider.GetParams().MaxSlots())
	for i := 0; i < 128; i++ {
		centroid[i] = float64(i+1) / 128.0
	}

	// Pre-encrypt to avoid pool contention in the loop.
	encQuery, _ := provider.EncryptVector(query)

	engine := provider.Acquire()
	defer provider.Release(engine)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dot, _ := engine.HomomorphicDotProduct(encQuery, centroid)
		_, _ = engine.DecryptScalar(dot)
	}
}
