//go:build sift1m

package test

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// TestBridge_GPURoundTrip exercises the exact gRPC path that SIFT 100K uses,
// but with controlled inputs: a random normalized query q and centroid pack c
// whose expected batched dot products we compute in plain Go. This is the
// production twin of bridge_suite.cpp — if bridge_suite passes on the GPU but
// this test fails, the bug is in the Go client or the gRPC transport layer
// (including scale/depth metadata), not in HEonGPU compute.
func TestBridge_GPURoundTrip(t *testing.T) {
	addr := os.Getenv("GPU_HE_SERVER")
	if addr == "" {
		t.Skip("GPU_HE_SERVER not set")
	}

	provider, err := crypto.NewGPUHEProvider(crypto.GPUHEProviderConfig{
		ServerAddress: addr,
		LocalPoolSize: 1,
		Timeout:       60 * time.Second,
	})
	if err != nil {
		t.Fatalf("NewGPUHEProvider: %v", err)
	}
	defer provider.Close()
	_ = context.Background()

	// Warm up / register keys.
	if err := provider.RegisterEvalKeys(); err != nil {
		t.Fatalf("RegisterEvalKeys: %v", err)
	}

	params := provider.GetParams()
	encoder := provider.GetEncoder()
	slotCount := 1 << (params.LogN() - 1)
	const dim = 128
	packs := slotCount / dim

	r := rand.New(rand.NewSource(1337))

	// Generate ONE query+centroid set used across all cases. If a case fails
	// with identical inputs, the divergence is clearly server state, not the
	// input data.
	const nCases = 5
	baseQuery := make([]float64, dim)
	for i := range baseQuery {
		baseQuery[i] = r.Float64()*2 - 1
	}

	for tc := 0; tc < nCases; tc++ {
		// Production-style packing: single dim-vector query replicated across
		// all centroid slots. Mirrors BatchCentroidCache.PackQuery.
		base := make([]float64, dim)
		copy(base, baseQuery)
		// Normalize.
		{
			var n float64
			for _, v := range base {
				n += v * v
			}
			if n > 0 {
				s := 1.0 / math.Sqrt(n)
				for i := range base {
					base[i] *= s
				}
			}
		}
		q := make([]float64, slotCount)
		for p := 0; p < packs; p++ {
			copy(q[p*dim:], base)
		}

		// Re-seed per-case so the centroid set is identical each time.
		r2 := rand.New(rand.NewSource(99))
		c := make([]float64, slotCount)
		for i := 0; i < packs*dim; i++ {
			c[i] = r2.Float64()*2 - 1
		}
		normalizePacks(c, dim)

		expected := make([]float64, packs)
		for p := 0; p < packs; p++ {
			for d := 0; d < dim; d++ {
				expected[p] += q[p*dim+d] * c[p*dim+d]
			}
		}

		ctQ, err := provider.EncryptVector(q)
		if err != nil {
			t.Fatalf("EncryptVector: %v", err)
		}
		ptC := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(c, ptC); err != nil {
			t.Fatalf("Encode: %v", err)
		}

		res, err := provider.HomomorphicBatchDotProduct(ctQ, ptC, packs, dim)
		if err != nil {
			t.Fatalf("HomomorphicBatchDotProduct: %v", err)
		}

		// Decrypt every slot (we want to see which slots hold the dot products
		// and whether any scale / shift mangles them).
		scores, err := provider.DecryptBatchScalars(res, packs, dim)
		if err != nil {
			t.Fatalf("DecryptBatchScalars: %v", err)
		}

		var worstErr float64
		var worstP int
		for p := 0; p < packs; p++ {
			e := math.Abs(scores[p] - expected[p])
			if e > worstErr {
				worstErr = e
				worstP = p
			}
		}
		t.Logf("case %d: worst pack=%d got=%.6f want=%.6f diff=%.3e",
			tc, worstP, scores[worstP], expected[worstP], worstErr)
		// Sample a few for context.
		for p := 0; p < 4; p++ {
			t.Logf("  pack %d: got=%+.4f want=%+.4f", p, scores[p], expected[p])
		}

		if worstErr > 1e-2 {
			t.Errorf("case %d: bridge result diverges at pack %d (err=%.3e)",
				tc, worstP, worstErr)
		}
	}
	if t.Failed() {
		t.Fatal("bridge round-trip failed — gRPC or Go client path is the suspect")
	}
	fmt.Println("bridge gRPC round-trip PASSED — bug lives elsewhere")
}

func normalizePacks(v []float64, dim int) {
	for base := 0; base < len(v); base += dim {
		if base+dim > len(v) {
			return
		}
		var n float64
		for j := 0; j < dim; j++ {
			n += v[base+j] * v[base+j]
		}
		if n == 0 {
			continue
		}
		inv := 1.0 / math.Sqrt(n)
		for j := 0; j < dim; j++ {
			v[base+j] *= inv
		}
	}
}
