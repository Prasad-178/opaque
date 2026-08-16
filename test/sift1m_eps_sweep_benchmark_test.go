//go:build sift1m

package test

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/Prasad-178/opaque"
	"github.com/Prasad-178/opaque/pkg/embeddings"
)

// TestSIFT1M_EpsilonSweep produces the "tunable knob" money figure: the
// access-pattern-privacy vs latency tradeoff curve.
//
// The search config is held FIXED (NC=128, strict top-8, no PQ, Bucketed
// padding) and only the DP privacy parameter ε is swept. Smaller ε ⇒ more
// decoy clusters fetched per query ⇒ stronger access-pattern privacy
// (per-query distinguishability ≤ e^ε) ⇒ higher latency. Recall is
// ε-independent — decoys never change which real clusters are fetched — so the
// curve is pure privacy-vs-latency at (approximately) fixed recall, which is
// exactly the story the paper's headline contribution needs.
//
// Deterministic decoy count at strict probe:
//
//	NumDecoys = ⌈(NumClusters − TopClusters) · e^(−ε)⌉
//
// Emits both a human-readable table and a CSV block for plotting.
func TestSIFT1M_EpsilonSweep(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT1M epsilon sweep in short mode")
	}
	dataPath := getSIFT1MDataPath()
	if dataPath == "" {
		t.Skip("SIFT1M dataset not found; run scripts/download_sift1m.sh first")
	}

	ctx := context.Background()

	t.Log("Loading SIFT1M dataset...")
	dataset, err := embeddings.SIFT1M(dataPath)
	if err != nil {
		t.Fatalf("Failed to load SIFT1M: %v", err)
	}
	t.Logf("Loaded %d vectors (%d-dim)", len(dataset.Vectors), dataset.Dimension)

	const (
		numClusters = 128
		topClusters = 8 // strict probe ⇒ deterministic K_real = 8
		numQueries  = 50
		topK        = 10
	)

	// The privacy sweep. Lower ε = stronger privacy = more decoys = slower.
	// ε=0.5 ⇒ 73 decoys (near-max of 120 pool); ε=5.0 ⇒ 1 decoy.
	epsilons := []float64{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}

	nq := numQueries
	if nq > len(dataset.Queries) {
		nq = len(dataset.Queries)
	}

	t.Log("Computing brute-force ground truth...")
	groundTruth := bruteForceTopK(dataset.Queries[:nq], dataset.Vectors, dataset.Dimension, topK)

	t.Log("")
	t.Log("================================================================")
	t.Log("      SIFT1M ε-SWEEP — access-pattern privacy vs latency")
	t.Log("================================================================")
	t.Logf("Vectors: %d | Dim: %d | Clusters: %d | strict top-%d | Queries: %d | CPUs: %d",
		len(dataset.Vectors), dataset.Dimension, numClusters, topClusters, nq, runtime.NumCPU())
	t.Log("")

	type row struct {
		eps         float64
		decoys      int
		distinBound float64
		recall1     float64
		recall10    float64
		avgQuery    time.Duration
		p50         time.Duration
	}
	var rows []row

	for _, eps := range epsilons {
		decoys := int(math.Ceil(float64(numClusters-topClusters) * math.Exp(-eps)))
		t.Logf("--- ε=%.2f  (derived decoys=%d, distinguishability ≤ e^ε=%.2f) ---",
			eps, decoys, math.Exp(eps))

		db, err := opaque.NewDB(opaque.Config{
			Dimension:      dataset.Dimension,
			NumClusters:    numClusters,
			TopClusters:    topClusters,
			ProbeThreshold: 1.0, // strict ⇒ deterministic decoy count
			PaddingMode:    opaque.PaddingBucketed,
			TargetEpsilon:  eps, // supersedes NumDecoys at search time
		})
		if err != nil {
			t.Fatalf("NewDB failed (ε=%.2f): %v", eps, err)
		}
		if err := db.AddBatch(ctx, dataset.IDs, dataset.Vectors); err != nil {
			t.Fatalf("AddBatch failed: %v", err)
		}
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build failed (ε=%.2f): %v", eps, err)
		}

		db.Search(ctx, dataset.Queries[0], topK) // warm up

		lat := make([]time.Duration, 0, nq)
		var recall1Sum, recall10Sum float64
		for q := 0; q < nq; q++ {
			start := time.Now()
			res, err := db.Search(ctx, dataset.Queries[q], topK)
			lat = append(lat, time.Since(start))
			if err != nil {
				t.Fatalf("Search %d failed (ε=%.2f): %v", q, eps, err)
			}
			ids := make(map[string]bool, len(res))
			for _, r := range res {
				ids[r.ID] = true
			}
			if ids[fmt.Sprintf("sift_%d", groundTruth[q][0])] {
				recall1Sum++
			}
			hits := 0
			for i := 0; i < topK; i++ {
				if ids[fmt.Sprintf("sift_%d", groundTruth[q][i])] {
					hits++
				}
			}
			recall10Sum += float64(hits) / float64(topK)
		}
		db.Close()

		var total time.Duration
		for _, d := range lat {
			total += d
		}
		sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })

		rows = append(rows, row{
			eps:         eps,
			decoys:      decoys,
			distinBound: math.Exp(eps),
			recall1:     recall1Sum / float64(nq),
			recall10:    recall10Sum / float64(nq),
			avgQuery:    total / time.Duration(nq),
			p50:         lat[len(lat)/2],
		})
		t.Logf("  decoys=%d  R@1=%.1f%%  R@10=%.1f%%  avg=%v  p50=%v",
			decoys, recall1Sum/float64(nq)*100, recall10Sum/float64(nq)*100,
			total/time.Duration(nq), lat[len(lat)/2])
	}

	// Money-figure table.
	t.Log("")
	t.Log("================================================================")
	t.Log("   ε-SWEEP RESULTS (recall ~fixed; privacy traded for latency)")
	t.Log("================================================================")
	t.Logf("  %5s  %7s  %10s  %8s  %9s  %10s  %10s",
		"eps", "decoys", "<=e^eps", "R@1", "R@10", "avg", "p50")
	t.Logf("  %5s  %7s  %10s  %8s  %9s  %10s  %10s",
		"-----", "------", "------", "-----", "-----", "----", "----")
	for _, r := range rows {
		t.Logf("  %5.2f  %7d  %10.2f  %6.1f%%  %7.1f%%  %10v  %10v",
			r.eps, r.decoys, r.distinBound, r.recall1*100, r.recall10*100, r.avgQuery, r.p50)
	}

	// CSV block — copy/paste into the plotting script for the paper figure.
	t.Log("")
	t.Log("CSV_BEGIN")
	t.Log("eps,decoys,distinguishability_bound,recall1,recall10,avg_ms,p50_ms")
	for _, r := range rows {
		t.Logf("%.2f,%d,%.4f,%.4f,%.4f,%.3f,%.3f",
			r.eps, r.decoys, r.distinBound, r.recall1, r.recall10,
			float64(r.avgQuery.Microseconds())/1000.0, float64(r.p50.Microseconds())/1000.0)
	}
	t.Log("CSV_END")
	t.Log("================================================================")
}
