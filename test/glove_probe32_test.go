//go:build glove

package test

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/Prasad-178/opaque"
	"github.com/Prasad-178/opaque/pkg/embeddings"
)

// TestGloVe100K_Probe32 runs a quick probe-32 benchmark on GloVe 100K.
// Probing 32 of 32 clusters = exhaustive cluster scan, should give near-perfect recall.
//
// Run with: go test -tags glove -v -run TestGloVe100K_Probe32 ./test/ -timeout 15m
func TestGloVe100K_Probe32(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	dataPath := getGloVeDataPath()
	if dataPath == "" {
		t.Skip("GloVe 6B 300d not found")
	}

	ctx := context.Background()

	t.Log("Loading GloVe 6B 300d...")
	dataset, err := embeddings.GloVe(dataPath)
	if err != nil {
		t.Fatalf("Failed to load GloVe: %v", err)
	}

	n := 100000
	if n > len(dataset.Vectors) {
		n = len(dataset.Vectors)
	}
	vectors := dataset.Vectors[:n]
	ids := dataset.IDs[:n]

	numQueries := 50
	queryStart := n
	if queryStart+numQueries > len(dataset.Vectors) {
		queryStart = n - numQueries
	}
	queries := dataset.Vectors[queryStart : queryStart+numQueries]

	numClusters := 32
	topK := 10
	numDecoys := 8

	t.Log("Computing brute-force ground truth...")
	groundTruth := gloveBruteForceTopK(queries, vectors, ids, dataset.Dimension, topK)

	t.Log("")
	t.Logf("GloVe 100K probe-32 benchmark (%d-dim, %d clusters, %d CPUs)",
		dataset.Dimension, numClusters, runtime.NumCPU())
	t.Log("")

	db, err := opaque.NewDB(opaque.Config{
		Dimension:      dataset.Dimension,
		NumClusters:    numClusters,
		TopClusters:    32,
		NumDecoys:      numDecoys,
		ProbeThreshold: 0.95,
	})
	if err != nil {
		t.Fatalf("NewDB failed: %v", err)
	}

	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch failed: %v", err)
	}

	buildStart := time.Now()
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	t.Logf("Build: %v", time.Since(buildStart))

	// Warm up
	db.Search(ctx, queries[0], topK)

	latencies := make([]time.Duration, numQueries)
	var recall1Sum, recall10Sum float64

	for q := 0; q < numQueries; q++ {
		start := time.Now()
		searchResults, err := db.Search(ctx, queries[q], topK)
		latencies[q] = time.Since(start)

		if err != nil {
			t.Fatalf("Search %d failed: %v", q, err)
		}

		resultIDs := make(map[string]bool)
		for _, r := range searchResults {
			resultIDs[r.ID] = true
		}

		if resultIDs[groundTruth[q][0]] {
			recall1Sum++
		}

		hits := 0
		for i := 0; i < topK; i++ {
			if resultIDs[groundTruth[q][i]] {
				hits++
			}
		}
		recall10Sum += float64(hits) / float64(topK)
	}

	var totalLatency time.Duration
	for _, l := range latencies {
		totalLatency += l
	}
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	avgLatency := totalLatency / time.Duration(numQueries)
	p50Latency := latencies[numQueries/2]
	r1 := recall1Sum / float64(numQueries)
	r10 := recall10Sum / float64(numQueries)

	t.Log("")
	t.Log("================================================================")
	t.Logf("  probe-32 (100%%):  Recall@1=%.1f%%  Recall@10=%.1f%%  Avg=%v  P50=%v",
		r1*100, r10*100, avgLatency.Round(time.Millisecond), p50Latency.Round(time.Millisecond))
	t.Log("================================================================")
	fmt.Printf("\nprobe-32: R@1=%.1f%% R@10=%.1f%% Avg=%v P50=%v\n",
		r1*100, r10*100, avgLatency.Round(time.Millisecond), p50Latency.Round(time.Millisecond))

	db.Close()
}
