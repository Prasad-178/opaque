//go:build sift1m

package test

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/Prasad-178/opaque"
	"github.com/Prasad-178/opaque/pkg/embeddings"
)

// TestGPU_E2E_SIFT100K runs the full Opaque search pipeline through a real
// GPU HE server and compares against CPU baseline.
//
// Prerequisites:
//   - SIFT1M dataset downloaded (scripts/download_sift1m.sh)
//   - GPU HE server running (set GPU_HE_SERVER=host:port)
//
// The test builds two identical indexes (same data, clusters, decoys) and runs
// the same queries through both CPU and GPU paths. The ONLY difference is that
// the GPU path offloads HE batch dot product to the GPU server.
//
// Privacy is identical: secret key never leaves client, server only sees
// encrypted data and evaluation keys.
//
// Run on GPU instance:
//
//	GPU_HE_SERVER=localhost:50052 go test -tags sift1m -v -run TestGPU_E2E_SIFT100K ./test/ -timeout 30m
func TestGPU_E2E_SIFT100K(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GPU E2E benchmark in short mode")
	}

	gpuServer := os.Getenv("GPU_HE_SERVER")
	if gpuServer == "" {
		t.Skip("GPU_HE_SERVER not set; skipping GPU E2E test")
	}

	dataPath := getSIFT1MDataPath()
	if dataPath == "" {
		t.Skip("SIFT1M dataset not found; run scripts/download_sift1m.sh first")
	}

	ctx := context.Background()

	t.Log("Loading SIFT dataset...")
	loadStart := time.Now()
	dataset, err := embeddings.SIFT1M(dataPath)
	if err != nil {
		t.Fatalf("Failed to load SIFT: %v", err)
	}
	t.Logf("Loaded %d vectors (%d-dim) in %v", len(dataset.Vectors), dataset.Dimension, time.Since(loadStart))

	N := 100000
	if N > len(dataset.Vectors) {
		N = len(dataset.Vectors)
	}
	ids := dataset.IDs[:N]
	vectors := dataset.Vectors[:N]

	numClusters := 64
	numQueries := 30
	topK := 10
	numDecoys := 8

	if numQueries > len(dataset.Queries) {
		numQueries = len(dataset.Queries)
	}
	queries := dataset.Queries[:numQueries]

	t.Log("Computing brute-force cosine ground truth (100K vectors)...")
	gtStart := time.Now()
	groundTruth := bruteForceTopK(queries, vectors, dataset.Dimension, topK)
	t.Logf("Ground truth computed in %v", time.Since(gtStart))

	t.Log("")
	t.Log("================================================================")
	t.Log("     SIFT 100K — GPU vs CPU End-to-End Benchmark")
	t.Log("================================================================")
	t.Logf("Vectors:     %d (real SIFT 128-dim)", N)
	t.Logf("Clusters:    %d (~%d vectors each)", numClusters, N/numClusters)
	t.Logf("Queries:     %d, TopK: %d", numQueries, topK)
	t.Logf("Decoys:      %d per query", numDecoys)
	t.Logf("CPUs:        %d", runtime.NumCPU())
	t.Logf("GPU Server:  %s", gpuServer)
	t.Log("")

	type benchResult struct {
		name      string
		buildTime time.Duration
		avgQuery  time.Duration
		p50Query  time.Duration
		minQuery  time.Duration
		maxQuery  time.Duration
		recall1   float64
		recall10  float64
	}
	var results []benchResult

	configs := []struct {
		name        string
		topClusters int
		probeThresh float64
		gpuServer   string
	}{
		// CPU baseline
		{"cpu-strict8", 8, 1.0, ""},
		// GPU path
		{"gpu-strict8", 8, 1.0, gpuServer},
		// Also test probe-16 on GPU for higher recall
		{"cpu-probe16", 16, 0.95, ""},
		{"gpu-probe16", 16, 0.95, gpuServer},
	}

	for _, cfg := range configs {
		isGPU := cfg.gpuServer != ""
		label := "CPU"
		if isGPU {
			label = "GPU"
		}
		t.Logf("--- %s (%s, TopClusters=%d) ---", cfg.name, label, cfg.topClusters)

		dbCfg := opaque.Config{
			Dimension:      dataset.Dimension,
			NumClusters:    numClusters,
			TopClusters:    cfg.topClusters,
			NumDecoys:      numDecoys,
			ProbeThreshold: cfg.probeThresh,
		}
		if isGPU {
			dbCfg.GPUServerAddress = cfg.gpuServer
		}

		db, err := opaque.NewDB(dbCfg)
		if err != nil {
			t.Fatalf("NewDB(%s): %v", cfg.name, err)
		}

		if err := db.AddBatch(ctx, ids, vectors); err != nil {
			t.Fatalf("AddBatch(%s): %v", cfg.name, err)
		}

		buildStart := time.Now()
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build(%s): %v", cfg.name, err)
		}
		buildTime := time.Since(buildStart)
		t.Logf("  Build: %v", buildTime.Round(time.Millisecond))

		// Warm-up (also triggers eval key registration for GPU).
		t.Logf("  Warm-up query...")
		warmStart := time.Now()
		_, err = db.Search(ctx, queries[0], topK)
		if err != nil {
			t.Fatalf("Warm-up Search(%s): %v", cfg.name, err)
		}
		t.Logf("  Warm-up: %v (includes key registration for GPU)", time.Since(warmStart).Round(time.Millisecond))

		// Benchmark queries.
		latencies := make([]time.Duration, numQueries)
		var recall1Sum, recall10Sum float64

		for q := 0; q < numQueries; q++ {
			start := time.Now()
			searchResults, err := db.Search(ctx, queries[q], topK)
			latencies[q] = time.Since(start)

			if err != nil {
				t.Fatalf("Search(%s, q=%d): %v", cfg.name, q, err)
			}

			resultIDs := make(map[string]bool)
			for _, r := range searchResults {
				resultIDs[r.ID] = true
			}

			// Recall@1.
			gtID := fmt.Sprintf("sift_%d", groundTruth[q][0])
			if resultIDs[gtID] {
				recall1Sum++
			}

			// Recall@10.
			hits := 0
			for i := 0; i < topK; i++ {
				gtID := fmt.Sprintf("sift_%d", groundTruth[q][i])
				if resultIDs[gtID] {
					hits++
				}
			}
			recall10Sum += float64(hits) / float64(topK)
		}

		// Stats.
		var totalLatency time.Duration
		for _, l := range latencies {
			totalLatency += l
		}
		avgLatency := totalLatency / time.Duration(numQueries)

		sorted := make([]time.Duration, numQueries)
		copy(sorted, latencies)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
		p50 := sorted[numQueries/2]
		minL := sorted[0]
		maxL := sorted[len(sorted)-1]

		r1 := recall1Sum / float64(numQueries)
		r10 := recall10Sum / float64(numQueries)

		results = append(results, benchResult{
			name:      cfg.name,
			buildTime: buildTime,
			avgQuery:  avgLatency,
			p50Query:  p50,
			minQuery:  minL,
			maxQuery:  maxL,
			recall1:   r1,
			recall10:  r10,
		})

		t.Logf("  Recall@1: %.1f%%, Recall@10: %.1f%%, Avg: %v, P50: %v, Min: %v, Max: %v",
			r1*100, r10*100,
			avgLatency.Round(time.Millisecond), p50.Round(time.Millisecond),
			minL.Round(time.Millisecond), maxL.Round(time.Millisecond))

		db.Close()
		t.Log("")
	}

	// Summary table.
	t.Log("================================================================")
	t.Log("                    RESULTS SUMMARY")
	t.Log("================================================================")
	t.Log("")
	t.Logf("  %-16s  %9s  %9s  %9s  %9s  %9s  %9s",
		"Config", "Recall@1", "Recall@10", "Avg", "P50", "Min", "Max")
	t.Logf("  %-16s  %9s  %9s  %9s  %9s  %9s  %9s",
		"------", "--------", "---------", "---", "---", "---", "---")
	for _, r := range results {
		t.Logf("  %-16s  %7.1f%%   %7.1f%%   %8v  %8v  %8v  %8v",
			r.name,
			r.recall1*100, r.recall10*100,
			r.avgQuery.Round(time.Millisecond),
			r.p50Query.Round(time.Millisecond),
			r.minQuery.Round(time.Millisecond),
			r.maxQuery.Round(time.Millisecond))
	}

	// GPU vs CPU speedup.
	t.Log("")
	t.Log("  GPU vs CPU speedup:")
	for i := 0; i < len(results)-1; i += 2 {
		cpu := results[i]
		gpu := results[i+1]
		speedup := float64(cpu.avgQuery) / float64(gpu.avgQuery)
		recallDelta := gpu.recall10 - cpu.recall10
		t.Logf("    %s → %s:  %.2fx speedup, %+.1f pp Recall@10",
			cpu.name, gpu.name, speedup, recallDelta*100)
	}

	t.Log("")
	t.Log("  Privacy: IDENTICAL between CPU and GPU paths")
	t.Log("    - Secret key NEVER leaves client")
	t.Log("    - GPU server only sees encrypted ciphertexts + eval keys")
	t.Log("    - Same CKKS HE, AES-256-GCM, decoy clusters")
	t.Log("================================================================")
}
