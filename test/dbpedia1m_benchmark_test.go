//go:build dbpedia1m

package test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/Prasad-178/opaque"
	"github.com/Prasad-178/opaque/pkg/embeddings"
)

// getDBpedia1MDataPath finds the DBpedia-OpenAI-1M dataset directory.
// Expects dbpedia_base.fvecs + dbpedia_query.fvecs produced by
// scripts/download_dbpedia1m.sh.
func getDBpedia1MDataPath() string {
	candidates := []string{
		"../data/dbpedia",
		"../../data/dbpedia",
		"../../../data/dbpedia",
	}
	for _, path := range candidates {
		absPath, _ := filepath.Abs(path)
		if _, err := os.Stat(filepath.Join(absPath, "dbpedia_base.fvecs")); err == nil {
			return absPath
		}
	}
	return ""
}

// TestDBpedia1MAccuracy benchmarks privacy-preserving search on 1M ada-002
// (1536-dim) text embeddings — the real-world high-dim text-retrieval workload
// the headline SIFT1M numbers don't directly speak to.
//
// Default NC=128. If SIFT NC=256 bench shows clear win, mirror in NC=256 test.
//
// Run: go test -tags dbpedia1m -v -run TestDBpedia1MAccuracy ./test/ -timeout 120m
func TestDBpedia1MAccuracy(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping DBpedia1M benchmark in short mode")
	}

	dataPath := getDBpedia1MDataPath()
	if dataPath == "" {
		t.Skip("DBpedia1M dataset not found; run scripts/download_dbpedia1m.sh first")
	}

	ctx := context.Background()

	// Native-float32 load — saves ~6 GB at 1M × 1536-dim vs the float64
	// upcasting Dataset wrapper. We still call the float64 loader once for
	// the brute-force GT path because bruteForceTopK takes float64; we'll
	// drop that copy immediately after GT is computed.
	t.Log("Loading DBpedia1M base (float32 native)...")
	loadStart := time.Now()
	baseF32, ids, dim, err := embeddings.DBpedia1MF32(dataPath)
	if err != nil {
		t.Fatalf("Failed to load DBpedia1M base: %v", err)
	}
	t.Logf("Loaded %d vectors (%d-dim, float32) in %v", len(baseF32), dim, time.Since(loadStart))

	queriesF64, err := embeddings.DBpedia1MQueries(dataPath)
	if err != nil {
		t.Fatalf("Failed to load DBpedia1M queries: %v", err)
	}
	if len(queriesF64) == 0 {
		t.Fatal("DBpedia1M loader returned no queries — dbpedia_query.fvecs missing or empty")
	}

	numClusters := 128
	numQueries := 50
	topK := 10
	numDecoys := 8
	numVectors := len(baseF32)

	if numQueries > len(queriesF64) {
		numQueries = len(queriesF64)
	}

	// Brute-force GT on the full base. 1M × 50 × 1536 dot products ≈ 75 GFLOP — minutes.
	// bruteForceTopK takes float64 — do a one-shot upcast that we drop immediately
	// after GT is computed. Peaks ~12 GB transient (vs holding it throughout).
	t.Log("Computing brute-force cosine ground truth (1M × 1536-dim)...")
	gtStart := time.Now()
	baseF64 := make([][]float64, len(baseF32))
	for i, v := range baseF32 {
		row := make([]float64, len(v))
		for j, x := range v {
			row[j] = float64(x)
		}
		baseF64[i] = row
	}
	groundTruth := bruteForceTopK(queriesF64[:numQueries], baseF64, dim, topK)
	t.Logf("Ground truth computed in %v", time.Since(gtStart))

	// Drop the f64 GT scratch + our local f32 base — per-config reload below
	// re-fetches a fresh f32 copy and AddBatchF32 avoids the float64 boundary
	// transient entirely. Test-side memory: 0 GB held during build phases.
	baseF64 = nil
	baseF32 = nil
	queries := queriesF64[:numQueries]
	queriesF64 = nil
	runtime.GC()

	// Trimmed config set — get a few comparison points, not exhaustive.
	configs := []struct {
		name        string
		topClusters int
		probeThresh float64
	}{
		{"probe-8", 8, 0.95},   // 6.25% — primary headline candidate
		{"probe-16", 16, 0.95}, // 12.5% — higher-recall variant
	}

	t.Log("")
	t.Log("================================================================")
	t.Log("    DBpedia1M ACCURACY BENCHMARK (1M × 1536-dim ada-002)")
	t.Log("================================================================")
	t.Logf("Vectors:     %d", numVectors)
	t.Logf("Dimension:   %d", dim)
	t.Logf("Clusters:    %d (~%d vectors each)", numClusters, numVectors/numClusters)
	t.Logf("Queries:     %d, TopK: %d", numQueries, topK)
	t.Logf("Decoys:      %d", numDecoys)
	t.Logf("CPUs:        %d", runtime.NumCPU())
	t.Log("")

	type result struct {
		name      string
		topK      int
		probe     float64
		buildTime time.Duration
		avgQuery  time.Duration
		recall1   float64
		recall10  float64
	}
	var results []result

	for _, cfg := range configs {
		t.Logf("--- Config: %s (TopClusters=%d, ProbeThreshold=%.2f) ---",
			cfg.name, cfg.topClusters, cfg.probeThresh)

		// File-backed storage offloads ~12 GB of ciphertexts to disk vs the
		// default in-memory backend, which is critical at 1M × 1536-dim where
		// the build path's peak otherwise OOMs a 64 GB instance.
		storageDir := filepath.Join(os.TempDir(), fmt.Sprintf("opaque-dbpedia-%s", cfg.name))
		os.RemoveAll(storageDir)
		db, err := opaque.NewDB(opaque.Config{
			Dimension:      dim,
			NumClusters:    numClusters,
			TopClusters:    cfg.topClusters,
			NumDecoys:      numDecoys,
			ProbeThreshold: cfg.probeThresh,
			PaddingMode:    opaque.PaddingBucketed,
			TargetEpsilon:  2.5,
			Storage:        opaque.File,
			StoragePath:    storageDir,
		})
		if err != nil {
			t.Fatalf("NewDB failed: %v", err)
		}

		// Re-load fresh per config in native float32 (the master copy was
		// freed before the loop). ~14 s per re-load × N configs is small
		// vs the per-build wall time. AddBatchF32 skips the float64→float32
		// transient — caller's f32 + db's f32 copy = ~12 GB transient, then
		// caller's f32 freed → 6 GB during build.
		freshLoadStart := time.Now()
		freshF32, _, _, err := embeddings.DBpedia1MF32(dataPath)
		if err != nil {
			t.Fatalf("re-load failed: %v", err)
		}
		t.Logf("  re-loaded %d vectors in %v", len(freshF32), time.Since(freshLoadStart))

		if err := db.AddBatchF32(ctx, ids, freshF32); err != nil {
			t.Fatalf("AddBatchF32 failed: %v", err)
		}
		freshF32 = nil
		runtime.GC()

		buildStart := time.Now()
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build failed: %v", err)
		}
		buildTime := time.Since(buildStart)

		db.Search(ctx, queries[0], topK)

		var totalLatency time.Duration
		var recall1Sum, recall10Sum float64

		for q := 0; q < numQueries; q++ {
			start := time.Now()
			searchResults, err := db.Search(ctx, queries[q], topK)
			totalLatency += time.Since(start)
			if err != nil {
				t.Fatalf("Search %d failed: %v", q, err)
			}

			resultIDs := make(map[string]bool)
			for _, r := range searchResults {
				resultIDs[r.ID] = true
			}

			gtID := fmt.Sprintf("dbpedia_%d", groundTruth[q][0])
			if resultIDs[gtID] {
				recall1Sum++
			}

			hits := 0
			for i := 0; i < topK; i++ {
				gtID := fmt.Sprintf("dbpedia_%d", groundTruth[q][i])
				if resultIDs[gtID] {
					hits++
				}
			}
			recall10Sum += float64(hits) / float64(topK)
		}

		avgLatency := totalLatency / time.Duration(numQueries)
		r1 := recall1Sum / float64(numQueries)
		r10 := recall10Sum / float64(numQueries)

		results = append(results, result{
			name:      cfg.name,
			topK:      cfg.topClusters,
			probe:     cfg.probeThresh,
			buildTime: buildTime,
			avgQuery:  avgLatency,
			recall1:   r1,
			recall10:  r10,
		})

		t.Logf("  Build: %v, Recall@1: %.1f%%, Recall@10: %.1f%%, Avg Query: %v",
			buildTime.Round(time.Millisecond), r1*100, r10*100, avgLatency)

		db.Close()
	}

	t.Log("")
	t.Log("================================================================")
	t.Log("              RESULTS — DBpedia1M (1536-dim)")
	t.Log("================================================================")
	t.Log("")
	t.Logf("  %-12s  %5s  %9s  %9s  %10s", "Config", "Probe", "Recall@1", "Recall@10", "Avg Query")
	t.Logf("  %-12s  %5s  %9s  %9s  %10s", "------", "-----", "--------", "---------", "---------")
	for _, r := range results {
		t.Logf("  %-12s  %4d%%   %7.1f%%   %7.1f%%   %9v",
			r.name,
			r.topK*100/numClusters,
			r.recall1*100,
			r.recall10*100,
			r.avgQuery)
	}
	t.Log("")
	t.Log("================================================================")
}

// TestPQ_DBpedia1M benchmarks PQ-accelerated search at 1536-dim — the regime
// where PQ should win bigger than at SIFT1M's 128-dim. Codebook size grows with
// number of subspaces M; common ada-002 choices: M=48 (32-dim subspaces) and
// M=96 (16-dim subspaces).
//
// Run: go test -tags dbpedia1m -v -run TestPQ_DBpedia1M ./test/ -timeout 120m
func TestPQ_DBpedia1M(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping DBpedia1M PQ benchmark in short mode")
	}

	dataPath := getDBpedia1MDataPath()
	if dataPath == "" {
		t.Skip("DBpedia1M dataset not found; run scripts/download_dbpedia1m.sh first")
	}

	ctx := context.Background()

	// Native-float32 load — see TestDBpedia1MAccuracy for memory rationale.
	t.Log("Loading DBpedia1M base (float32 native)...")
	loadStart := time.Now()
	baseF32, ids, dim, err := embeddings.DBpedia1MF32(dataPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Logf("Loaded %d vectors (%d-dim, float32) in %v", len(baseF32), dim, time.Since(loadStart))

	queriesF64, err := embeddings.DBpedia1MQueries(dataPath)
	if err != nil {
		t.Fatalf("Failed to load DBpedia1M queries: %v", err)
	}
	if len(queriesF64) == 0 {
		t.Fatal("DBpedia1M loader returned no queries — dbpedia_query.fvecs missing or empty")
	}

	N := len(baseF32)

	numClusters := 128
	numQueries := 50
	topK := 10
	numDecoys := 8

	if numQueries > len(queriesF64) {
		numQueries = len(queriesF64)
	}
	queries := queriesF64[:numQueries]

	// One-shot upcast for the brute-force GT path then drop the f64 copy.
	t.Log("Computing brute-force cosine ground truth (1M × 1536-dim)...")
	gtStart := time.Now()
	baseF64 := make([][]float64, len(baseF32))
	for i, v := range baseF32 {
		row := make([]float64, len(v))
		for j, x := range v {
			row[j] = float64(x)
		}
		baseF64[i] = row
	}
	groundTruth := bruteForceTopK(queries, baseF64, dim, topK)
	t.Logf("Ground truth computed in %v", time.Since(gtStart))

	// Drop everything held in the test's scope before the per-config loop.
	baseF64 = nil
	baseF32 = nil
	queriesF64 = nil
	runtime.GC()

	t.Log("")
	t.Log("================================================================")
	t.Log("   DBpedia1M (1536-dim ada-002) — PQ vs Standard")
	t.Log("================================================================")
	t.Logf("Vectors:     %d", N)
	t.Logf("Clusters:    %d (~%d vectors each)", numClusters, N/numClusters)
	t.Logf("Queries:     %d, TopK: %d", numQueries, topK)
	t.Logf("Decoys:      %d per query", numDecoys)
	t.Logf("CPUs:        %d", runtime.NumCPU())
	t.Log("")

	type benchResult struct {
		name      string
		buildTime time.Duration
		avgQuery  time.Duration
		p50Query  time.Duration
		recall1   float64
		recall10  float64
	}
	var results []benchResult

	// Trimmed config set — three meaningful PQ comparison points.
	// PQ M values must divide 1536. M=48 ⇒ 32-dim subspaces. M=96 ⇒ 16-dim.
	configs := []struct {
		name        string
		pqM         int
		topClusters int
		probeThresh float64
		epsilon     float64
	}{
		{"PQ-M48-probe8-eps25", 48, 8, 0.95, 2.5},   // headline candidate (fastest)
		{"PQ-M96-probe16-eps25", 96, 16, 0.95, 2.5}, // finer PQ + more probing for recall
		{"standard-probe8-eps25", 0, 8, 0.95, 2.5},  // PQ-vs-standard comparison
	}

	for _, cfg := range configs {
		t.Logf("--- %s (TopClusters=%d, PQ=%d, probe=%.2f, eps=%.2f) ---",
			cfg.name, cfg.topClusters, cfg.pqM, cfg.probeThresh, cfg.epsilon)

		// File-backed storage — see TestDBpedia1MAccuracy for memory rationale.
		storageDir := filepath.Join(os.TempDir(), fmt.Sprintf("opaque-dbpedia-pq-%s", cfg.name))
		os.RemoveAll(storageDir)
		dbCfg := opaque.Config{
			Dimension:      dim,
			NumClusters:    numClusters,
			TopClusters:    cfg.topClusters,
			NumDecoys:      numDecoys,
			ProbeThreshold: cfg.probeThresh,
			PaddingMode:    opaque.PaddingBucketed,
			TargetEpsilon:  cfg.epsilon,
			Storage:        opaque.File,
			StoragePath:    storageDir,
		}
		if cfg.pqM > 0 {
			dbCfg.PQSubspaces = cfg.pqM
		}

		db, err := opaque.NewDB(dbCfg)
		if err != nil {
			t.Fatalf("NewDB: %v", err)
		}

		// Re-load fresh per config — float32 native + AddBatchF32 saves the
		// float64-transient peak. See TestDBpedia1MAccuracy.
		freshLoadStart := time.Now()
		freshF32, _, _, err := embeddings.DBpedia1MF32(dataPath)
		if err != nil {
			t.Fatalf("re-load failed: %v", err)
		}
		t.Logf("  re-loaded %d vectors in %v", len(freshF32), time.Since(freshLoadStart))

		if err := db.AddBatchF32(ctx, ids, freshF32); err != nil {
			t.Fatalf("AddBatchF32: %v", err)
		}
		freshF32 = nil
		runtime.GC()

		buildStart := time.Now()
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build: %v", err)
		}
		buildTime := time.Since(buildStart)

		db.Search(ctx, queries[0], topK)

		latencies := make([]time.Duration, numQueries)
		var recall1Sum, recall10Sum float64

		for q := 0; q < numQueries; q++ {
			start := time.Now()
			searchResults, err := db.Search(ctx, queries[q], topK)
			latencies[q] = time.Since(start)
			if err != nil {
				t.Fatalf("Search %d: %v", q, err)
			}

			resultIDs := make(map[string]bool)
			for _, r := range searchResults {
				resultIDs[r.ID] = true
			}

			gtID := fmt.Sprintf("dbpedia_%d", groundTruth[q][0])
			if resultIDs[gtID] {
				recall1Sum++
			}

			hits := 0
			for i := 0; i < topK; i++ {
				gtID := fmt.Sprintf("dbpedia_%d", groundTruth[q][i])
				if resultIDs[gtID] {
					hits++
				}
			}
			recall10Sum += float64(hits) / float64(topK)
		}

		var totalLatency time.Duration
		for _, l := range latencies {
			totalLatency += l
		}
		avgLatency := totalLatency / time.Duration(numQueries)

		sorted := make([]time.Duration, numQueries)
		copy(sorted, latencies)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
		p50 := sorted[numQueries/2]

		r1 := recall1Sum / float64(numQueries)
		r10 := recall10Sum / float64(numQueries)

		results = append(results, benchResult{
			name:      cfg.name,
			buildTime: buildTime,
			avgQuery:  avgLatency,
			p50Query:  p50,
			recall1:   r1,
			recall10:  r10,
		})

		t.Logf("  Build: %v, Recall@1: %.1f%%, Recall@10: %.1f%%, Avg: %v, P50: %v",
			buildTime.Round(time.Millisecond), r1*100, r10*100,
			avgLatency.Round(time.Millisecond), p50.Round(time.Millisecond))

		db.Close()
	}

	t.Log("")
	t.Log("================================================================")
	t.Log("              RESULTS — DBpedia1M PQ")
	t.Log("================================================================")
	t.Log("")
	t.Logf("  %-22s  %9s  %9s  %10s  %10s  %12s", "Config", "Recall@1", "Recall@10", "Avg Query", "P50 Query", "Build")
	t.Logf("  %-22s  %9s  %9s  %10s  %10s  %12s", "------", "--------", "---------", "---------", "---------", "-----")
	for _, r := range results {
		t.Logf("  %-22s  %7.1f%%   %7.1f%%   %9v  %9v  %11v",
			r.name,
			r.recall1*100,
			r.recall10*100,
			r.avgQuery.Round(time.Millisecond),
			r.p50Query.Round(time.Millisecond),
			r.buildTime.Round(time.Millisecond))
	}

	if len(results) >= 2 {
		std := results[len(results)-1] // last entry is the standard-no-PQ baseline
		t.Log("")
		t.Log("  Speedup vs standard-probe8:")
		for _, r := range results[:len(results)-1] {
			speedup := float64(std.avgQuery) / float64(r.avgQuery)
			recallDelta := r.recall10 - std.recall10
			t.Logf("    %-22s  %.2fx  %+.1f pp Recall@10", r.name, speedup, recallDelta*100)
		}
	}

	t.Log("")
	t.Log("================================================================")
}
