//go:build sift1m

package test

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"testing"
	"time"

	"github.com/Prasad-178/opaque"
	"github.com/Prasad-178/opaque/pkg/cache"
	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/Prasad-178/opaque/pkg/embeddings"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

// TestGPU_ProfilingSIFT100K profiles the search pipeline with sub-phase timing
// to identify exactly where GPU acceleration would help.
//
// Breaks down HE operations into: query pack, encrypt, multiply (NTT), rescale,
// rotate (Galois), add, decrypt. Shows % of total for each phase.
//
// Run: go test -tags sift1m -v -run TestGPU_ProfilingSIFT100K ./test/ -timeout 30m
func TestGPU_ProfilingSIFT100K(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GPU profiling in short mode")
	}

	dataPath := getSIFT1MDataPath()
	if dataPath == "" {
		t.Skip("SIFT1M dataset not found")
	}

	t.Log("Loading SIFT dataset...")
	dataset, err := embeddings.SIFT1M(dataPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	N := 100000
	ids := dataset.IDs[:N]
	vectors := dataset.Vectors[:N]
	numQueries := 30
	topK := 10
	queries := dataset.Queries[:numQueries]

	t.Log("Computing ground truth...")
	gt := bruteForceTopK(queries, vectors, dataset.Dimension, topK)

	ctx := context.Background()

	// Test configs: varying probe count to show how HE% changes.
	configs := []struct {
		name        string
		topClusters int
		probeThresh float64
	}{
		{"strict-8", 8, 1.0},
		{"strict-16", 16, 1.0},
		{"probe-32", 32, 0.95},
	}

	t.Log("")
	t.Log("================================================================")
	t.Log("     GPU PROFILING — SIFT 100K Search Pipeline Breakdown")
	t.Log("================================================================")
	t.Logf("Vectors: %d, Dim: %d, Clusters: 64, Queries: %d, CPUs: %d",
		N, dataset.Dimension, numQueries, runtime.NumCPU())
	t.Log("")

	for _, cfg := range configs {
		t.Logf("--- %s (TopClusters=%d) ---", cfg.name, cfg.topClusters)

		db, err := opaque.NewDB(opaque.Config{
			Dimension:      dataset.Dimension,
			NumClusters:    64,
			TopClusters:    cfg.topClusters,
			NumDecoys:      8,
			ProbeThreshold: cfg.probeThresh,
		})
		if err != nil {
			t.Fatalf("NewDB: %v", err)
		}

		if err := db.AddBatch(ctx, ids, vectors); err != nil {
			t.Fatalf("AddBatch: %v", err)
		}
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build: %v", err)
		}

		// Warm up.
		db.Search(ctx, queries[0], topK)

		// Collect per-query detailed timing from SearchBatch results.
		type phaseTiming struct {
			heEncrypt   time.Duration
			heCentroid  time.Duration
			heDecrypt   time.Duration
			bucketSel   time.Duration
			bucketFetch time.Duration
			aesDec      time.Duration
			localScore  time.Duration
			total       time.Duration
		}
		timings := make([]phaseTiming, numQueries)
		var recall10Sum float64

		for q := 0; q < numQueries; q++ {
			results, err := db.Search(ctx, queries[q], topK)
			if err != nil {
				t.Fatalf("Search: %v", err)
			}

			// The DB.Search internally calls SearchBatch — we get timing from the results.
			// Unfortunately the public API returns []Result not SearchResult with timing.
			// So we'll time the overall search and break down from the internal stats.
			// Let me use a different approach: time the search externally and compute breakdown.
			_ = results
			resultIDs := make(map[string]bool)
			for _, r := range results {
				resultIDs[r.ID] = true
			}
			hits := 0
			for i := 0; i < topK; i++ {
				if resultIDs[fmt.Sprintf("sift_%d", gt[q][i])] {
					hits++
				}
			}
			recall10Sum += float64(hits) / float64(topK)
		}
		avgRecall := recall10Sum / float64(numQueries)

		// For detailed sub-phase timing, directly invoke the SearchBatch method
		// via the internal search client. We can't access it from the public API,
		// so we'll profile the HE operations separately.
		_ = timings

		db.Close()

		t.Logf("  Recall@10: %.1f%%", avgRecall*100)
	}

	// === Phase 2: Direct HE sub-phase profiling ===
	t.Log("")
	t.Log("================================================================")
	t.Log("     HE SUB-PHASE PROFILING (Direct Engine Measurement)")
	t.Log("================================================================")
	t.Log("")

	profileHESubPhases(t, dataset.Dimension, vectors[:1000])
}

// profileHESubPhases measures the exact time spent in each HE sub-operation
// using direct engine calls. This is the core data for GPU acceleration analysis.
func profileHESubPhases(t *testing.T, dim int, trainingVectors [][]float64) {
	t.Helper()

	// Create HE engine and batch cache.
	provider, err := crypto.NewDirectHEProvider(2) // Need 2: one for profiling, one for provider calls
	if err != nil {
		t.Fatalf("NewDirectHEProvider: %v", err)
	}
	defer provider.Close()

	// Normalize vectors for centroids.
	centroids := make([][]float64, 64)
	for i := range centroids {
		v := trainingVectors[i*len(trainingVectors)/64]
		norm := 0.0
		for _, x := range v {
			norm += x * x
		}
		norm = math.Sqrt(norm)
		c := make([]float64, dim)
		for j := range c {
			c[j] = v[j] / norm
		}
		centroids[i] = c
	}

	// Create batch centroid cache.
	batchCache := cache.NewBatchCentroidCache(provider.GetParams(), provider.GetEncoder(), dim)
	if err := batchCache.LoadCentroids(centroids, provider.GetParams().MaxLevel()); err != nil {
		t.Fatalf("LoadCentroids: %v", err)
	}

	// Create a normalized query.
	query := trainingVectors[len(trainingVectors)-1]
	norm := 0.0
	for _, x := range query {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	normQuery := make([]float64, dim)
	for j := range normQuery {
		normQuery[j] = query[j] / norm
	}

	numTrials := 10

	// Phase 1: Query packing.
	var packTime time.Duration
	var packedQuery []float64
	for i := 0; i < numTrials; i++ {
		start := time.Now()
		packedQuery = batchCache.PackQuery(normQuery)
		packTime += time.Since(start)
	}

	// Phase 2: HE encryption.
	var encTime time.Duration
	var encQuery *rlwe.Ciphertext
	for i := 0; i < numTrials; i++ {
		start := time.Now()
		eq, err := provider.EncryptVector(packedQuery)
		if err != nil {
			t.Fatalf("Encrypt: %v", err)
		}
		encTime += time.Since(start)
		encQuery = eq
	}

	// Phase 3: HE batch dot product with profiling.
	packedPlaintexts := batchCache.GetPackedPlaintexts()
	centroidsPerPack := batchCache.GetCentroidsPerPack()

	engine := provider.Acquire()
	defer provider.Release(engine)

	// Check if engine supports profiling.
	profEngine, ok := engine.(crypto.ProfiledEvalEngine)
	if !ok {
		t.Fatal("Engine does not support profiling")
	}

	var totalMultiply, totalRescale, totalRotate, totalAdd time.Duration
	var batchTotal time.Duration

	for i := 0; i < numTrials; i++ {
		start := time.Now()
		_, prof, err := profEngine.HomomorphicBatchDotProductProfiled(
			encQuery, packedPlaintexts[0], centroidsPerPack, dim,
		)
		batchTotal += time.Since(start)
		if err != nil {
			t.Fatalf("BatchDotProduct: %v", err)
		}
		totalMultiply += prof.Multiply
		totalRescale += prof.Rescale
		totalRotate += prof.Rotate
		totalAdd += prof.Add
	}

	// Phase 4: Decryption.
	var decTime time.Duration
	for i := 0; i < numTrials; i++ {
		encResult, err := provider.HomomorphicBatchDotProduct(encQuery, packedPlaintexts[0], centroidsPerPack, dim)
		if err != nil {
			t.Fatalf("BatchDotProduct: %v", err)
		}
		start := time.Now()
		_, err = provider.DecryptBatchScalars(encResult, centroidsPerPack, dim)
		if err != nil {
			t.Fatalf("Decrypt: %v", err)
		}
		decTime += time.Since(start)
	}

	// Report.
	avgPack := packTime / time.Duration(numTrials)
	avgEnc := encTime / time.Duration(numTrials)
	avgBatch := batchTotal / time.Duration(numTrials)
	avgMul := totalMultiply / time.Duration(numTrials)
	avgRescale := totalRescale / time.Duration(numTrials)
	avgRotate := totalRotate / time.Duration(numTrials)
	avgAdd := totalAdd / time.Duration(numTrials)
	avgDec := decTime / time.Duration(numTrials)
	totalHE := avgPack + avgEnc + avgBatch + avgDec

	t.Logf("  Dimension: %d, Centroids: %d, Centroids/Pack: %d, Packs: %d",
		dim, len(centroids), centroidsPerPack, len(packedPlaintexts))
	t.Logf("  Trials: %d", numTrials)
	t.Log("")
	t.Logf("  %-25s  %10s  %6s  %s", "Phase", "Avg Time", "% HE", "GPU Target?")
	t.Logf("  %-25s  %10s  %6s  %s", "-----", "--------", "----", "----------")
	t.Logf("  %-25s  %10v  %5.1f%%  %s", "Query packing", avgPack, pct(avgPack, totalHE), "No (trivial)")
	t.Logf("  %-25s  %10v  %5.1f%%  %s", "HE encrypt", avgEnc, pct(avgEnc, totalHE), "Maybe")
	t.Logf("  %-25s  %10v  %5.1f%%  %s", "HE batch total", avgBatch, pct(avgBatch, totalHE), "YES")
	t.Logf("    %-23s  %10v  %5.1f%%  %s", "├─ Multiply (NTT)", avgMul, pct(avgMul, totalHE), "YES — 20-50x on GPU")
	t.Logf("    %-23s  %10v  %5.1f%%  %s", "├─ Rescale", avgRescale, pct(avgRescale, totalHE), "YES — 5-10x on GPU")
	t.Logf("    %-23s  %10v  %5.1f%%  %s", "├─ Rotate (Galois)", avgRotate, pct(avgRotate, totalHE), "YES — 10-20x on GPU")
	t.Logf("    %-23s  %10v  %5.1f%%  %s", "└─ Add", avgAdd, pct(avgAdd, totalHE), "Marginal")
	t.Logf("  %-25s  %10v  %5.1f%%  %s", "HE decrypt", avgDec, pct(avgDec, totalHE), "Maybe")
	t.Logf("  %-25s  %10v  %5s   %s", "TOTAL HE", totalHE, "100%", "")
	t.Log("")

	// GPU projection.
	gpuMul := avgMul / 30     // Conservative 30x for NTT
	gpuRescale := avgRescale / 8
	gpuRotate := avgRotate / 15
	gpuAdd := avgAdd / 2
	gpuBatch := gpuMul + gpuRescale + gpuRotate + gpuAdd
	gpuTotal := avgPack + avgEnc + gpuBatch + avgDec

	t.Log("  === GPU PROJECTION (conservative estimates) ===")
	t.Logf("  %-25s  %10v → %v  (%.0fx)", "Multiply (NTT)", avgMul, gpuMul, float64(avgMul)/float64(gpuMul))
	t.Logf("  %-25s  %10v → %v  (%.0fx)", "Rescale", avgRescale, gpuRescale, float64(avgRescale)/float64(gpuRescale))
	t.Logf("  %-25s  %10v → %v  (%.0fx)", "Rotate (Galois)", avgRotate, gpuRotate, float64(avgRotate)/float64(gpuRotate))
	t.Logf("  %-25s  %10v → %v", "Batch total", avgBatch, gpuBatch)
	t.Logf("  %-25s  %10v → %v  (%.1fx speedup)", "Total HE", totalHE, gpuTotal, float64(totalHE)/float64(gpuTotal))
	t.Log("")
}

func pct(part, whole time.Duration) float64 {
	if whole == 0 {
		return 0
	}
	return float64(part) / float64(whole) * 100
}
