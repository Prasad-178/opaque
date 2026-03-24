//go:build gist

package test

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/Prasad-178/opaque/pkg/cache"
	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/Prasad-178/opaque/pkg/embeddings"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

// TestGPU_ProfilingGIST profiles HE sub-phases on 960-dim GIST vectors.
// This is the critical case for GPU acceleration: with 960-dim, only 8 centroids
// fit per CKKS pack (vs 128 for SIFT), requiring 4x more HE operations.
// HE dominates 80%+ of query time at this dimensionality.
//
// Run: go test -tags gist -v -run TestGPU_ProfilingGIST ./test/ -timeout 30m
func TestGPU_ProfilingGIST(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GIST GPU profiling in short mode")
	}

	dataPath := getGISTDataPath()
	if dataPath == "" {
		t.Skip("GIST dataset not found; run scripts/download_gist1m.sh first")
	}

	t.Log("Loading GIST dataset (960-dim)...")
	dataset, err := embeddings.GIST1M(dataPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Use first 1000 vectors for centroid simulation.
	dim := dataset.Dimension
	vectors := dataset.Vectors[:1000]

	t.Log("")
	t.Log("================================================================")
	t.Log("     HE SUB-PHASE PROFILING — GIST 960-dim (GPU Critical Case)")
	t.Log("================================================================")
	t.Logf("  This is where GPU acceleration matters most: 960-dim vectors")
	t.Logf("  require 4 CKKS packs (vs 1 for SIFT 128-dim), making HE 80%%+ of query time.")
	t.Log("")

	profileHESubPhasesGeneric(t, dim, vectors, 32) // 32 clusters for GIST
}

// profileHESubPhasesGeneric measures HE sub-phase timing for any dimension.
func profileHESubPhasesGeneric(t *testing.T, dim int, trainingVectors [][]float64, numClusters int) {
	t.Helper()

	provider, err := crypto.NewDirectHEProvider(2) // Need 2: one for profiling, one for provider calls
	if err != nil {
		t.Fatalf("NewDirectHEProvider: %v", err)
	}
	defer provider.Close()

	// Create centroids from training vectors.
	centroids := make([][]float64, numClusters)
	for i := range centroids {
		v := trainingVectors[i*len(trainingVectors)/numClusters]
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

	batchCache := cache.NewBatchCentroidCache(provider.GetParams(), provider.GetEncoder(), dim)
	if err := batchCache.LoadCentroids(centroids, provider.GetParams().MaxLevel()); err != nil {
		t.Fatalf("LoadCentroids: %v", err)
	}

	// Create normalized query.
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

	numTrials := 5 // Fewer trials for 960-dim (each op is slower)
	packedPlaintexts := batchCache.GetPackedPlaintexts()
	centroidsPerPack := batchCache.GetCentroidsPerPack()
	numPacks := len(packedPlaintexts)

	t.Logf("  Dimension: %d, Centroids: %d, Centroids/Pack: %d, Packs: %d",
		dim, numClusters, centroidsPerPack, numPacks)
	t.Logf("  Trials: %d", numTrials)

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

	// Phase 3: Profile ALL packs (GIST has 4 packs).
	engine := provider.Acquire()
	defer provider.Release(engine)

	profEngine, ok := engine.(crypto.ProfiledEvalEngine)
	if !ok {
		t.Fatal("Engine does not support profiling")
	}

	var totalMultiply, totalRescale, totalRotate, totalAdd time.Duration
	var batchTotal time.Duration

	for i := 0; i < numTrials; i++ {
		for p := 0; p < numPacks; p++ {
			start := time.Now()
			_, prof, err := profEngine.HomomorphicBatchDotProductProfiled(
				encQuery, packedPlaintexts[p], centroidsPerPack, dim,
			)
			batchTotal += time.Since(start)
			if err != nil {
				t.Fatalf("BatchDotProduct pack %d: %v", p, err)
			}
			totalMultiply += prof.Multiply
			totalRescale += prof.Rescale
			totalRotate += prof.Rotate
			totalAdd += prof.Add
		}
	}

	// Phase 4: Decryption (all packs).
	var decTime time.Duration
	for i := 0; i < numTrials; i++ {
		for p := 0; p < numPacks; p++ {
			encResult, err := provider.HomomorphicBatchDotProduct(
				encQuery, packedPlaintexts[p], centroidsPerPack, dim,
			)
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
	}

	// Report (all times are per-query = all packs summed).
	avgPack := packTime / time.Duration(numTrials)
	avgEnc := encTime / time.Duration(numTrials)
	avgBatch := batchTotal / time.Duration(numTrials)
	avgMul := totalMultiply / time.Duration(numTrials)
	avgRescale := totalRescale / time.Duration(numTrials)
	avgRotate := totalRotate / time.Duration(numTrials)
	avgAdd := totalAdd / time.Duration(numTrials)
	avgDec := decTime / time.Duration(numTrials)
	totalHE := avgPack + avgEnc + avgBatch + avgDec

	t.Log("")
	t.Logf("  %-25s  %10s  %6s  %s", "Phase", "Avg Time", "% HE", "GPU Target?")
	t.Logf("  %-25s  %10s  %6s  %s", "-----", "--------", "----", "----------")
	t.Logf("  %-25s  %10v  %5.1f%%  %s", "Query packing", avgPack, pctDur(avgPack, totalHE), "No (trivial)")
	t.Logf("  %-25s  %10v  %5.1f%%  %s", "HE encrypt", avgEnc, pctDur(avgEnc, totalHE), "Maybe")
	t.Logf("  %-25s  %10v  %5.1f%%  %s", fmt.Sprintf("HE batch (%d packs)", numPacks), avgBatch, pctDur(avgBatch, totalHE), "YES")
	t.Logf("    %-23s  %10v  %5.1f%%  %s", "├─ Multiply (NTT)", avgMul, pctDur(avgMul, totalHE), "YES — 20-50x on GPU")
	t.Logf("    %-23s  %10v  %5.1f%%  %s", "├─ Rescale", avgRescale, pctDur(avgRescale, totalHE), "YES — 5-10x on GPU")
	t.Logf("    %-23s  %10v  %5.1f%%  %s", "├─ Rotate (Galois)", avgRotate, pctDur(avgRotate, totalHE), "YES — 10-20x on GPU")
	t.Logf("    %-23s  %10v  %5.1f%%  %s", "└─ Add", avgAdd, pctDur(avgAdd, totalHE), "Marginal")
	t.Logf("  %-25s  %10v  %5.1f%%  %s", "HE decrypt", avgDec, pctDur(avgDec, totalHE), "Maybe")
	t.Logf("  %-25s  %10v  %5s   %s", "TOTAL HE (per query)", totalHE, "100%", "")
	t.Log("")

	// GPU projection.
	gpuMul := avgMul / 30
	gpuRescale := avgRescale / 8
	gpuRotate := avgRotate / 15
	gpuAdd := avgAdd / 2
	gpuBatch := gpuMul + gpuRescale + gpuRotate + gpuAdd
	gpuTotal := avgPack + avgEnc + gpuBatch + avgDec

	t.Log("  === GPU PROJECTION (conservative estimates) ===")
	t.Logf("  %-25s  %10v → %v  (%.0fx)", "Multiply (NTT)", avgMul, gpuMul, float64(avgMul)/float64(gpuMul))
	t.Logf("  %-25s  %10v → %v  (%.0fx)", "Rescale", avgRescale, gpuRescale, float64(avgRescale)/float64(gpuRescale))
	t.Logf("  %-25s  %10v → %v  (%.0fx)", "Rotate (Galois)", avgRotate, gpuRotate, float64(avgRotate)/float64(gpuRotate))
	t.Logf("  %-25s  %10v → %v", fmt.Sprintf("Batch total (%d packs)", numPacks), avgBatch, gpuBatch)
	t.Logf("  %-25s  %10v → %v  (%.1fx speedup)", "Total HE (per query)", totalHE, gpuTotal, float64(totalHE)/float64(gpuTotal))
	t.Log("")
}

func pctDur(part, whole time.Duration) float64 {
	if whole == 0 {
		return 0
	}
	return float64(part) / float64(whole) * 100
}
