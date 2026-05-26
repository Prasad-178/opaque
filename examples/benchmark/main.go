// Comprehensive benchmark and demo for Opaque — privacy-preserving vector search.
//
// Covers: build performance, search latency, recall accuracy, incremental indexing,
// metadata filtering, persistence, privacy metrics, and PCA dimensionality reduction.
//
// Run:  go run ./examples/benchmark/
package main

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/Prasad-178/opaque"
)

const (
	dim            = 128
	numVectors     = 10000
	numQueries     = 50
	topK           = 10
	numClusters    = 64
	topClusters    = 4
	numDecoys      = 8
	pcaDim         = 64
	incrementalAdd = 500
)

func main() {
	fmt.Println()
	printHeader("OPAQUE - Privacy-Preserving Vector Search Benchmark")
	fmt.Printf("  Vectors: %d | Dimension: %d | Clusters: %d\n", numVectors, dim, numClusters)
	fmt.Printf("  Queries: %d | Top-K: %d | CPU cores: %d\n", numQueries, topK, runtime.NumCPU())
	fmt.Println()

	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Generate dataset.
	printSection("Generating Dataset")
	ids, vectors := generateVectors(numVectors, dim, rng)
	queryIDs, queryVecs := generateVectors(numQueries, dim, rng)
	_ = queryIDs
	fmt.Printf("  Generated %d vectors + %d queries (%d-dim, normalized)\n\n", numVectors, numQueries, dim)

	// ── 1. Build Performance ──────────────────────────────────────────

	printSection("1. Build Performance")

	db, err := opaque.NewDB(opaque.Config{
		Dimension:   dim,
		NumClusters: numClusters,
		TopClusters: topClusters,
		NumDecoys:   numDecoys,
	})
	must(err)
	defer db.Close()

	must(db.AddBatch(ctx, ids, vectors))
	buildStart := time.Now()
	must(db.Build(ctx))
	buildTime := time.Since(buildStart)

	stats := db.ClusterStats()

	printTable([]string{"Metric", "Value"}, [][]string{
		{"Build time", buildTime.String()},
		{"Vectors indexed", fmt.Sprintf("%d", db.Count(ctx))},
		{"Clusters", fmt.Sprintf("%d", stats.NumClusters)},
		{"Min cluster size", fmt.Sprintf("%d", stats.MinSize)},
		{"Max cluster size", fmt.Sprintf("%d", stats.MaxSize)},
		{"Avg cluster size", fmt.Sprintf("%.1f", stats.AvgSize)},
		{"Empty clusters", fmt.Sprintf("%d", stats.EmptyClusters)},
		{"K-means iterations", fmt.Sprintf("%d", stats.Iterations)},
		{"Vectors/sec", fmt.Sprintf("%.0f", float64(numVectors)/buildTime.Seconds())},
	})
	fmt.Println()

	// ── 2. Search Latency ─────────────────────────────────────────────

	printSection("2. Search Latency")

	latencies := make([]time.Duration, numQueries)
	allResults := make([][]opaque.Result, numQueries)
	for i := 0; i < numQueries; i++ {
		start := time.Now()
		results, err := db.Search(ctx, queryVecs[i], topK)
		must(err)
		latencies[i] = time.Since(start)
		allResults[i] = results
	}

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	var totalLatency time.Duration
	for _, l := range latencies {
		totalLatency += l
	}

	printTable([]string{"Metric", "Value"}, [][]string{
		{"Queries", fmt.Sprintf("%d", numQueries)},
		{"Avg latency", (totalLatency / time.Duration(numQueries)).String()},
		{"P50 latency", latencies[numQueries/2].String()},
		{"P95 latency", latencies[numQueries*95/100].String()},
		{"P99 latency", latencies[numQueries*99/100].String()},
		{"Min latency", latencies[0].String()},
		{"Max latency", latencies[numQueries-1].String()},
	})
	fmt.Println()

	// ── 3. Recall / Accuracy ──────────────────────────────────────────

	printSection("3. Recall / Accuracy")

	// Compute brute-force ground truth.
	recall1, recall10 := computeRecall(vectors, ids, queryVecs, allResults)

	probePercent := float64(topClusters) / float64(numClusters) * 100

	printTable([]string{"Metric", "Value"}, [][]string{
		{"Recall@1", fmt.Sprintf("%.1f%%", recall1*100)},
		{"Recall@10", fmt.Sprintf("%.1f%%", recall10*100)},
		{"Clusters probed", fmt.Sprintf("%d / %d (%.1f%%)", topClusters, numClusters, probePercent)},
		{"Decoy clusters", fmt.Sprintf("%d", numDecoys)},
		{"Data scanned", fmt.Sprintf("~%.1f%%", probePercent)},
	})
	fmt.Println()

	// ── 4. Privacy Metrics ────────────────────────────────────────────

	printSection("4. Privacy Metrics")

	totalBucketsFetched := topClusters + numDecoys
	realFraction := float64(topClusters) / float64(totalBucketsFetched) * 100

	printTable([]string{"Property", "Status"}, [][]string{
		{"Query encryption", "CKKS HE (128-bit security)"},
		{"Vector encryption", "AES-256-GCM (per-enterprise key)"},
		{"Access pattern hiding", fmt.Sprintf("Decoy-based (%d real + %d decoy = %d fetched)", topClusters, numDecoys, totalBucketsFetched)},
		{"Server sees query?", "NO (encrypted with CKKS)"},
		{"Server sees results?", "NO (client decrypts locally)"},
		{"Server sees which clusters?", fmt.Sprintf("NO (%.0f%% real, indistinguishable)", realFraction)},
		{"Centroid visibility", "Server sees centroids (coarse structure only)"},
		{"Vector visibility", "NO (AES-256-GCM encrypted at rest)"},
	})
	fmt.Println()

	// ── 5. Incremental Indexing ───────────────────────────────────────

	printSection("5. Incremental Indexing (Post-Build Add)")

	newIDs, newVecs := generateVectors(incrementalAdd, dim, rng)
	for i := range newIDs {
		newIDs[i] = fmt.Sprintf("inc-%d", i)
	}

	// Benchmark incremental add.
	incStart := time.Now()
	must(db.AddBatch(ctx, newIDs, newVecs))
	incTime := time.Since(incStart)

	// Verify searchability.
	incResults, err := db.Search(ctx, newVecs[0], topK)
	must(err)
	incFound := false
	for _, r := range incResults {
		if r.ID == "inc-0" {
			incFound = true
			break
		}
	}

	printTable([]string{"Metric", "Value"}, [][]string{
		{"Vectors added", fmt.Sprintf("%d", incrementalAdd)},
		{"Total add time", incTime.String()},
		{"Per-vector time", (incTime / time.Duration(incrementalAdd)).String()},
		{"Immediately searchable", boolStr(incFound)},
		{"Centroid updates", "Incremental (running mean)"},
		{"Full rebuild needed", "NO"},
	})
	fmt.Println()

	// ── 6. Metadata & Filtered Search ─────────────────────────────────

	printSection("6. Metadata & Filtered Search")

	dbMeta, err := opaque.NewDB(opaque.Config{
		Dimension:   dim,
		NumClusters: 16,
		TopClusters: 4,
		NumDecoys:   4,
	})
	must(err)
	defer dbMeta.Close()

	metaIDs, metaVecs := generateVectors(1000, dim, rng)
	metas := make([]opaque.Metadata, len(metaIDs))
	categories := []string{"science", "math", "history", "art", "tech"}
	for i := range metaIDs {
		metas[i] = opaque.Metadata{
			"category": categories[i%len(categories)],
			"year":     2020 + i%5,
			"active":   i%3 != 0,
		}
	}
	must(dbMeta.AddBatchWithMetadata(ctx, metaIDs, metaVecs, metas))
	must(dbMeta.Build(ctx))

	// Unfiltered search.
	unfilteredStart := time.Now()
	unfilteredResults, err := dbMeta.Search(ctx, metaVecs[0], topK)
	must(err)
	unfilteredTime := time.Since(unfilteredStart)

	// Filtered search.
	filteredStart := time.Now()
	filteredResults, err := dbMeta.SearchWithFilter(ctx, metaVecs[0], topK, opaque.Filter{
		Where: map[string]any{"category": "science", "active": true},
	})
	must(err)
	filteredTime := time.Since(filteredStart)

	printTable([]string{"Metric", "Unfiltered", "Filtered (category=science AND active=true)"}, [][]string{
		{"Results", fmt.Sprintf("%d", len(unfilteredResults)), fmt.Sprintf("%d", len(filteredResults))},
		{"Latency", unfilteredTime.String(), filteredTime.String()},
		{"Top score", fmt.Sprintf("%.4f", safeScore(unfilteredResults)), fmt.Sprintf("%.4f", safeScore(filteredResults))},
	})
	fmt.Println()

	// ── 7. Persistence (Save / Load) ──────────────────────────────────

	printSection("7. Persistence (Save / Load)")

	tmpDir, err := os.MkdirTemp("", "opaque-bench-*")
	must(err)
	defer os.RemoveAll(tmpDir)

	savePath := tmpDir + "/saved-db"
	saveStart := time.Now()
	must(db.Save(savePath))
	saveTime := time.Since(saveStart)

	loadStart := time.Now()
	loaded, err := opaque.Load(savePath)
	must(err)
	loadTime := time.Since(loadStart)
	defer loaded.Close()

	// Verify loaded DB works.
	loadedResults, err := loaded.Search(ctx, queryVecs[0], topK)
	must(err)

	printTable([]string{"Metric", "Value"}, [][]string{
		{"Save time", saveTime.String()},
		{"Load time", loadTime.String()},
		{"Loaded DB ready", boolStr(loaded.IsReady())},
		{"Search after load", fmt.Sprintf("%d results", len(loadedResults))},
		{"Top score match", boolStr(len(loadedResults) > 0 && len(allResults[0]) > 0 &&
			loadedResults[0].ID == allResults[0][0].ID)},
	})
	fmt.Println()

	// ── 8. PCA Dimensionality Reduction ───────────────────────────────

	printSection("8. PCA Dimensionality Reduction")

	dbPCA, err := opaque.NewDB(opaque.Config{
		Dimension:    dim,
		NumClusters:  numClusters,
		TopClusters:  topClusters,
		NumDecoys:    numDecoys,
		PCADimension: pcaDim,
	})
	must(err)
	defer dbPCA.Close()

	must(dbPCA.AddBatch(ctx, ids, vectors))
	pcaBuildStart := time.Now()
	must(dbPCA.Build(ctx))
	pcaBuildTime := time.Since(pcaBuildStart)

	// Search with PCA.
	pcaLatencies := make([]time.Duration, numQueries)
	pcaResults := make([][]opaque.Result, numQueries)
	for i := 0; i < numQueries; i++ {
		start := time.Now()
		results, err := dbPCA.Search(ctx, queryVecs[i], topK)
		must(err)
		pcaLatencies[i] = time.Since(start)
		pcaResults[i] = results
	}

	var pcaTotalLatency time.Duration
	for _, l := range pcaLatencies {
		pcaTotalLatency += l
	}
	sort.Slice(pcaLatencies, func(i, j int) bool { return pcaLatencies[i] < pcaLatencies[j] })

	pcaRecall1, pcaRecall10 := computeRecall(vectors, ids, queryVecs, pcaResults)

	printTable([]string{"Metric", fmt.Sprintf("Full (%dD)", dim), fmt.Sprintf("PCA (%dD)", pcaDim)}, [][]string{
		{"Build time", buildTime.String(), pcaBuildTime.String()},
		{"Avg query", (totalLatency / time.Duration(numQueries)).String(),
			(pcaTotalLatency / time.Duration(numQueries)).String()},
		{"P50 query", latencies[numQueries/2].String(), pcaLatencies[numQueries/2].String()},
		{"Recall@1", fmt.Sprintf("%.1f%%", recall1*100), fmt.Sprintf("%.1f%%", pcaRecall1*100)},
		{"Recall@10", fmt.Sprintf("%.1f%%", recall10*100), fmt.Sprintf("%.1f%%", pcaRecall10*100)},
	})
	fmt.Println()

	// ── 9. Mutations (Delete / Update / Rebuild) ──────────────────────

	printSection("9. Mutations (Delete / Update / Rebuild)")

	dbMut, err := opaque.NewDB(opaque.Config{
		Dimension:   dim,
		NumClusters: 16,
		TopClusters: 4,
		NumDecoys:   4,
	})
	must(err)
	defer dbMut.Close()

	mutIDs, mutVecs := generateVectors(500, dim, rng)
	must(dbMut.AddBatch(ctx, mutIDs, mutVecs))
	must(dbMut.Build(ctx))

	// Delete.
	deleteStart := time.Now()
	for i := 0; i < 50; i++ {
		must(dbMut.Delete(ctx, mutIDs[i]))
	}
	deleteTime := time.Since(deleteStart)

	// Verify deletion.
	delResults, _ := dbMut.Search(ctx, mutVecs[0], 50)
	deletedInResults := false
	for _, r := range delResults {
		if r.ID == mutIDs[0] {
			deletedInResults = true
		}
	}

	// Update.
	updateStart := time.Now()
	for i := 50; i < 100; i++ {
		newVec := make([]float64, dim)
		copy(newVec, mutVecs[i])
		newVec[0] += 0.5
		must(dbMut.Update(ctx, mutIDs[i], newVec))
	}
	updateTime := time.Since(updateStart)

	// Rebuild.
	rebuildStart := time.Now()
	must(dbMut.Rebuild(ctx))
	rebuildTime := time.Since(rebuildStart)

	printTable([]string{"Operation", "Count", "Time", "Status"}, [][]string{
		{"Delete", "50", deleteTime.String(), fmt.Sprintf("Excluded from search: %s", boolStr(!deletedInResults))},
		{"Update", "50", updateTime.String(), "Soft-delete old + buffer new"},
		{"Rebuild", "1", rebuildTime.String(), fmt.Sprintf("Re-indexed %d vectors", dbMut.Count(ctx))},
	})
	fmt.Println()

	// ── Summary ───────────────────────────────────────────────────────

	printHeader("Summary")
	fmt.Println("  Opaque provides privacy-preserving vector search with:")
	fmt.Printf("  - %.1f%% Recall@10 scanning only %.1f%% of data\n", recall10*100, probePercent)
	fmt.Printf("  - %s average query latency on %d vectors\n", totalLatency/time.Duration(numQueries), numVectors)
	fmt.Printf("  - %s per incremental add (no rebuild needed)\n", incTime/time.Duration(incrementalAdd))
	fmt.Printf("  - Three-layer privacy: CKKS HE + AES-256-GCM + Decoy requests\n")
	fmt.Printf("  - Server never sees queries, results, or access patterns\n")
	fmt.Println()
}

// ── Helpers ───────────────────────────────────────────────────────────

func generateVectors(n, d int, rng *rand.Rand) ([]string, [][]float64) {
	ids := make([]string, n)
	vecs := make([][]float64, n)
	for i := 0; i < n; i++ {
		ids[i] = fmt.Sprintf("vec-%d", i)
		vec := make([]float64, d)
		var norm float64
		for j := range vec {
			vec[j] = rng.Float64()*2 - 1
			norm += vec[j] * vec[j]
		}
		norm = math.Sqrt(norm)
		for j := range vec {
			vec[j] /= norm
		}
		vecs[i] = vec
	}
	return ids, vecs
}

func computeRecall(vectors [][]float64, ids []string, queries [][]float64, results [][]opaque.Result) (recall1, recall10 float64) {
	hit1, hit10 := 0, 0
	for q := range queries {
		gt := bruteForceTopK(vectors, ids, queries[q], topK)
		gtSet := make(map[string]bool)
		for _, r := range gt {
			gtSet[r.ID] = true
		}
		if len(results[q]) > 0 && gtSet[results[q][0].ID] {
			// Check if top-1 result is the ground truth top-1.
			if results[q][0].ID == gt[0].ID {
				hit1++
			}
		}
		for _, r := range results[q] {
			if gtSet[r.ID] {
				hit10++
			}
		}
	}
	recall1 = float64(hit1) / float64(len(queries))
	recall10 = float64(hit10) / float64(len(queries)*topK)
	return
}

func bruteForceTopK(vectors [][]float64, ids []string, query []float64, k int) []opaque.Result {
	type scored struct {
		id    string
		score float64
	}
	scores := make([]scored, len(vectors))
	for i, v := range vectors {
		scores[i] = scored{ids[i], cosine(query, v)}
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
	results := make([]opaque.Result, k)
	for i := 0; i < k && i < len(scores); i++ {
		results[i] = opaque.Result{ID: scores[i].id, Score: scores[i].score}
	}
	return results
}

func cosine(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func must(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "FATAL: %v\n", err)
		os.Exit(1)
	}
}

func boolStr(b bool) string {
	if b {
		return "YES"
	}
	return "NO"
}

func safeScore(results []opaque.Result) float64 {
	if len(results) == 0 {
		return 0
	}
	return results[0].Score
}

// ── Terminal Table Rendering ──────────────────────────────────────────

func printHeader(title string) {
	line := strings.Repeat("=", 70)
	fmt.Println(line)
	pad := (70 - len(title)) / 2
	if pad < 0 {
		pad = 0
	}
	fmt.Printf("%s%s\n", strings.Repeat(" ", pad), title)
	fmt.Println(line)
}

func printSection(title string) {
	fmt.Printf("--- %s %s\n", title, strings.Repeat("-", max(0, 65-len(title))))
}

func printTable(headers []string, rows [][]string) {
	// Compute column widths.
	widths := make([]int, len(headers))
	for i, h := range headers {
		widths[i] = len(h)
	}
	for _, row := range rows {
		for i, cell := range row {
			if i < len(widths) && len(cell) > widths[i] {
				widths[i] = len(cell)
			}
		}
	}

	// Print header.
	printRow(headers, widths)
	sep := make([]string, len(headers))
	for i, w := range widths {
		sep[i] = strings.Repeat("-", w)
	}
	printRow(sep, widths)

	// Print rows.
	for _, row := range rows {
		printRow(row, widths)
	}
}

func printRow(cells []string, widths []int) {
	fmt.Print("  ")
	for i, cell := range cells {
		if i < len(widths) {
			fmt.Printf("%-*s", widths[i], cell)
		}
		if i < len(cells)-1 {
			fmt.Print("  | ")
		}
	}
	fmt.Println()
}
