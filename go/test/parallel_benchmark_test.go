//go:build integration

package test

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/lsh"
)

// WorkerPool manages multiple crypto engines for true parallelism
type WorkerPool struct {
	engines []*crypto.Engine
	tasks   chan func(*crypto.Engine)
	wg      sync.WaitGroup
}

func NewWorkerPool(numWorkers int) (*WorkerPool, error) {
	pool := &WorkerPool{
		engines: make([]*crypto.Engine, numWorkers),
		tasks:   make(chan func(*crypto.Engine), numWorkers*10),
	}

	// Create one engine per worker
	for i := 0; i < numWorkers; i++ {
		engine, err := crypto.NewClientEngine()
		if err != nil {
			return nil, err
		}
		pool.engines[i] = engine
	}

	// Start workers
	for i := 0; i < numWorkers; i++ {
		go pool.worker(i)
	}

	return pool, nil
}

func (p *WorkerPool) worker(id int) {
	engine := p.engines[id]
	for task := range p.tasks {
		task(engine)
		p.wg.Done()
	}
}

func (p *WorkerPool) Submit(task func(*crypto.Engine)) {
	p.wg.Add(1)
	p.tasks <- task
}

func (p *WorkerPool) Wait() {
	p.wg.Wait()
}

func (p *WorkerPool) Close() {
	close(p.tasks)
}

func (p *WorkerPool) GetEngine() *crypto.Engine {
	return p.engines[0]
}

func TestTrueParallelBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping parallel benchmark in short mode")
	}

	rand.Seed(42)
	numCPU := runtime.NumCPU()

	const (
		numVectors    = 100000
		dimension     = 128
		numQueries    = 10
		topK          = 20
		maxCandidates = 100
	)

	fmt.Println("============================================================")
	fmt.Printf("TRUE PARALLEL Benchmark: %d vectors, %d workers\n", numVectors, numCPU)
	fmt.Println("============================================================")

	// Generate data
	fmt.Printf("\n[1/5] Generating %d vectors...\n", numVectors)
	start := time.Now()
	ids, vectors := generateTestVectors(numVectors, dimension)
	fmt.Printf("   Generated in %v\n", time.Since(start))

	// Build LSH index
	fmt.Println("\n[2/5] Building LSH index...")
	start = time.Now()
	lshIndex := lsh.NewIndex(lsh.Config{
		Dimension: dimension,
		NumBits:   128,
		Seed:      42,
	})
	lshIndex.Add(ids, vectors)
	fmt.Printf("   Built in %v\n", time.Since(start))

	// Create worker pool with multiple engines
	fmt.Printf("\n[3/5] Creating worker pool with %d engines...\n", numCPU)
	start = time.Now()
	pool, err := NewWorkerPool(numCPU)
	if err != nil {
		t.Fatalf("Failed to create worker pool: %v", err)
	}
	defer pool.Close()
	fmt.Printf("   Pool created in %v\n", time.Since(start))

	// Create vector map
	vectorMap := make(map[string][]float64, numVectors)
	for i, id := range ids {
		vectorMap[id] = vectors[i]
	}

	// Benchmark: Process queries with true parallelism
	fmt.Println("\n[4/5] Benchmarking with TRUE parallelism...")

	var totalTime time.Duration
	var totalDotProductTime time.Duration

	for q := 0; q < numQueries; q++ {
		targetIdx := rand.Intn(numVectors)
		query := addNoiseVec(vectors[targetIdx], 0.1)
		normalizedQuery := crypto.NormalizeVector(query)

		queryStart := time.Now()

		// LSH search
		queryHash := lshIndex.HashBytes(normalizedQuery)
		candidates, _ := lshIndex.Search(queryHash, maxCandidates)

		// Encrypt query (single engine is fine for this)
		engine := pool.GetEngine()
		encQuery, _ := engine.EncryptVector(normalizedQuery)

		// Parallel dot products using worker pool
		numToScore := topK
		if numToScore > len(candidates) {
			numToScore = len(candidates)
		}

		dotStart := time.Now()
		results := make([]*struct {
			score interface{}
			err   error
		}, numToScore)

		for i := 0; i < numToScore; i++ {
			results[i] = &struct {
				score interface{}
				err   error
			}{}
		}

		for i := 0; i < numToScore; i++ {
			idx := i
			vec := vectorMap[candidates[idx].ID]
			pool.Submit(func(eng *crypto.Engine) {
				score, err := eng.HomomorphicDotProduct(encQuery, vec)
				results[idx].score = score
				results[idx].err = err
			})
		}
		pool.Wait()
		dotTime := time.Since(dotStart)
		totalDotProductTime += dotTime

		queryTime := time.Since(queryStart)
		totalTime += queryTime

		fmt.Printf("   Query %d: %d× dot products in %v (total: %v)\n",
			q+1, numToScore, dotTime, queryTime)
	}

	avgTime := totalTime / time.Duration(numQueries)
	avgDotTime := totalDotProductTime / time.Duration(numQueries)

	fmt.Println("\n[5/5] Summary")
	fmt.Println("============================================================")
	fmt.Printf("   Avg query time:       %v\n", avgTime)
	fmt.Printf("   Avg dot product time: %v (parallel across %d cores)\n", avgDotTime, numCPU)
	fmt.Printf("   Estimated QPS:        %.2f\n", float64(time.Second)/float64(avgTime))

	// Compare with sequential
	fmt.Println("\n--- Sequential comparison ---")
	seqStart := time.Now()
	query := vectors[0]
	normalizedQuery := crypto.NormalizeVector(query)
	encQuery, _ := pool.GetEngine().EncryptVector(normalizedQuery)
	for i := 0; i < topK; i++ {
		pool.GetEngine().HomomorphicDotProduct(encQuery, vectors[i])
	}
	seqTime := time.Since(seqStart)
	fmt.Printf("   Sequential %d× dot products: %v\n", topK, seqTime)
	fmt.Printf("   Parallel speedup: %.2fx\n", float64(seqTime)/float64(avgDotTime))
}

// Test multiple queries in parallel (batch processing)
func TestBatchQueryParallel(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping batch query test in short mode")
	}

	rand.Seed(42)

	const (
		numVectors    = 100000
		dimension     = 128
		batchSize     = 10 // Process 10 queries in parallel
		topK          = 10 // Fewer results per query for speed
		maxCandidates = 50
	)

	fmt.Println("============================================================")
	fmt.Printf("BATCH QUERY Benchmark: %d queries in parallel\n", batchSize)
	fmt.Println("============================================================")

	// Setup
	ids, vectors := generateTestVectors(numVectors, dimension)
	lshIndex := lsh.NewIndex(lsh.Config{Dimension: dimension, NumBits: 128, Seed: 42})
	lshIndex.Add(ids, vectors)

	vectorMap := make(map[string][]float64, numVectors)
	for i, id := range ids {
		vectorMap[id] = vectors[i]
	}

	// Create multiple engines for parallel query processing
	engines := make([]*crypto.Engine, batchSize)
	for i := 0; i < batchSize; i++ {
		engines[i], _ = crypto.NewClientEngine()
	}

	// Generate batch of queries
	queries := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		queries[i] = crypto.NormalizeVector(addNoiseVec(vectors[rand.Intn(numVectors)], 0.1))
	}

	// Process batch in parallel
	fmt.Println("\nProcessing batch of queries in parallel...")
	start := time.Now()

	var wg sync.WaitGroup
	queryTimes := make([]time.Duration, batchSize)

	for i := 0; i < batchSize; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			qStart := time.Now()

			// Each query uses its own engine
			engine := engines[idx]
			query := queries[idx]

			// LSH
			hash := lshIndex.HashBytes(query)
			candidates, _ := lshIndex.Search(hash, maxCandidates)

			// Encrypt
			encQuery, _ := engine.EncryptVector(query)

			// Dot products (sequential within each query)
			numToScore := topK
			if numToScore > len(candidates) {
				numToScore = len(candidates)
			}
			for j := 0; j < numToScore; j++ {
				vec := vectorMap[candidates[j].ID]
				engine.HomomorphicDotProduct(encQuery, vec)
			}

			queryTimes[idx] = time.Since(qStart)
		}(i)
	}
	wg.Wait()
	totalBatchTime := time.Since(start)

	// Results
	var sumQueryTime time.Duration
	for i, qt := range queryTimes {
		fmt.Printf("   Query %d: %v\n", i+1, qt)
		sumQueryTime += qt
	}

	fmt.Println("\n--- Summary ---")
	fmt.Printf("   Total batch time:     %v\n", totalBatchTime)
	fmt.Printf("   Sum of query times:   %v\n", sumQueryTime)
	fmt.Printf("   Parallelization efficiency: %.1f%%\n",
		float64(sumQueryTime)/float64(totalBatchTime)/float64(batchSize)*100)
	fmt.Printf("   Effective QPS:        %.2f\n", float64(batchSize)/totalBatchTime.Seconds())
}

func generateTestVectors(n, dim int) ([]string, [][]float64) {
	ids := make([]string, n)
	vectors := make([][]float64, n)
	for i := 0; i < n; i++ {
		ids[i] = fmt.Sprintf("doc_%d", i)
		vec := make([]float64, dim)
		var norm float64
		for j := 0; j < dim; j++ {
			vec[j] = rand.NormFloat64()
			norm += vec[j] * vec[j]
		}
		norm = math.Sqrt(norm)
		for j := range vec {
			vec[j] /= norm
		}
		vectors[i] = vec
	}
	return ids, vectors
}

func addNoiseVec(vec []float64, scale float64) []float64 {
	noisy := make([]float64, len(vec))
	var norm float64
	for i, v := range vec {
		noisy[i] = v + rand.NormFloat64()*scale
		norm += noisy[i] * noisy[i]
	}
	norm = math.Sqrt(norm)
	for i := range noisy {
		noisy[i] /= norm
	}
	return noisy
}
