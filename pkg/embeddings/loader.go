package embeddings

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// SIFT1M loads the SIFT1M dataset from the given directory.
// Expected files:
//   - sift_base.fvecs: 1M base vectors (128-dim)
//   - sift_query.fvecs: 10K query vectors
//   - sift_groundtruth.ivecs: Ground truth nearest neighbors
//
// Download from: http://corpus-texmex.irisa.fr/
func SIFT1M(dir string) (*Dataset, error) {
	basePath := filepath.Join(dir, "sift_base.fvecs")
	queryPath := filepath.Join(dir, "sift_query.fvecs")
	gtPath := filepath.Join(dir, "sift_groundtruth.ivecs")

	// Check if files exist
	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("base vectors not found: %s\nDownload from http://corpus-texmex.irisa.fr/", basePath)
	}

	// Load base vectors
	vectors, err := LoadFvecs(basePath)
	if err != nil {
		return nil, fmt.Errorf("failed to load base vectors: %w", err)
	}

	// Load queries (optional)
	var queries [][]float64
	if _, err := os.Stat(queryPath); err == nil {
		queries, err = LoadFvecs(queryPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load queries: %w", err)
		}
	}

	// Load ground truth (optional)
	var groundTruth [][]int
	if _, err := os.Stat(gtPath); err == nil {
		groundTruth, err = LoadIvecs(gtPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load ground truth: %w", err)
		}
	}

	// Generate IDs
	ids := make([]string, len(vectors))
	for i := range ids {
		ids[i] = fmt.Sprintf("sift_%d", i)
	}

	dim := 0
	if len(vectors) > 0 {
		dim = len(vectors[0])
	}

	return &Dataset{
		Name:        "sift1m",
		Dimension:   dim,
		Vectors:     vectors,
		IDs:         ids,
		Queries:     queries,
		GroundTruth: groundTruth,
	}, nil
}

// SIFT10K loads a smaller subset for quick testing.
// Uses the same format as SIFT1M but with 10K vectors.
func SIFT10K(dir string) (*Dataset, error) {
	basePath := filepath.Join(dir, "siftsmall_base.fvecs")
	queryPath := filepath.Join(dir, "siftsmall_query.fvecs")
	gtPath := filepath.Join(dir, "siftsmall_groundtruth.ivecs")

	// Check if files exist
	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("base vectors not found: %s\nDownload from http://corpus-texmex.irisa.fr/", basePath)
	}

	// Load base vectors
	vectors, err := LoadFvecs(basePath)
	if err != nil {
		return nil, fmt.Errorf("failed to load base vectors: %w", err)
	}

	// Load queries (optional)
	var queries [][]float64
	if _, err := os.Stat(queryPath); err == nil {
		queries, err = LoadFvecs(queryPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load queries: %w", err)
		}
	}

	// Load ground truth (optional)
	var groundTruth [][]int
	if _, err := os.Stat(gtPath); err == nil {
		groundTruth, err = LoadIvecs(gtPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load ground truth: %w", err)
		}
	}

	// Generate IDs
	ids := make([]string, len(vectors))
	for i := range ids {
		ids[i] = fmt.Sprintf("sift_%d", i)
	}

	dim := 0
	if len(vectors) > 0 {
		dim = len(vectors[0])
	}

	return &Dataset{
		Name:        "sift10k",
		Dimension:   dim,
		Vectors:     vectors,
		IDs:         ids,
		Queries:     queries,
		GroundTruth: groundTruth,
	}, nil
}

// GIST1M loads the GIST1M dataset from the given directory.
// Expected files:
//   - gist_base.fvecs: 1M base vectors (960-dim)
//   - gist_query.fvecs: 1K query vectors
//   - gist_groundtruth.ivecs: Ground truth nearest neighbors
//
// Download from: http://corpus-texmex.irisa.fr/
func GIST1M(dir string) (*Dataset, error) {
	basePath := filepath.Join(dir, "gist_base.fvecs")
	queryPath := filepath.Join(dir, "gist_query.fvecs")
	gtPath := filepath.Join(dir, "gist_groundtruth.ivecs")

	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("base vectors not found: %s\nDownload with: scripts/download_gist1m.sh", basePath)
	}

	vectors, err := LoadFvecs(basePath)
	if err != nil {
		return nil, fmt.Errorf("failed to load base vectors: %w", err)
	}

	var queries [][]float64
	if _, err := os.Stat(queryPath); err == nil {
		queries, err = LoadFvecs(queryPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load queries: %w", err)
		}
	}

	var groundTruth [][]int
	if _, err := os.Stat(gtPath); err == nil {
		groundTruth, err = LoadIvecs(gtPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load ground truth: %w", err)
		}
	}

	ids := make([]string, len(vectors))
	for i := range ids {
		ids[i] = fmt.Sprintf("gist_%d", i)
	}

	dim := 0
	if len(vectors) > 0 {
		dim = len(vectors[0])
	}

	return &Dataset{
		Name:        "gist1m",
		Dimension:   dim,
		Vectors:     vectors,
		IDs:         ids,
		Queries:     queries,
		GroundTruth: groundTruth,
	}, nil
}

// SIFT10M loads the SIFT10M dataset (first 10M vectors from BigANN SIFT1B).
// Expected files:
//   - sift10m_base.bvecs: 10M base vectors (128-dim, uint8)
//   - sift10m_query.fvecs: query vectors (reused from SIFT1M)
//
// Download with: scripts/download_sift10m.sh
func SIFT10M(dir string) (*Dataset, error) {
	basePath := filepath.Join(dir, "sift10m_base.bvecs")
	queryPath := filepath.Join(dir, "sift10m_query.fvecs")

	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("SIFT10M base not found: %s\nDownload with: scripts/download_sift10m.sh", basePath)
	}

	vectors, err := LoadBvecs(basePath)
	if err != nil {
		return nil, fmt.Errorf("failed to load SIFT10M base vectors: %w", err)
	}

	var queries [][]float64
	if _, err := os.Stat(queryPath); err == nil {
		queries, err = LoadFvecs(queryPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load queries: %w", err)
		}
	}

	ids := make([]string, len(vectors))
	for i := range ids {
		ids[i] = fmt.Sprintf("sift10m_%d", i)
	}

	dim := 0
	if len(vectors) > 0 {
		dim = len(vectors[0])
	}

	return &Dataset{
		Name:      "sift10m",
		Dimension: dim,
		Vectors:   vectors,
		IDs:       ids,
		Queries:   queries,
	}, nil
}

// DBpedia1M loads the DBpedia-OpenAI-1M dataset from the given directory.
// 1M Wikipedia/DBpedia entity descriptions embedded with OpenAI text-embedding-ada-002
// (1536-dim float32). Source on HuggingFace: KShivendu/dbpedia-entities-openai-1M.
//
// The dataset arrives as parquet; convert it with scripts/download_dbpedia1m.sh
// which writes the following files into dir:
//   - dbpedia_base.fvecs:  base vectors (typically 999,000 × 1536)
//   - dbpedia_query.fvecs: held-out query vectors (typically 1,000 × 1536)
//
// No precomputed ground truth — tests compute brute-force cosine GT at run time.
func DBpedia1M(dir string) (*Dataset, error) {
	basePath := filepath.Join(dir, "dbpedia_base.fvecs")
	queryPath := filepath.Join(dir, "dbpedia_query.fvecs")

	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("base vectors not found: %s\nRun scripts/download_dbpedia1m.sh first", basePath)
	}

	vectors, err := LoadFvecs(basePath)
	if err != nil {
		return nil, fmt.Errorf("failed to load base vectors: %w", err)
	}

	var queries [][]float64
	if _, err := os.Stat(queryPath); err == nil {
		queries, err = LoadFvecs(queryPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load queries: %w", err)
		}
	}

	ids := make([]string, len(vectors))
	for i := range ids {
		ids[i] = fmt.Sprintf("dbpedia_%d", i)
	}

	dim := 0
	if len(vectors) > 0 {
		dim = len(vectors[0])
	}

	return &Dataset{
		Name:      "dbpedia1m",
		Dimension: dim,
		Vectors:   vectors,
		IDs:       ids,
		Queries:   queries,
	}, nil
}

// GloVe loads GloVe word vectors from a text file.
// Format: word dim1 dim2 dim3 ... dimN (space-separated, one vector per line)
//
// Download from: https://nlp.stanford.edu/projects/glove/
func GloVe(path string) (*Dataset, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, fmt.Errorf("GloVe file not found: %s\nDownload from https://nlp.stanford.edu/projects/glove/", path)
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open GloVe file: %w", err)
	}
	defer f.Close()

	var vectors [][]float64
	var ids []string
	dim := 0

	scanner := bufio.NewScanner(f)
	// GloVe lines can be long; increase buffer to 1 MB.
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		if line == "" {
			continue
		}

		// First space separates the word from the vector components.
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return nil, fmt.Errorf("line %d: expected at least word + 1 dimension, got %d fields", lineNum, len(fields))
		}

		word := fields[0]
		vecLen := len(fields) - 1

		// Set expected dimension from first vector.
		if dim == 0 {
			dim = vecLen
		} else if vecLen != dim {
			return nil, fmt.Errorf("line %d (%q): expected %d dimensions, got %d", lineNum, word, dim, vecLen)
		}

		vec := make([]float64, vecLen)
		for i := 0; i < vecLen; i++ {
			v, err := strconv.ParseFloat(fields[i+1], 64)
			if err != nil {
				return nil, fmt.Errorf("line %d (%q): failed to parse dimension %d: %w", lineNum, word, i, err)
			}
			vec[i] = v
		}

		ids = append(ids, word)
		vectors = append(vectors, vec)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading GloVe file: %w", err)
	}

	if len(vectors) == 0 {
		return nil, fmt.Errorf("no vectors found in %s", path)
	}

	return &Dataset{
		Name:      "glove",
		Dimension: dim,
		Vectors:   vectors,
		IDs:       ids,
	}, nil
}

// FromFvecs loads a dataset from a single .fvecs file.
// No queries or ground truth.
func FromFvecs(path string, name string) (*Dataset, error) {
	vectors, err := LoadFvecs(path)
	if err != nil {
		return nil, err
	}

	ids := make([]string, len(vectors))
	for i := range ids {
		ids[i] = fmt.Sprintf("%s_%d", name, i)
	}

	dim := 0
	if len(vectors) > 0 {
		dim = len(vectors[0])
	}

	return &Dataset{
		Name:      name,
		Dimension: dim,
		Vectors:   vectors,
		IDs:       ids,
	}, nil
}

// Generate creates a synthetic dataset with random vectors.
// Useful for testing when real datasets aren't available.
func Generate(n, dim int, seed int64) *Dataset {
	// Use existing random generation
	// This is for compatibility when no real dataset is available
	vectors := make([][]float64, n)
	ids := make([]string, n)

	// Simple LCG for reproducibility
	lcg := uint64(seed)
	next := func() float64 {
		lcg = lcg*6364136223846793005 + 1442695040888963407
		return float64(int64(lcg>>33)-int64(1<<30)) / float64(1<<30)
	}

	for i := 0; i < n; i++ {
		ids[i] = fmt.Sprintf("rand_%d", i)
		vectors[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = next()
		}
	}

	return &Dataset{
		Name:      "random",
		Dimension: dim,
		Vectors:   vectors,
		IDs:       ids,
	}
}
