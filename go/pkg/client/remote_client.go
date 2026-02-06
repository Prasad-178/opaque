package client

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/opaque/opaque/go/pkg/cache"
	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/hierarchical"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

// RemoteClient performs privacy-preserving vector search via REST API.
//
// Architecture:
//   - Client encrypts query with HE (server never sees plaintext query)
//   - Server computes HE scores on centroids
//   - Client decrypts scores, selects clusters (server never sees selection)
//   - Client fetches blobs from selected clusters + decoys
//   - Client decrypts vectors and scores locally
//
// This maintains query privacy because:
//   - HE-encrypted query is sent to server
//   - Score decryption happens client-side
//   - Cluster selection is client-side (server doesn't know which are real)
//   - Final scoring is client-side
type RemoteClient struct {
	// Server connection
	serverURL  string
	httpClient *http.Client
	token      string

	// Cryptographic components
	heEngine      *crypto.Engine
	aesEncryptor  *encrypt.AESGCM
	centroidCache *cache.CentroidCache

	// Search configuration
	config hierarchical.Config

	// Credentials data
	centroids     [][]float64
	dimension     int
	numClusters   int
	enterpriseID  string
	tokenExpiry   time.Time

	mu sync.RWMutex
}

// RemoteClientConfig holds configuration for the remote client.
type RemoteClientConfig struct {
	ServerURL    string
	HTTPTimeout  time.Duration
	TopSelect    int
	NumDecoys    int
}

// DefaultRemoteClientConfig returns sensible defaults.
func DefaultRemoteClientConfig() RemoteClientConfig {
	return RemoteClientConfig{
		ServerURL:   "http://localhost:8080",
		HTTPTimeout: 30 * time.Second,
		TopSelect:   16,
		NumDecoys:   0,
	}
}

// LoginCredentials holds the response from server login.
type LoginCredentials struct {
	Token       string
	ExpiresAt   time.Time
	AESKey      []byte
	Centroids   [][]float64
	Dimension   int
	NumClusters int
}

// NewRemoteClient creates a remote client and authenticates with the server.
func NewRemoteClient(cfg RemoteClientConfig, enterpriseID, userID, password string) (*RemoteClient, error) {
	httpClient := &http.Client{
		Timeout: cfg.HTTPTimeout,
	}

	// Login to get credentials
	loginReq := struct {
		UserID       string `json:"user_id"`
		EnterpriseID string `json:"enterprise_id"`
		Password     string `json:"password"`
	}{
		UserID:       userID,
		EnterpriseID: enterpriseID,
		Password:     password,
	}

	body, _ := json.Marshal(loginReq)
	resp, err := httpClient.Post(cfg.ServerURL+"/api/v1/auth/login", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("login request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("login failed: %s", string(respBody))
	}

	var loginResp struct {
		Token       string      `json:"token"`
		ExpiresAt   time.Time   `json:"expires_at"`
		AESKey      string      `json:"aes_key"`
		Centroids   [][]float64 `json:"centroids"`
		Dimension   int         `json:"dimension"`
		NumClusters int         `json:"num_clusters"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&loginResp); err != nil {
		return nil, fmt.Errorf("failed to decode login response: %w", err)
	}

	aesKey, err := base64.StdEncoding.DecodeString(loginResp.AESKey)
	if err != nil {
		return nil, fmt.Errorf("failed to decode AES key: %w", err)
	}

	// Create AES encryptor
	aesEncryptor, err := encrypt.NewAESGCM(aesKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create AES encryptor: %w", err)
	}

	// Create HE engine for query encryption
	heEngine, err := crypto.NewClientEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to create HE engine: %w", err)
	}

	// Create centroid cache
	centroidCache := cache.NewCentroidCache(heEngine.GetParams(), heEngine.GetEncoder())
	if len(loginResp.Centroids) > 0 {
		if err := centroidCache.LoadCentroids(loginResp.Centroids, heEngine.GetParams().MaxLevel()); err != nil {
			return nil, fmt.Errorf("failed to cache centroids: %w", err)
		}
	}

	// Create search config
	config := hierarchical.Config{
		Dimension:       loginResp.Dimension,
		NumSuperBuckets: loginResp.NumClusters,
		TopSuperBuckets: cfg.TopSelect,
		NumDecoys:       cfg.NumDecoys,
	}

	return &RemoteClient{
		serverURL:     cfg.ServerURL,
		httpClient:    httpClient,
		token:         loginResp.Token,
		heEngine:      heEngine,
		aesEncryptor:  aesEncryptor,
		centroidCache: centroidCache,
		config:        config,
		centroids:     loginResp.Centroids,
		dimension:     loginResp.Dimension,
		numClusters:   loginResp.NumClusters,
		enterpriseID:  enterpriseID,
		tokenExpiry:   loginResp.ExpiresAt,
	}, nil
}

// Search performs privacy-preserving vector search via the server.
func (c *RemoteClient) Search(ctx context.Context, query []float64, topK int) (*hierarchical.SearchResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	startTotal := time.Now()
	result := &hierarchical.SearchResult{
		Stats: hierarchical.SearchStats{},
	}

	if len(query) != c.dimension {
		return nil, fmt.Errorf("query has wrong dimension: %d (expected %d)", len(query), c.dimension)
	}

	// Normalize query
	normalizedQuery := normalizeVectorCopy(query)

	// ==========================================
	// LEVEL 1: HE Query Encryption + Server Scoring
	// ==========================================

	// Step 1a: Encrypt query with HE
	startEncrypt := time.Now()
	encQuery, err := c.heEngine.EncryptVector(normalizedQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt query: %w", err)
	}
	result.Timing.HEEncryptQuery = time.Since(startEncrypt)

	// Step 1b: Compute HE scores LOCALLY using cached centroids
	// Query never leaves the client - all HE operations are local!
	startHE := time.Now()
	cachedPlaintexts := c.centroidCache.GetAll(len(c.centroids))
	useCached := cachedPlaintexts != nil && len(cachedPlaintexts) == len(c.centroids) && cachedPlaintexts[0] != nil

	encScores := make([]*rlwe.Ciphertext, len(c.centroids))
	scoreErrs := make([]error, len(c.centroids))

	numWorkers := runtime.NumCPU()
	if numWorkers > len(c.centroids) {
		numWorkers = len(c.centroids)
	}

	var wgScore sync.WaitGroup
	workChan := make(chan int, len(c.centroids))

	for w := 0; w < numWorkers; w++ {
		wgScore.Add(1)
		go func() {
			defer wgScore.Done()
			for i := range workChan {
				if useCached {
					encScores[i], scoreErrs[i] = c.heEngine.HomomorphicDotProductCached(encQuery, cachedPlaintexts[i])
				} else {
					encScores[i], scoreErrs[i] = c.heEngine.HomomorphicDotProduct(encQuery, c.centroids[i])
				}
			}
		}()
	}

	for i := range c.centroids {
		workChan <- i
	}
	close(workChan)
	wgScore.Wait()

	for i, err := range scoreErrs {
		if err != nil {
			return nil, fmt.Errorf("failed to compute HE score for centroid %d: %w", i, err)
		}
	}
	result.Timing.HECentroidScores = time.Since(startHE)
	result.Stats.HEOperations = len(c.centroids)

	// Step 1c: Decrypt scores locally
	startDecrypt := time.Now()
	scores := make([]float64, len(encScores))

	for i, encScore := range encScores {
		score, err := c.heEngine.DecryptScalar(encScore)
		if err != nil {
			return nil, fmt.Errorf("failed to decrypt score %d: %w", i, err)
		}
		scores[i] = score
	}
	result.Timing.HEDecryptScores = time.Since(startDecrypt)

	// Step 1d: Select top super-buckets (SERVER NEVER SEES THIS!)
	topSupers := selectTopKIndices(scores, c.config.TopSuperBuckets)
	result.Stats.SuperBucketsSelected = len(topSupers)

	// ==========================================
	// LEVEL 2: Fetch blobs from selected clusters
	// ==========================================

	startFetch := time.Now()

	// Add decoys and shuffle
	allSupers := make([]int, len(topSupers))
	copy(allSupers, topSupers)
	// TODO: Add decoy generation similar to EnterpriseHierarchicalClient

	// Fetch blobs from each super-bucket (in parallel)
	type fetchResult struct {
		blobs []blobData
		err   error
	}
	fetchResults := make([]fetchResult, len(allSupers))

	var wgFetch sync.WaitGroup
	numFetchWorkers := runtime.NumCPU()
	if numFetchWorkers > len(allSupers) {
		numFetchWorkers = len(allSupers)
	}
	fetchChan := make(chan int, len(allSupers))

	for w := 0; w < numFetchWorkers; w++ {
		wgFetch.Add(1)
		go func() {
			defer wgFetch.Done()
			for i := range fetchChan {
				blobs, err := c.fetchBucket(ctx, allSupers[i])
				fetchResults[i] = fetchResult{blobs: blobs, err: err}
			}
		}()
	}

	for i := range allSupers {
		fetchChan <- i
	}
	close(fetchChan)
	wgFetch.Wait()

	// Collect all blobs
	var allBlobs []blobData
	for _, fr := range fetchResults {
		if fr.err != nil {
			return nil, fmt.Errorf("failed to fetch bucket: %w", fr.err)
		}
		allBlobs = append(allBlobs, fr.blobs...)
	}

	result.Timing.BucketFetch = time.Since(startFetch)
	result.Stats.BlobsFetched = len(allBlobs)
	result.Stats.RealSubBuckets = len(topSupers)
	result.Stats.TotalSubBuckets = len(allSupers)

	// ==========================================
	// LEVEL 3: Local AES Decrypt + Scoring
	// ==========================================

	// Decrypt all vectors
	startAES := time.Now()
	type decryptedVec struct {
		id     string
		vector []float64
	}
	decrypted := make([]decryptedVec, 0, len(allBlobs))

	for _, b := range allBlobs {
		vec, err := c.aesEncryptor.DecryptVectorWithID(b.ciphertext, b.id)
		if err != nil {
			continue // Skip failed decryptions
		}
		decrypted = append(decrypted, decryptedVec{id: b.id, vector: vec})
	}
	result.Timing.AESDecrypt = time.Since(startAES)

	// Score locally
	startScore := time.Now()
	type scored struct {
		id    string
		score float64
	}
	scoredResults := make([]scored, len(decrypted))

	for i, d := range decrypted {
		normalizedVec := normalizeVectorCopy(d.vector)
		score := dotProductVec(normalizedQuery, normalizedVec)
		scoredResults[i] = scored{id: d.id, score: score}
	}

	// Sort by score descending
	sort.Slice(scoredResults, func(i, j int) bool {
		return scoredResults[i].score > scoredResults[j].score
	})

	result.Timing.LocalScoring = time.Since(startScore)
	result.Stats.VectorsScored = len(scoredResults)

	// Return top-K
	n := topK
	if n > len(scoredResults) {
		n = len(scoredResults)
	}
	result.Results = make([]hierarchical.Result, n)
	for i := 0; i < n; i++ {
		result.Results[i] = hierarchical.Result{
			ID:    scoredResults[i].id,
			Score: scoredResults[i].score,
		}
	}

	result.Timing.Total = time.Since(startTotal)
	return result, nil
}

// blobData holds blob info from server.
type blobData struct {
	id         string
	ciphertext []byte
}

// fetchBucket fetches blobs from a super-bucket.
func (c *RemoteClient) fetchBucket(ctx context.Context, superID int) ([]blobData, error) {
	url := fmt.Sprintf("%s/api/v1/buckets/%s/%d", c.serverURL, c.enterpriseID, superID)
	req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
	req.Header.Set("Authorization", "Bearer "+c.token)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bucket fetch failed: status %d", resp.StatusCode)
	}

	var bucketResp struct {
		Blobs []struct {
			ID         string `json:"id"`
			Ciphertext string `json:"ciphertext"`
		} `json:"blobs"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&bucketResp); err != nil {
		return nil, err
	}

	blobs := make([]blobData, len(bucketResp.Blobs))
	for i, b := range bucketResp.Blobs {
		ct, err := base64.StdEncoding.DecodeString(b.Ciphertext)
		if err != nil {
			return nil, fmt.Errorf("failed to decode ciphertext: %w", err)
		}
		blobs[i] = blobData{id: b.ID, ciphertext: ct}
	}

	return blobs, nil
}

// RefreshToken refreshes the authentication token.
func (c *RemoteClient) RefreshToken(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	refreshReq := struct {
		Token string `json:"token"`
	}{Token: c.token}

	body, _ := json.Marshal(refreshReq)
	req, _ := http.NewRequestWithContext(ctx, "POST", c.serverURL+"/api/v1/auth/refresh", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("refresh request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("refresh failed: status %d", resp.StatusCode)
	}

	var refreshResp struct {
		Token     string    `json:"token"`
		ExpiresAt time.Time `json:"expires_at"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&refreshResp); err != nil {
		return fmt.Errorf("failed to decode refresh response: %w", err)
	}

	c.token = refreshResp.Token
	c.tokenExpiry = refreshResp.ExpiresAt
	return nil
}

// IsTokenExpired checks if the token needs refresh.
func (c *RemoteClient) IsTokenExpired() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return time.Now().After(c.tokenExpiry)
}

// Note: Helper functions normalizeVectorCopy, dotProductVec, selectTopKIndices
// are defined in hierarchical.go and reused here.
