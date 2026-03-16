// Privacy utilities for Tier 2 data-private search.
//
// This file provides privacy-enhancing features:
// - Timing obfuscation: Add random delays to hide query patterns
// - Query padding: Fetch extra data to hide true interest
// - Dummy queries: Background noise queries
// - Result shuffling: Randomize order before processing

package client

import (
	"context"
	"crypto/rand"
	"math/big"
	"sync"
	"time"
)

// PrivacyConfig holds privacy-related configuration.
type PrivacyConfig struct {
	// Timing obfuscation
	MinLatency   time.Duration // Minimum response time (pad short queries)
	MaxLatency   time.Duration // Maximum latency cap
	JitterRange  time.Duration // Random jitter added to responses

	// Query obfuscation
	DecoyBuckets int  // Number of random buckets to fetch
	PadResults   bool // Pad results to fixed size

	// Background noise
	EnableDummyQueries bool          // Send periodic dummy queries
	DummyQueryInterval time.Duration // How often to send dummy queries

	// Result handling
	ShuffleBeforeProcess bool // Randomize blob order before decryption
}

// DefaultPrivacyConfig returns privacy settings optimized for security.
func DefaultPrivacyConfig() PrivacyConfig {
	return PrivacyConfig{
		MinLatency:           50 * time.Millisecond,
		MaxLatency:           5 * time.Second,
		JitterRange:          20 * time.Millisecond,
		DecoyBuckets:         3,
		PadResults:           false,
		EnableDummyQueries:   false,
		DummyQueryInterval:   30 * time.Second,
		ShuffleBeforeProcess: true,
	}
}

// HighPrivacyConfig returns settings for maximum privacy (slower).
func HighPrivacyConfig() PrivacyConfig {
	return PrivacyConfig{
		MinLatency:           100 * time.Millisecond,
		MaxLatency:           10 * time.Second,
		JitterRange:          50 * time.Millisecond,
		DecoyBuckets:         10,
		PadResults:           true,
		EnableDummyQueries:   true,
		DummyQueryInterval:   10 * time.Second,
		ShuffleBeforeProcess: true,
	}
}

// LowLatencyConfig returns settings optimized for speed (less private).
func LowLatencyConfig() PrivacyConfig {
	return PrivacyConfig{
		MinLatency:           0,
		MaxLatency:           1 * time.Second,
		JitterRange:          0,
		DecoyBuckets:         0,
		PadResults:           false,
		EnableDummyQueries:   false,
		ShuffleBeforeProcess: false,
	}
}

// TimingObfuscator adds random delays to hide query timing patterns.
type TimingObfuscator struct {
	minLatency  time.Duration
	jitterRange time.Duration
}

// NewTimingObfuscator creates a new timing obfuscator.
func NewTimingObfuscator(minLatency, jitterRange time.Duration) *TimingObfuscator {
	return &TimingObfuscator{
		minLatency:  minLatency,
		jitterRange: jitterRange,
	}
}

// Obfuscate ensures the operation takes at least minLatency + random jitter.
// Call at the start of an operation, defer the returned function.
func (t *TimingObfuscator) Obfuscate() func() {
	start := time.Now()
	targetDuration := t.minLatency + t.randomDuration(t.jitterRange)

	return func() {
		elapsed := time.Since(start)
		if elapsed < targetDuration {
			time.Sleep(targetDuration - elapsed)
		}
	}
}

// ObfuscateContext is like Obfuscate but respects context cancellation.
func (t *TimingObfuscator) ObfuscateContext(ctx context.Context) func() {
	start := time.Now()
	targetDuration := t.minLatency + t.randomDuration(t.jitterRange)

	return func() {
		elapsed := time.Since(start)
		if elapsed < targetDuration {
			remaining := targetDuration - elapsed
			select {
			case <-ctx.Done():
				return
			case <-time.After(remaining):
				return
			}
		}
	}
}

// randomDuration returns a random duration between 0 and max.
func (t *TimingObfuscator) randomDuration(max time.Duration) time.Duration {
	if max <= 0 {
		return 0
	}
	n, err := rand.Int(rand.Reader, big.NewInt(int64(max)))
	if err != nil {
		return max / 2 // Fallback to middle value
	}
	return time.Duration(n.Int64())
}

// DummyQueryRunner sends periodic dummy queries to create noise.
type DummyQueryRunner struct {
	client   *Tier2Client
	interval time.Duration
	stopCh   chan struct{}
	wg       sync.WaitGroup
}

// NewDummyQueryRunner creates a runner that sends periodic dummy queries.
func NewDummyQueryRunner(client *Tier2Client, interval time.Duration) *DummyQueryRunner {
	return &DummyQueryRunner{
		client:   client,
		interval: interval,
		stopCh:   make(chan struct{}),
	}
}

// Start begins sending dummy queries in the background.
func (d *DummyQueryRunner) Start() {
	d.wg.Add(1)
	go func() {
		defer d.wg.Done()
		ticker := time.NewTicker(d.interval)
		defer ticker.Stop()

		for {
			select {
			case <-d.stopCh:
				return
			case <-ticker.C:
				d.sendDummyQuery()
			}
		}
	}()
}

// Stop stops sending dummy queries.
func (d *DummyQueryRunner) Stop() {
	close(d.stopCh)
	d.wg.Wait()
}

// sendDummyQuery sends a random query to create noise.
func (d *DummyQueryRunner) sendDummyQuery() {
	// Generate random query vector
	query := make([]float64, d.client.config.Dimension)
	for i := range query {
		n, _ := rand.Int(rand.Reader, big.NewInt(1000))
		query[i] = float64(n.Int64())/500.0 - 1.0 // Random in [-1, 1]
	}

	// Send query (ignore results)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, _ = d.client.Search(ctx, query, 1)
}

// shuffleSlice randomly reorders a slice in-place.
func shuffleSlice[T any](slice []T) {
	for i := len(slice) - 1; i > 0; i-- {
		n, err := rand.Int(rand.Reader, big.NewInt(int64(i+1)))
		if err != nil {
			continue
		}
		j := int(n.Int64())
		slice[i], slice[j] = slice[j], slice[i]
	}
}

// PrivacyMetrics tracks privacy-related statistics.
type PrivacyMetrics struct {
	TotalQueries       int64
	DummyQueries       int64
	DecoyBucketsFetched int64
	AvgObfuscationDelay time.Duration

	mu sync.Mutex
}

// RecordQuery records a query for metrics.
func (m *PrivacyMetrics) RecordQuery(isDummy bool, decoyCount int, obfuscationDelay time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.TotalQueries++
	if isDummy {
		m.DummyQueries++
	}
	m.DecoyBucketsFetched += int64(decoyCount)

	// Update running average
	if m.TotalQueries == 1 {
		m.AvgObfuscationDelay = obfuscationDelay
	} else {
		m.AvgObfuscationDelay = time.Duration(
			(int64(m.AvgObfuscationDelay)*int64(m.TotalQueries-1) + int64(obfuscationDelay)) / int64(m.TotalQueries),
		)
	}
}

// GetMetrics returns a copy of the current metrics.
func (m *PrivacyMetrics) GetMetrics() PrivacyMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	return PrivacyMetrics{
		TotalQueries:        m.TotalQueries,
		DummyQueries:        m.DummyQueries,
		DecoyBucketsFetched: m.DecoyBucketsFetched,
		AvgObfuscationDelay: m.AvgObfuscationDelay,
	}
}
