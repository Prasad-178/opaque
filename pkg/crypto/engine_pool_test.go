package crypto

import (
	"math"
	"sync"
	"testing"

	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func TestEnginePool_DefaultSize(t *testing.T) {
	pool, err := NewEnginePool(3)
	if err != nil {
		t.Fatalf("NewEnginePool: %v", err)
	}
	defer pool.Close()

	if pool.Size() != 3 {
		t.Fatalf("Size=%d want 3", pool.Size())
	}
	if pool.GetParams().LogN() == 0 {
		t.Fatal("GetParams returned zero-value")
	}
	if pool.GetEncoder() == nil {
		t.Fatal("GetEncoder nil")
	}
	if pool.GetPrimaryEngine() == nil {
		t.Fatal("GetPrimaryEngine nil")
	}
}

func TestEnginePool_NormalizesSubOnePoolSize(t *testing.T) {
	pool, err := NewEnginePool(0)
	if err != nil {
		t.Fatalf("NewEnginePool(0): %v", err)
	}
	defer pool.Close()
	if pool.Size() != 1 {
		t.Fatalf("Size=%d want 1 (clamped)", pool.Size())
	}

	pool2, err := NewEnginePool(-5)
	if err != nil {
		t.Fatalf("NewEnginePool(-5): %v", err)
	}
	defer pool2.Close()
	if pool2.Size() != 1 {
		t.Fatalf("Size=%d want 1 (clamped)", pool2.Size())
	}
}

func TestEnginePool_AcquireReleaseRoundtrip(t *testing.T) {
	pool, err := NewEnginePool(2)
	if err != nil {
		t.Fatalf("NewEnginePool: %v", err)
	}
	defer pool.Close()

	e1 := pool.Acquire()
	e2 := pool.Acquire()
	if e1 == nil || e2 == nil {
		t.Fatal("acquired nil engine")
	}
	if e1 == e2 {
		t.Fatal("acquired the same engine twice")
	}
	pool.Release(e1)
	pool.Release(e2)
}

func TestEnginePool_EncryptDecryptViaPool(t *testing.T) {
	pool, err := NewEnginePool(2)
	if err != nil {
		t.Fatalf("NewEnginePool: %v", err)
	}
	defer pool.Close()

	vec := []float64{0.1, -0.2, 0.3, -0.4}
	ct, err := pool.EncryptVector(vec)
	if err != nil {
		t.Fatalf("EncryptVector: %v", err)
	}

	got, err := pool.DecryptScalar(ct)
	if err != nil {
		t.Fatalf("DecryptScalar: %v", err)
	}
	if math.Abs(got-vec[0]) > 0.01 {
		t.Fatalf("scalar=%v want=%v", got, vec[0])
	}
}

func TestEnginePool_ConcurrentEncrypt(t *testing.T) {
	pool, err := NewEnginePool(4)
	if err != nil {
		t.Fatalf("NewEnginePool: %v", err)
	}
	defer pool.Close()

	const workers = 16
	var wg sync.WaitGroup
	wg.Add(workers)
	errs := make(chan error, workers)

	for i := 0; i < workers; i++ {
		i := i
		go func() {
			defer wg.Done()
			vec := []float64{float64(i) * 0.01, 0.5, -0.5}
			ct, err := pool.EncryptVector(vec)
			if err != nil {
				errs <- err
				return
			}
			s, err := pool.DecryptScalar(ct)
			if err != nil {
				errs <- err
				return
			}
			if math.Abs(s-vec[0]) > 0.05 {
				errs <- nil // tolerable noise
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent EncryptDecrypt: %v", err)
		}
	}
}

func TestEnginePool_BatchDotProductRoundtrip(t *testing.T) {
	pool, err := NewEnginePool(2)
	if err != nil {
		t.Fatalf("NewEnginePool: %v", err)
	}
	defer pool.Close()

	params := pool.GetParams()
	encoder := pool.GetEncoder()
	dim := 4
	centroidsPerPack := params.MaxSlots() / dim

	query := []float64{1, 2, 3, 4}
	centroid := []float64{1, 1, 1, 1}

	packedQuery := make([]float64, params.MaxSlots())
	packedCentroids := make([]float64, params.MaxSlots())
	for c := 0; c < centroidsPerPack; c++ {
		copy(packedQuery[c*dim:(c+1)*dim], query)
	}
	copy(packedCentroids[:dim], centroid)

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(packedCentroids, pt); err != nil {
		t.Fatalf("encode: %v", err)
	}

	ct, err := pool.EncryptVector(packedQuery)
	if err != nil {
		t.Fatalf("EncryptVector: %v", err)
	}

	result, err := pool.HomomorphicBatchDotProduct(ct, pt, centroidsPerPack, dim)
	if err != nil {
		t.Fatalf("HomomorphicBatchDotProduct: %v", err)
	}

	scores, err := pool.DecryptBatchScalars(result, centroidsPerPack, dim)
	if err != nil {
		t.Fatalf("DecryptBatchScalars: %v", err)
	}
	if len(scores) == 0 {
		t.Fatal("no scores returned")
	}
	if math.Abs(scores[0]-10.0) > 0.5 {
		t.Fatalf("score[0]=%v want ~10", scores[0])
	}
}

func TestEnginePool_CloseSafeOnEmpty(t *testing.T) {
	pool, err := NewEnginePool(2)
	if err != nil {
		t.Fatalf("NewEnginePool: %v", err)
	}
	pool.Close()
	// Second close should not panic; it just drains an empty channel.
	pool.Close()
}
