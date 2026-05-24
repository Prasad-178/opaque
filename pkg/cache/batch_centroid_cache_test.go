package cache

import (
	"testing"

	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func testBatchParams(t *testing.T) hefloat.Parameters {
	t.Helper()
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,
		LogQ:            []int{60, 45, 45, 45, 45, 45, 45, 45},
		LogP:            []int{61, 61},
		LogDefaultScale: 45,
	})
	if err != nil {
		t.Fatalf("params: %v", err)
	}
	return params
}

func makeCentroids(n, dim int) [][]float64 {
	out := make([][]float64, n)
	for i := range out {
		row := make([]float64, dim)
		for j := range row {
			row[j] = float64(i*1000+j) * 1e-4
		}
		out[i] = row
	}
	return out
}

func TestBatchCentroidCache_PackingArithmetic(t *testing.T) {
	params := testBatchParams(t)
	encoder := hefloat.NewEncoder(params)

	dim := 128
	cache := NewBatchCentroidCache(params, encoder, dim)

	maxSlots := params.MaxSlots()
	wantPerPack := maxSlots / dim
	if cache.GetCentroidsPerPack() != wantPerPack {
		t.Fatalf("CentroidsPerPack=%d want=%d", cache.GetCentroidsPerPack(), wantPerPack)
	}
	if cache.GetDimension() != dim {
		t.Fatalf("Dimension=%d want=%d", cache.GetDimension(), dim)
	}
	if cache.Size() != 0 || cache.GetNumPacks() != 0 {
		t.Fatalf("fresh cache should be empty")
	}
}

func TestBatchCentroidCache_LoadAndPackQuery(t *testing.T) {
	params := testBatchParams(t)
	encoder := hefloat.NewEncoder(params)
	dim := 128
	cache := NewBatchCentroidCache(params, encoder, dim)
	perPack := cache.GetCentroidsPerPack()

	// Two full packs plus one half-pack to exercise tail handling.
	numCentroids := 2*perPack + perPack/2
	centroids := makeCentroids(numCentroids, dim)

	if err := cache.LoadCentroids(centroids, params.MaxLevel()); err != nil {
		t.Fatalf("LoadCentroids: %v", err)
	}

	wantPacks := (numCentroids + perPack - 1) / perPack
	if cache.GetNumPacks() != wantPacks {
		t.Fatalf("NumPacks=%d want=%d", cache.GetNumPacks(), wantPacks)
	}
	if cache.Size() != wantPacks {
		t.Fatalf("Size=%d want=%d", cache.Size(), wantPacks)
	}
	if cache.GetNumCentroids() != numCentroids {
		t.Fatalf("NumCentroids=%d want=%d", cache.GetNumCentroids(), numCentroids)
	}

	pts := cache.GetPackedPlaintexts()
	if len(pts) != wantPacks {
		t.Fatalf("GetPackedPlaintexts len=%d want=%d", len(pts), wantPacks)
	}
	for i, pt := range pts {
		if pt == nil {
			t.Fatalf("pack %d is nil", i)
		}
	}

	// PackQuery layout: query replicated centroidsPerPack times.
	query := make([]float64, dim)
	for i := range query {
		query[i] = float64(i + 1)
	}
	packed := cache.PackQuery(query)
	if len(packed) != params.MaxSlots() {
		t.Fatalf("packed len=%d want=%d", len(packed), params.MaxSlots())
	}
	for c := 0; c < perPack; c++ {
		offset := c * dim
		for j := 0; j < dim; j++ {
			if packed[offset+j] != query[j] {
				t.Fatalf("replica %d slot %d = %v want %v", c, j, packed[offset+j], query[j])
			}
		}
	}
}

func TestBatchCentroidCache_LoadEmpty(t *testing.T) {
	params := testBatchParams(t)
	encoder := hefloat.NewEncoder(params)
	cache := NewBatchCentroidCache(params, encoder, 128)

	if err := cache.LoadCentroids(nil, params.MaxLevel()); err != nil {
		t.Fatalf("LoadCentroids(nil): %v", err)
	}
	if cache.Size() != 0 || cache.GetNumCentroids() != 0 {
		t.Fatalf("empty load should leave cache empty")
	}
}

func TestBatchCentroidCache_ClearResets(t *testing.T) {
	params := testBatchParams(t)
	encoder := hefloat.NewEncoder(params)
	cache := NewBatchCentroidCache(params, encoder, 128)

	centroids := makeCentroids(8, 128)
	if err := cache.LoadCentroids(centroids, params.MaxLevel()); err != nil {
		t.Fatalf("LoadCentroids: %v", err)
	}
	if cache.Size() == 0 {
		t.Fatalf("expected populated cache")
	}
	cache.Clear()
	if cache.Size() != 0 {
		t.Fatalf("Size after Clear=%d want 0", cache.Size())
	}
	if cache.GetNumCentroids() != 0 {
		t.Fatalf("NumCentroids after Clear=%d want 0", cache.GetNumCentroids())
	}
}

func TestBatchCentroidCache_ShortCentroidPadsZero(t *testing.T) {
	params := testBatchParams(t)
	encoder := hefloat.NewEncoder(params)
	dim := 128
	cache := NewBatchCentroidCache(params, encoder, dim)

	// Mix of short, exact, and oversized vectors — must not panic and must
	// truncate / pad to dim.
	centroids := [][]float64{
		make([]float64, dim/2), // short → tail zeros
		make([]float64, dim),   // exact
		make([]float64, dim*2), // longer → truncated to dim
	}
	if err := cache.LoadCentroids(centroids, params.MaxLevel()); err != nil {
		t.Fatalf("LoadCentroids: %v", err)
	}
	if cache.GetNumCentroids() != 3 {
		t.Fatalf("NumCentroids=%d want 3", cache.GetNumCentroids())
	}
}

func TestBatchCentroidCache_PackQueryHandlesShortQuery(t *testing.T) {
	params := testBatchParams(t)
	encoder := hefloat.NewEncoder(params)
	dim := 128
	cache := NewBatchCentroidCache(params, encoder, dim)

	short := []float64{1, 2, 3}
	packed := cache.PackQuery(short)
	if len(packed) != params.MaxSlots() {
		t.Fatalf("packed len=%d want=%d", len(packed), params.MaxSlots())
	}
	for c := 0; c < cache.GetCentroidsPerPack(); c++ {
		for j := 0; j < len(short); j++ {
			if packed[c*dim+j] != short[j] {
				t.Fatalf("replica %d slot %d = %v want %v", c, j, packed[c*dim+j], short[j])
			}
		}
		// Remaining slots in this replica must be zero.
		for j := len(short); j < dim; j++ {
			if packed[c*dim+j] != 0 {
				t.Fatalf("replica %d slot %d not zero: %v", c, j, packed[c*dim+j])
			}
		}
	}
}
