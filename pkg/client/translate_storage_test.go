package client

import (
	"reflect"
	"testing"
)

func TestTranslateToStorage_NilPermutation(t *testing.T) {
	logical := []int{0, 5, 12, 127}
	got := translateToStorage(logical, nil)
	if !reflect.DeepEqual(got, logical) {
		t.Fatalf("nil permutation should be identity; got %v want %v", got, logical)
	}
	// Verify input is not mutated, and result is a fresh slice.
	if &got[0] == &logical[0] {
		t.Fatalf("expected fresh slice, got aliased reference")
	}
}

func TestTranslateToStorage_AppliesPermutation(t *testing.T) {
	// π[0]=3, π[1]=0, π[2]=2, π[3]=1
	π := []int{3, 0, 2, 1}
	logical := []int{0, 1, 2, 3}
	got := translateToStorage(logical, π)
	want := []int{3, 0, 2, 1}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestTranslateToStorage_OutOfRange(t *testing.T) {
	π := []int{2, 0, 1}
	logical := []int{0, 5, 1, -1, 2}
	got := translateToStorage(logical, π)
	want := []int{2, 5, 0, -1, 1} // out-of-range pass through
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestTranslateToStorage_DoesNotMutateInput(t *testing.T) {
	π := []int{1, 0}
	logical := []int{0, 1}
	original := []int{0, 1}
	_ = translateToStorage(logical, π)
	if !reflect.DeepEqual(logical, original) {
		t.Fatalf("input mutated: got %v want %v", logical, original)
	}
}

func TestTranslateToStorage_EmptyInput(t *testing.T) {
	π := []int{0, 1, 2}
	got := translateToStorage([]int{}, π)
	if len(got) != 0 {
		t.Fatalf("expected empty slice, got %v", got)
	}
}
