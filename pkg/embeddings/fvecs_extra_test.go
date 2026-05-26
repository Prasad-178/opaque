package embeddings

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadFvecs_InconsistentDimension(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, int32(3))
	binary.Write(&buf, binary.LittleEndian, []float32{1, 2, 3})
	binary.Write(&buf, binary.LittleEndian, int32(4)) // different dim
	binary.Write(&buf, binary.LittleEndian, []float32{4, 5, 6, 7})

	if _, err := ReadFvecs(&buf); err == nil {
		t.Fatal("expected error on inconsistent dim, got nil")
	} else if !strings.Contains(err.Error(), "inconsistent") {
		t.Fatalf("expected inconsistent-dim error, got %v", err)
	}
}

func TestReadFvecs_TruncatedTail(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, int32(4))
	binary.Write(&buf, binary.LittleEndian, []float32{1, 2, 3}) // short

	_, err := ReadFvecs(&buf)
	if err == nil {
		t.Fatal("expected error on truncated payload")
	}
	// io.EOF on header-only is handled separately by ReadFvecs (returns nil,
	// nil at EOF). A truncated payload mid-vector should surface as some
	// non-nil error — either io.ErrUnexpectedEOF or the wrapped binary.Read
	// error. We only assert the latter.
	_ = errors.Is(err, io.EOF)
}

func TestReadFvecsF32_RoundtripVsReadFvecs(t *testing.T) {
	original := [][]float64{
		{0.5, -1.5, 2.25, -3.125},
		{4.0, 5.0, 6.0, 7.0},
	}
	var buf bytes.Buffer
	if err := WriteFvecs(&buf, original); err != nil {
		t.Fatalf("WriteFvecs: %v", err)
	}

	// ReadFvecsF32 — native precision, no upcast.
	bufCopy := bytes.NewReader(buf.Bytes())
	gotF32, err := ReadFvecsF32(bufCopy)
	if err != nil {
		t.Fatalf("ReadFvecsF32: %v", err)
	}
	if len(gotF32) != len(original) {
		t.Fatalf("len=%d want=%d", len(gotF32), len(original))
	}
	for i, vec := range gotF32 {
		for j, v := range vec {
			if float64(v) != original[i][j] {
				t.Fatalf("vec[%d][%d] = %v want %v", i, j, v, original[i][j])
			}
		}
	}
}

func TestLoadFvecs_FileRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.fvecs")
	original := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}
	if err := SaveFvecs(path, original); err != nil {
		t.Fatalf("SaveFvecs: %v", err)
	}
	got, err := LoadFvecs(path)
	if err != nil {
		t.Fatalf("LoadFvecs: %v", err)
	}
	if len(got) != len(original) {
		t.Fatalf("len=%d want=%d", len(got), len(original))
	}
	for i := range got {
		for j := range got[i] {
			if got[i][j] != original[i][j] {
				t.Fatalf("vec[%d][%d] = %v want %v", i, j, got[i][j], original[i][j])
			}
		}
	}

	got32, err := LoadFvecsF32(path)
	if err != nil {
		t.Fatalf("LoadFvecsF32: %v", err)
	}
	if len(got32) != len(original) {
		t.Fatalf("F32 len=%d want=%d", len(got32), len(original))
	}
}

func TestLoadFvecs_MissingFile(t *testing.T) {
	if _, err := LoadFvecs(filepath.Join(t.TempDir(), "nope.fvecs")); err == nil {
		t.Fatal("expected error on missing file")
	}
	if _, err := LoadFvecsF32(filepath.Join(t.TempDir(), "nope.fvecs")); err == nil {
		t.Fatal("expected error on missing file (F32)")
	}
	if _, err := LoadIvecs(filepath.Join(t.TempDir(), "nope.ivecs")); err == nil {
		t.Fatal("expected error on missing file (ivecs)")
	}
	if _, err := LoadBvecs(filepath.Join(t.TempDir(), "nope.bvecs")); err == nil {
		t.Fatal("expected error on missing file (bvecs)")
	}
}

func TestReadIvecs_InconsistentDimension(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, int32(3))
	binary.Write(&buf, binary.LittleEndian, []int32{1, 2, 3})
	binary.Write(&buf, binary.LittleEndian, int32(2))
	binary.Write(&buf, binary.LittleEndian, []int32{4, 5})

	if _, err := ReadIvecs(&buf); err == nil {
		t.Fatal("expected inconsistent-dim error")
	}
}

func TestReadBvecs_Roundtrip(t *testing.T) {
	var buf bytes.Buffer
	values := [][]uint8{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
	}
	for _, vec := range values {
		binary.Write(&buf, binary.LittleEndian, int32(len(vec)))
		buf.Write(vec)
	}
	got, err := ReadBvecs(&buf)
	if err != nil {
		t.Fatalf("ReadBvecs: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len=%d want 2", len(got))
	}
	for i, vec := range got {
		for j, v := range vec {
			if int(v) != int(values[i][j]) {
				t.Fatalf("vec[%d][%d] = %v want %v", i, j, v, values[i][j])
			}
		}
	}
}

func TestReadBvecs_InconsistentDimension(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, int32(4))
	buf.Write([]byte{1, 2, 3, 4})
	binary.Write(&buf, binary.LittleEndian, int32(3))
	buf.Write([]byte{5, 6, 7})

	if _, err := ReadBvecs(&buf); err == nil {
		t.Fatal("expected inconsistent-dim error")
	}
}

func TestSaveFvecs_BadPath(t *testing.T) {
	if err := SaveFvecs(filepath.Join(t.TempDir(), "no-such-dir", "out.fvecs"), [][]float64{{1, 2}}); err == nil {
		t.Fatal("expected error writing to nonexistent directory")
	}
}

func TestReadFvecs_EmptyReader(t *testing.T) {
	got, err := ReadFvecs(bytes.NewReader(nil))
	if err != nil {
		t.Fatalf("ReadFvecs empty: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("len=%d want 0", len(got))
	}
}

// ensure WriteFvecs on a write-error-returning writer surfaces the error.
type errWriter struct{ failAfter int }

func (e *errWriter) Write(p []byte) (int, error) {
	if e.failAfter <= 0 {
		return 0, os.ErrClosed
	}
	n := len(p)
	if n > e.failAfter {
		n = e.failAfter
	}
	e.failAfter -= n
	if e.failAfter == 0 {
		return n, os.ErrClosed
	}
	return n, nil
}

func TestWriteFvecs_PropagatesError(t *testing.T) {
	w := &errWriter{failAfter: 0}
	if err := WriteFvecs(w, [][]float64{{1, 2, 3}}); err == nil {
		t.Fatal("expected error from failing writer")
	}
}
