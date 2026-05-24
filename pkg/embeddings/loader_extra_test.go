package embeddings

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeSIFTBase(t *testing.T, dir, basename string, vectors [][]float64) string {
	t.Helper()
	p := filepath.Join(dir, basename)
	if err := SaveFvecs(p, vectors); err != nil {
		t.Fatalf("SaveFvecs %s: %v", basename, err)
	}
	return p
}

func TestSIFT1M_MissingFiles(t *testing.T) {
	_, err := SIFT1M(t.TempDir())
	if err == nil {
		t.Fatal("expected error when sift_base.fvecs missing")
	}
}

func TestSIFT1M_BaseOnly(t *testing.T) {
	dir := t.TempDir()
	writeSIFTBase(t, dir, "sift_base.fvecs", [][]float64{{1, 2, 3}, {4, 5, 6}})

	ds, err := SIFT1M(dir)
	if err != nil {
		t.Fatalf("SIFT1M: %v", err)
	}
	if ds.Name != "sift1m" || ds.Dimension != 3 || len(ds.Vectors) != 2 || len(ds.Queries) != 0 || len(ds.GroundTruth) != 0 {
		t.Fatalf("unexpected dataset: %+v", ds.Stats())
	}
	if len(ds.IDs) != 2 || !strings.HasPrefix(ds.IDs[0], "sift_") {
		t.Fatalf("bad ids: %v", ds.IDs)
	}
}

func TestSIFT1M_AllArtifactsLoaded(t *testing.T) {
	dir := t.TempDir()
	writeSIFTBase(t, dir, "sift_base.fvecs", [][]float64{{1, 2}, {3, 4}})
	writeSIFTBase(t, dir, "sift_query.fvecs", [][]float64{{5, 6}})
	// Ground truth via ivecs is harder to forge with the public API — skip
	// writing it and confirm the SIFT1M tolerates its absence.

	ds, err := SIFT1M(dir)
	if err != nil {
		t.Fatalf("SIFT1M: %v", err)
	}
	if len(ds.Queries) != 1 {
		t.Fatalf("queries len=%d want 1", len(ds.Queries))
	}
}

func TestSIFT10K_MissingFiles(t *testing.T) {
	_, err := SIFT10K(t.TempDir())
	if err == nil {
		t.Fatal("expected error when siftsmall_base.fvecs missing")
	}
}

func TestSIFT10K_BaseOnly(t *testing.T) {
	dir := t.TempDir()
	writeSIFTBase(t, dir, "siftsmall_base.fvecs", [][]float64{{1, 2, 3}})
	ds, err := SIFT10K(dir)
	if err != nil {
		t.Fatalf("SIFT10K: %v", err)
	}
	if ds.Name != "sift10k" || ds.Dimension != 3 {
		t.Fatalf("unexpected dataset: %+v", ds.Stats())
	}
}

func TestGIST1M_MissingFiles(t *testing.T) {
	_, err := GIST1M(t.TempDir())
	if err == nil {
		t.Fatal("expected error when gist_base.fvecs missing")
	}
}

func TestGIST1M_BaseOnly(t *testing.T) {
	dir := t.TempDir()
	writeSIFTBase(t, dir, "gist_base.fvecs", [][]float64{{1, 2}})
	ds, err := GIST1M(dir)
	if err != nil {
		t.Fatalf("GIST1M: %v", err)
	}
	if ds.Name != "gist1m" {
		t.Fatalf("name=%s want gist1m", ds.Name)
	}
}

func TestSIFT10M_MissingFiles(t *testing.T) {
	_, err := SIFT10M(t.TempDir())
	if err == nil {
		t.Fatal("expected error when sift10m_base.bvecs missing")
	}
}

func TestDBpedia1M_MissingFiles(t *testing.T) {
	_, err := DBpedia1M(t.TempDir())
	if err == nil {
		t.Fatal("expected error when dbpedia_base.fvecs missing")
	}
	if _, _, _, err := DBpedia1MF32(t.TempDir()); err == nil {
		t.Fatal("expected error from DBpedia1MF32 when base missing")
	}
	if _, err := DBpedia1MQueries(t.TempDir()); err == nil {
		t.Fatal("expected error when dbpedia_query.fvecs missing")
	}
}

func TestDBpedia1M_BaseLoaded(t *testing.T) {
	dir := t.TempDir()
	writeSIFTBase(t, dir, "dbpedia_base.fvecs", [][]float64{{1, 2}, {3, 4}})

	ds, err := DBpedia1M(dir)
	if err != nil {
		t.Fatalf("DBpedia1M: %v", err)
	}
	if ds.Name != "dbpedia1m" || len(ds.Vectors) != 2 || len(ds.IDs) != 2 {
		t.Fatalf("unexpected: %+v", ds.Stats())
	}

	vecs, ids, dim, err := DBpedia1MF32(dir)
	if err != nil {
		t.Fatalf("DBpedia1MF32: %v", err)
	}
	if len(vecs) != 2 || len(ids) != 2 || dim != 2 {
		t.Fatalf("F32 mismatch: len=%d ids=%d dim=%d", len(vecs), len(ids), dim)
	}

	// queries loader requires the query file specifically.
	if _, err := DBpedia1MQueries(dir); err == nil {
		t.Fatal("expected queries-missing error")
	}
	writeSIFTBase(t, dir, "dbpedia_query.fvecs", [][]float64{{1, 1}})
	qs, err := DBpedia1MQueries(dir)
	if err != nil {
		t.Fatalf("DBpedia1MQueries: %v", err)
	}
	if len(qs) != 1 {
		t.Fatalf("queries len=%d want 1", len(qs))
	}
}

func TestFromFvecs(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "data.fvecs")
	if err := SaveFvecs(p, [][]float64{{1, 2}, {3, 4}, {5, 6}}); err != nil {
		t.Fatalf("SaveFvecs: %v", err)
	}
	ds, err := FromFvecs(p, "custom")
	if err != nil {
		t.Fatalf("FromFvecs: %v", err)
	}
	if ds.Name != "custom" || ds.Dimension != 2 || len(ds.Vectors) != 3 {
		t.Fatalf("unexpected: %+v", ds.Stats())
	}
	if ds.IDs[1] != "custom_1" {
		t.Fatalf("ids[1]=%s want custom_1", ds.IDs[1])
	}
}

func TestFromFvecs_Missing(t *testing.T) {
	if _, err := FromFvecs(filepath.Join(t.TempDir(), "nope.fvecs"), "x"); err == nil {
		t.Fatal("expected error on missing file")
	}
}

func TestGloVe_Missing(t *testing.T) {
	if _, err := GloVe(filepath.Join(t.TempDir(), "missing.txt")); err == nil {
		t.Fatal("expected error on missing GloVe file")
	}
}

func TestGloVe_HappyPath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "glove.txt")
	body := "the 0.1 0.2 0.3\nand 0.4 0.5 0.6\n\ncat 0.7 0.8 0.9\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	ds, err := GloVe(path)
	if err != nil {
		t.Fatalf("GloVe: %v", err)
	}
	if ds.Dimension != 3 {
		t.Fatalf("dim=%d want 3", ds.Dimension)
	}
	if len(ds.Vectors) != 3 || len(ds.IDs) != 3 {
		t.Fatalf("vectors=%d ids=%d want 3/3", len(ds.Vectors), len(ds.IDs))
	}
	if ds.IDs[0] != "the" || ds.IDs[2] != "cat" {
		t.Fatalf("words wrong: %v", ds.IDs)
	}
}

func TestGloVe_InconsistentDimension(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "glove.txt")
	body := "the 0.1 0.2 0.3\nand 0.4 0.5\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := GloVe(path); err == nil {
		t.Fatal("expected inconsistent-dim error")
	}
}

func TestGloVe_OnlyWord(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "glove.txt")
	if err := os.WriteFile(path, []byte("solo\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := GloVe(path); err == nil {
		t.Fatal("expected error for missing dims")
	}
}

func TestGloVe_BadNumber(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "glove.txt")
	if err := os.WriteFile(path, []byte("a 0.1 NaNXYZ 0.3\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := GloVe(path); err == nil {
		t.Fatal("expected parse error")
	}
}

func TestGloVe_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "glove.txt")
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := GloVe(path); err == nil {
		t.Fatal("expected error on empty file")
	}
}

func TestDataset_SubsetLargerThanN(t *testing.T) {
	d := Generate(10, 4, 1)
	sub := d.Subset(50)
	if len(sub.Vectors) != 10 {
		t.Fatalf("len=%d want 10 (clamped)", len(sub.Vectors))
	}
}

func TestDataset_StatsNoGT(t *testing.T) {
	d := Generate(5, 3, 1)
	st := d.Stats()
	if st.HasGroundTruth {
		t.Fatal("HasGroundTruth=true with no GT")
	}
	if st.GroundTruthDepth != 0 {
		t.Fatalf("depth=%d want 0", st.GroundTruthDepth)
	}
}
