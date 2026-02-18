.PHONY: test-fast test test-bench test-sift test-100k lint build

build:
	go build ./...

lint:
	go vet ./...

test-fast:
	go test -short ./...

test:
	go test ./...

test-bench:
	go test -bench=. -benchmem ./pkg/crypto/... ./pkg/lsh/...

test-sift:
	go test -v -run TestSIFTKMeansEndToEnd ./pkg/client/ -timeout 5m

test-100k:
	go test -v -run TestBenchmark100KOptimized ./pkg/client/ -timeout 15m
