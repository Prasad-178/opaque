.PHONY: test-unit test-integration test-fast test test-bench test-sift test-100k lint build

build:
	go build ./...

lint:
	go vet ./...

test-unit:
	go test -short ./...

test-integration:
	go test -tags integration -timeout 20m ./...

test-fast:
	go test -short ./...

test:
	go test -tags integration ./...

test-bench:
	go test -bench=. -benchmem ./pkg/crypto/... ./pkg/lsh/...

test-sift:
	go test -tags integration -v -run TestSIFTKMeansEndToEnd ./pkg/client/ -timeout 5m

test-100k:
	go test -tags integration -v -run TestBenchmark100KOptimized ./pkg/client/ -timeout 15m
