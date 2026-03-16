.PHONY: test-unit test-integration test-fast test test-bench test-sift test-100k test-sift1m lint build proto

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

test-sift1m:
	go test -tags sift1m -v -run "TestSIFT1M" ./test/ -timeout 45m

proto:
	protoc --go_out=. --go_opt=paths=source_relative \
	       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
	       api/proto/opaque.proto
