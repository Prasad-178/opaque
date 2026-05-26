.PHONY: build fmt fmt-check vet lint staticcheck govulncheck test-unit test-integration test-fast test test-race test-cover test-bench test-sift test-100k test-sift1m proto ci

build:
	go build ./...
	@for d in cmd/*/; do go build -o /tmp/$$(basename "$$d") "./$$d"; done
	@for d in examples/*/; do go build -o /tmp/example-$$(basename "$$d") "./$$d"; done

fmt:
	gofmt -w .

fmt-check:
	@unformatted=$$(gofmt -l .); \
	if [ -n "$$unformatted" ]; then \
		echo "Not gofmt-clean:"; echo "$$unformatted"; exit 1; \
	fi

vet:
	go vet ./...

lint: fmt-check vet

staticcheck:
	go run honnef.co/go/tools/cmd/staticcheck@2025.1.1 ./...

govulncheck:
	go run golang.org/x/vuln/cmd/govulncheck@latest ./...

test-unit:
	go test -short -timeout 10m ./...

test-fast: test-unit

test-integration:
	go test -tags integration -short -timeout 30m ./...

test:
	go test -tags integration ./...

test-race:
	go test -race -short -timeout 20m \
	  ./pkg/crypto/... \
	  ./pkg/crypto/threshold/... \
	  ./pkg/enterprise/... \
	  ./pkg/blob/... \
	  ./pkg/cache/... \
	  ./pkg/grpcserver/... \
	  ./internal/store/... \
	  ./internal/session/...

test-cover:
	go test -short -timeout 10m -coverprofile=coverage.out -coverpkg=./... ./...
	go tool cover -func=coverage.out | tail -30

test-bench:
	go test -bench=. -benchmem ./pkg/crypto/... ./pkg/lsh/...

test-sift:
	go test -tags integration -v -run TestSIFTKMeansEndToEnd ./pkg/client/ -timeout 5m

test-100k:
	go test -tags integration -v -run TestBenchmark100KOptimized ./pkg/client/ -timeout 15m

test-sift1m:
	go test -tags sift1m -v -run "TestSIFT1M" ./test/ -timeout 45m

# `make ci` runs the same gates as the CI workflow, locally.
ci: lint build test-unit test-race

proto:
	protoc --go_out=. --go_opt=paths=source_relative \
	       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
	       api/proto/opaque.proto
