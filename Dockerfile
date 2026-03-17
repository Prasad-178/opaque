FROM golang:1.25-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /opaque-server ./cmd/search-service

FROM alpine:3.19
COPY --from=builder /opaque-server /usr/local/bin/opaque-server
EXPOSE 50051
ENTRYPOINT ["opaque-server"]
