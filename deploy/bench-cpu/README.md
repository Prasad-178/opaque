# Opaque CPU benchmarks on AWS

Ephemeral infrastructure for running the SIFT 1M accuracy + PQ benchmarks on
commodity EC2 instances. Every run tears down after finishing; there is no
persistent infrastructure.

**Target sizes** (matches production vector DB pod ranges):

| Instance     | vCPU | RAM   | Cost        | Matches |
|--------------|------|-------|-------------|---------|
| c6i.2xlarge  | 8    | 16 GB | ~$0.34/hr   | Pinecone p1.x1, Qdrant 8-core |
| c6i.4xlarge  | 16   | 32 GB | ~$0.68/hr   | Qdrant/Weaviate standard pod  |

Always uses the **personal** AWS profile. The module refuses to apply
otherwise — do not change the default.

## Run

```bash
# 8-vCPU run (Pinecone-sized)
bash deploy/bench-cpu/run_bench.sh c6i.2xlarge

# 16-vCPU run (Qdrant-sized)
bash deploy/bench-cpu/run_bench.sh c6i.4xlarge
```

The script bundles the opaque source, brings up one EC2 instance, installs
Go, downloads SIFT 1M, runs `TestSIFT1MAccuracy` + `TestPQ_SIFT1M` with the
`sift1m` build tag, pulls logs into `results/<timestamp>-<instance>/`, and
then destroys the instance. A `trap` on EXIT ensures destruction even on
Ctrl-C or script failure.

Typical cost: ~$0.35 per run.
