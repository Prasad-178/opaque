# GPU Benchmark Infrastructure

Ephemeral GPU infrastructure for benchmarking HE operations on CUDA hardware.

## Quick Start

```bash
cd deploy/gpu

# Turn ON — provisions a g4dn.xlarge spot instance (~$0.16/hr)
terraform init
terraform apply -var="enabled=true"

# Wait ~5 min for setup, then run benchmarks
bash run_benchmarks.sh

# Turn OFF — destroys everything, $0 cost
terraform destroy -var="enabled=true"
```

## How It Works

1. **`terraform apply`** provisions:
   - g4dn.xlarge spot instance (NVIDIA T4 GPU, 16GB VRAM)
   - AWS Deep Learning AMI (CUDA + NVIDIA drivers pre-installed)
   - Security group (SSH only)
   - SSH key pair (auto-generated, saved locally)

2. **`setup.sh`** runs on boot:
   - Installs Go
   - Clones and builds Opaque
   - Clones and builds GPU-NTT (NTT microbenchmark)
   - Clones and builds HEonGPU (CKKS GPU library)

3. **`run_benchmarks.sh`** SSHs in and runs:
   - GPU info (nvidia-smi)
   - Opaque HE CPU benchmarks (baseline on GPU instance)
   - GPU-NTT benchmark (GPU vs CPU NTT comparison)
   - HEonGPU CKKS benchmark (full GPU CKKS operations)
   - Saves results to `results/`

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `enabled` | `false` | Toggle infra on/off |
| `aws_profile` | `personal` | AWS CLI profile |
| `aws_region` | `us-east-1` | AWS region |
| `instance_type` | `g4dn.xlarge` | EC2 instance type |
| `use_spot` | `true` | Use spot for ~70% savings |
| `spot_max_price` | `0.25` | Max spot price ($/hr) |

## Cost

| Mode | Cost/hr | Typical Run (30 min) |
|------|---------|---------------------|
| Spot (default) | ~$0.16 | ~$0.08 |
| On-demand | ~$0.53 | ~$0.27 |

Resources are fully destroyed on `terraform destroy` — zero cost when off.

## Manual SSH

```bash
ssh -i deploy/gpu/gpu-bench-key.pem ubuntu@$(terraform -chdir=deploy/gpu output -raw instance_ip)
```

## Files

```
deploy/gpu/
├── main.tf              # Instance, security group, key pair
├── variables.tf         # Configuration variables
├── outputs.tf           # Instance IP, SSH command, cost estimate
├── setup.sh             # Boot script: install Go, CUDA libs, build
├── run_benchmarks.sh    # Benchmark runner (SSH + collect results)
├── .gitignore           # Exclude state, keys, results
└── README.md            # This file
```
