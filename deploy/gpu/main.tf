terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
}

# --- Data Sources ---

# Find the latest NVIDIA Deep Learning AMI (CUDA pre-installed).
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# --- SSH Key Pair ---

resource "tls_private_key" "ssh" {
  count     = var.enabled ? 1 : 0
  algorithm = "ED25519"
}

resource "aws_key_pair" "gpu_bench" {
  count      = var.enabled ? 1 : 0
  key_name   = var.key_name
  public_key = tls_private_key.ssh[0].public_key_openssh
}

resource "local_file" "ssh_key" {
  count           = var.enabled ? 1 : 0
  content         = tls_private_key.ssh[0].private_key_openssh
  filename        = "${path.module}/gpu-bench-key.pem"
  file_permission = "0600"
}

# --- Security Group ---

resource "aws_security_group" "gpu_bench" {
  count       = var.enabled ? 1 : 0
  name        = "opaque-gpu-bench"
  description = "SSH access for Opaque GPU benchmarks"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "opaque-gpu-bench"
    Project = "opaque"
  }
}

# --- GPU Instance ---

resource "aws_instance" "gpu_bench" {
  count         = var.enabled && !var.use_spot ? 1 : 0
  ami           = data.aws_ami.deep_learning.id
  instance_type = var.instance_type
  key_name      = aws_key_pair.gpu_bench[0].key_name

  vpc_security_group_ids = [aws_security_group.gpu_bench[0].id]

  root_block_device {
    volume_size = 80
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/setup.sh", {
    repo_url    = var.repo_url
    repo_branch = var.repo_branch
  })

  tags = {
    Name    = "opaque-gpu-bench"
    Project = "opaque"
  }
}

# --- Spot Instance (default, cheaper) ---

resource "aws_spot_instance_request" "gpu_bench" {
  count                = var.enabled && var.use_spot ? 1 : 0
  ami                  = data.aws_ami.deep_learning.id
  instance_type        = var.instance_type
  key_name             = aws_key_pair.gpu_bench[0].key_name
  spot_price           = var.spot_max_price
  wait_for_fulfillment = true
  spot_type            = "one-time"

  vpc_security_group_ids = [aws_security_group.gpu_bench[0].id]

  root_block_device {
    volume_size = 80
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/setup.sh", {
    repo_url    = var.repo_url
    repo_branch = var.repo_branch
  })

  tags = {
    Name    = "opaque-gpu-bench"
    Project = "opaque"
  }
}

# --- Outputs are in outputs.tf ---
