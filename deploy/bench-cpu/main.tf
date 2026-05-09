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

# Ubuntu 22.04 LTS x86_64 — Canonical's latest published AMI.
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "state"
    values = ["available"]
  }
}

resource "tls_private_key" "ssh" {
  count     = var.enabled ? 1 : 0
  algorithm = "ED25519"
}

resource "aws_key_pair" "bench" {
  count      = var.enabled ? 1 : 0
  key_name   = var.key_name
  public_key = tls_private_key.ssh[0].public_key_openssh
}

resource "local_file" "ssh_key" {
  count           = var.enabled ? 1 : 0
  content         = tls_private_key.ssh[0].private_key_openssh
  filename        = "${path.module}/bench-cpu-key.pem"
  file_permission = "0600"
}

resource "aws_security_group" "bench" {
  count       = var.enabled ? 1 : 0
  name        = "opaque-bench-cpu"
  description = "SSH for Opaque CPU benchmarks"

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
    Name    = "opaque-bench-cpu"
    Project = "opaque"
  }
}

resource "aws_instance" "bench" {
  count         = var.enabled ? 1 : 0
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  key_name      = aws_key_pair.bench[0].key_name

  vpc_security_group_ids = [aws_security_group.bench[0].id]

  root_block_device {
    volume_size = 50 # SIFT1M ~600MB, DBpedia1M ~12GB peak (parquet+fvecs), plus Go toolchain. Same disk works for both bench paths.
    volume_type = "gp3"
  }

  tags = {
    Name         = "opaque-bench-cpu-${var.instance_type}"
    Project      = "opaque"
    InstanceType = var.instance_type
  }
}
