variable "enabled" {
  description = "Toggle bench infrastructure on/off. Always destroy when done."
  type        = bool
  default     = false
}

variable "aws_profile" {
  description = "AWS CLI profile. Personal account only — do not use realfy."
  type        = string
  default     = "personal"
}

variable "aws_region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type. c6i.2xlarge = 8 vCPU, c6i.4xlarge = 16 vCPU."
  type        = string
  default     = "c6i.2xlarge"
}

variable "key_name" {
  description = "Name of the SSH key pair to create."
  type        = string
  default     = "opaque-bench-cpu"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH. Default 0.0.0.0/0 — tighten for prod."
  type        = string
  default     = "0.0.0.0/0"
}
