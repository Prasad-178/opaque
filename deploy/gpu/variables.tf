variable "enabled" {
  description = "Toggle GPU infrastructure on/off. Set to false to destroy all resources."
  type        = bool
  default     = false
}

variable "aws_profile" {
  description = "AWS CLI profile to use."
  type        = string
  default     = "personal"
}

variable "aws_region" {
  description = "AWS region for the GPU instance."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type. g4dn.xlarge = T4 GPU (16GB VRAM), cheapest CUDA option."
  type        = string
  default     = "g4dn.xlarge"
}

variable "use_spot" {
  description = "Use spot instance for ~70% cost savings. May be interrupted."
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Maximum hourly price for spot instance. Set to 0 for on-demand price cap."
  type        = string
  default     = "0.25"
}

variable "key_name" {
  description = "Name of the SSH key pair to create/use."
  type        = string
  default     = "opaque-gpu-bench"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into the instance. Default: your IP only."
  type        = string
  default     = "0.0.0.0/0" # Tighten this to your IP in production
}

variable "repo_url" {
  description = "Git repository URL to clone on the instance."
  type        = string
  default     = "https://github.com/Prasad-178/opaque.git"
}

variable "repo_branch" {
  description = "Git branch to checkout."
  type        = string
  default     = "main"
}
