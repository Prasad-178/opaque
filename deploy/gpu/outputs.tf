output "instance_ip" {
  description = "Public IP of the GPU instance."
  value = var.enabled ? (
    var.use_spot
    ? try(aws_spot_instance_request.gpu_bench[0].public_ip, "pending...")
    : try(aws_instance.gpu_bench[0].public_ip, "pending...")
  ) : "disabled"
}

output "instance_id" {
  description = "EC2 instance ID."
  value = var.enabled ? (
    var.use_spot
    ? try(aws_spot_instance_request.gpu_bench[0].spot_instance_id, "pending...")
    : try(aws_instance.gpu_bench[0].id, "pending...")
  ) : "disabled"
}

output "ssh_command" {
  description = "SSH command to connect to the instance."
  value = var.enabled ? (
    var.use_spot
    ? "ssh -i deploy/gpu/gpu-bench-key.pem ubuntu@${try(aws_spot_instance_request.gpu_bench[0].public_ip, "PENDING")}"
    : "ssh -i deploy/gpu/gpu-bench-key.pem ubuntu@${try(aws_instance.gpu_bench[0].public_ip, "PENDING")}"
  ) : "disabled"
}

output "run_benchmarks_command" {
  description = "Command to run GPU benchmarks on the instance."
  value = var.enabled ? "bash deploy/gpu/run_benchmarks.sh" : "disabled"
}

output "ami_id" {
  description = "Deep Learning AMI used."
  value       = data.aws_ami.deep_learning.id
}

output "ami_name" {
  description = "Deep Learning AMI name."
  value       = data.aws_ami.deep_learning.name
}

output "estimated_cost" {
  description = "Estimated hourly cost."
  value       = var.use_spot ? "~$0.16/hr (spot)" : "~$0.53/hr (on-demand)"
}
