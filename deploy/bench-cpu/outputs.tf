output "instance_ip" {
  value = var.enabled ? aws_instance.bench[0].public_ip : "disabled"
}

output "instance_type" {
  value = var.instance_type
}

output "ssh_command" {
  value = var.enabled ? "ssh -i deploy/bench-cpu/bench-cpu-key.pem ubuntu@${aws_instance.bench[0].public_ip}" : "disabled"
}

output "estimated_cost" {
  value = var.instance_type == "c6i.4xlarge" ? "~$0.68/hr on-demand" : "~$0.34/hr on-demand"
}
