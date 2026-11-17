"""
AWS SageMaker Deployment Script for VYREX AI.
Deploys the fine-tuned model to an AWS SageMaker Real-Time Inference Endpoint
using an ml.g5.xlarge (NVIDIA A10G) or ml.g4dn.xlarge (NVIDIA T4) instance.

Usage:
    python deploy/sagemaker_deploy.py --role-arn arn:aws:iam::123456789012:role/SageMakerRole
"""

import argparse
import sys


def deploy_to_sagemaker(role_arn: str, instance_type: str = "ml.g5.xlarge"):
    try:
        import sagemaker
        from sagemaker.huggingface import HuggingFaceModel
    except ImportError:
        print("[!] Error: 'sagemaker' SDK is not installed.")
        print("    Install it with: pip install sagemaker")
        sys.exit(1)

    print("=" * 65)
    print("       VYREX AI: AWS SAGEMAKER REAL-TIME ENDPOINT DEPLOYMENT")
    print("=" * 65)
    print(f"Target Instance: {instance_type}")
    print(f"Role ARN:        {role_arn}")

    # SageMaker HuggingFace Model Configuration
    hub_config = {
        "HF_MODEL_ID": "Qwen/Qwen2.5-1.5B-Instruct",
        "HF_TASK": "text-generation",
    }

    huggingface_model = HuggingFaceModel(
        env=hub_config,
        role=role_arn,
        transformers_version="4.37.0",
        pytorch_version="2.1.0",
        py_version="py310",
    )

    print("[*] Deploying real-time endpoint (this may take 5-8 minutes)...")
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name="vyrex-ai-production-endpoint",
    )

    print("\n[SUCCESS] Endpoint successfully deployed!")
    print(f"Endpoint Name: {predictor.endpoint_name}")


def main():
    parser = argparse.ArgumentParser(description="Deploy VYREX AI to AWS SageMaker")
    parser.add_argument("--role-arn", type=str, required=True, help="AWS IAM Role ARN for SageMaker")
    parser.add_argument("--instance-type", type=str, default="ml.g5.xlarge")
    args = parser.parse_args()

    deploy_to_sagemaker(args.role_arn, args.instance_type)


if __name__ == "__main__":
    main()
