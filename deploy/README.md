# VYREX AI - Production Deployment & Cloud Infrastructure Guide

This guide details how to build, containerize, and deploy **VYREX AI** across local Docker environments and production cloud platforms (Modal, RunPod, Kubernetes, and AWS SageMaker).

---

## 1. Local Containerized Serving (Docker & Docker Compose)

### Prerequisites:
- Docker with NVIDIA Container Toolkit installed (`nvidia-container-toolkit`)
- NVIDIA GPU with >= 6GB VRAM

### Build & Run with Docker Compose:
```bash
# Build and run with GPU reservation
docker compose up --build -d

# Check live logs
docker compose logs -f

# Verify health status
curl http://localhost:8000/v1/health
```

### Manual Docker Build & Run:
```bash
# Build image
docker build -t vyrex-ai:latest .

# Run container with GPU passthrough
docker run -d --gpus all \
  -p 8000:8000 \
  -e VYREX_BACKEND=hf \
  --name vyrex-api \
  vyrex-ai:latest
```

---

## 2. Serverless Cloud GPU Deployment (Modal)

Modal provides instant serverless GPU hosting that auto-scales down to zero when idle, making it cost-efficient.

### Deploy in 2 commands:
```bash
pip install modal
modal setup

# Deploy directly to cloud endpoint
modal deploy deploy/modal_app.py
```
Modal will output a public HTTPS URL (e.g., `https://your-user--vyrex-ai-production-fastapi-app.modal.run`) which provides full OpenAI-compatible `/v1/chat/completions` access!

---

## 3. Serverless GPU Deployment (RunPod)

RunPod Serverless allows running pay-per-millisecond workers on high-end GPUs (RTX 3090, 4090, A4000).

### Deploy to RunPod:
1. Build and push the Docker image to Docker Hub:
   ```bash
   docker build -t <your-dockerhub-username>/vyrex-runpod:latest -f Dockerfile .
   docker push <your-dockerhub-username>/vyrex-runpod:latest
   ```
2. In the **RunPod Console**, go to **Serverless** -> **Create Endpoint**.
3. Set your container image to `<your-dockerhub-username>/vyrex-runpod:latest`.
4. Set container start command to: `python deploy/runpod_handler.py`.

---

## 4. Kubernetes Deployment (AWS EKS / GCP GKE)

For high-availability enterprise environments:

```bash
# Apply deployment, service, and HPA
kubectl apply -f deploy/k8s-deployment.yaml

# Verify pods
kubectl get pods -l app=vyrex-ai

# Check auto-scaler
kubectl get hpa vyrex-ai-hpa
```

---

## 5. AWS SageMaker Real-Time Inference

To host on AWS SageMaker with dedicated GPU instances (`ml.g5.xlarge` with NVIDIA A10G):

```bash
pip install sagemaker
python deploy/sagemaker_deploy.py --role-arn arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole
```

---

## 6. High-Throughput vLLM Serving

To serve with vLLM's continuous batching and PagedAttention:

```bash
# Option A: Direct dynamic LoRA serving
python scripts/serve_vllm.py --port 8000

# Option B: Merge weights first for 0% overhead
python scripts/merge_lora_weights.py --output-dir pretrained/vyrex_1_7b_merged
vllm serve pretrained/vyrex_1_7b_merged --port 8000 --max-model-len 4096
```
