<p align="center">
  <img src="docs/banner.png" alt="VYREX AI Banner" width="100%">
</p>

# VYREX AI — Vyrex Hyperion 3

<p align="center">
  <img src="https://img.shields.io/badge/Model-Vyrex_Hyperion_3-red" alt="Model Name">
  <img src="https://img.shields.io/badge/Base_Model-Qwen3--1.7B-blue" alt="Base Model">
  <img src="https://img.shields.io/badge/Fine--Tuning-QLoRA_V3-purple" alt="Fine-Tuning">
  <img src="https://img.shields.io/badge/Quantization-4--bit_NF4-green" alt="Quantization">
  <img src="https://img.shields.io/badge/Inference_Speed-7.3_tok%2Fsec-brightgreen" alt="Inference Speed">
  <img src="https://img.shields.io/badge/Precision-BF16-orange" alt="Compute Precision">
  <img src="https://img.shields.io/badge/License-Apache_2.0-yellow" alt="License">
  <img src="https://img.shields.io/badge/Deploy-Vercel_%2B_Render-black" alt="Deploy">
</p>

> **A high-efficiency, production-grade conversational AI powered by Vyrex Hyperion 3 (Qwen3-1.7B QLoRA fine-tune) featuring progressive natural sentence boundary calibration, zero-repetition circuit breaking, and sub-second streaming inference — engineered and trained by Suborno Roy Chowdhury.**

VYREX AI is an independent, locally-running conversational AI assistant. It holds multi-turn conversations, synthesizes idiomatic code, explains complex AI/ML architectures, and terminates thoughts cleanly without rambling or word-loop degeneration — running locally on consumer GPUs (RTX 3050 6GB) or deployed globally across Vercel and Render.

---

## Table of Contents

- [What is VYREX AI? (In Plain English)](#what-is-vyrex-ai-in-plain-english)
- [Model Architecture & Specifications](#model-architecture--specifications)
- [🛠️ Deep-Dive: Inference Challenges & Step-by-Step Engineering Solutions](#️-deep-dive-inference-challenges--step-by-step-engineering-solutions)
  - [1. Problem Diagnosis & Symptoms](#1-problem-diagnosis--symptoms)
  - [2. Step-by-Step Engineering Strategy](#2-step-by-step-engineering-strategy)
  - [3. Verified Benchmark Results (10/10 Prompts Passed)](#3-verified-benchmark-results-1010-prompts-passed)
- [Key Capabilities](#key-capabilities)
- [🖥️ Live Terminal Demos (Real Outputs)](#-live-terminal-demos-real-outputs)
- [What's Inside the Project?](#whats-inside-the-project)
- [System Requirements](#system-requirements)
- [Quickstart: Local Installation](#quickstart-local-installation)
- [💬 Interactive Terminal Chat](#-interactive-terminal-chat)
- [🌐 Full-Stack Web App: Vercel Frontend & Render Backend](#-full-stack-web-app-vercel-frontend--render-backend)
  - [Deploying Backend on Render](#1-deploying-backend-on-render)
  - [Deploying Frontend on Vercel](#2-deploying-frontend-on-vercel)
- [🚀 Production Serving: OpenAI-Compatible REST API](#-production-serving-openai-compatible-rest-api)
- [🏎️ Turbocharged Inference with vLLM](#️-turbocharged-inference-with-vllm)
- [☁️ Cloud Deployment Blueprints](#️-cloud-deployment-blueprints)
- [Training Methodology & Dataset Preparation](#training-methodology--dataset-preparation)
- [About the Creator](#about-the-creator)
- [License](#license)

---

## What is VYREX AI? (In Plain English)

Think of VYREX AI as an intelligent, sharply focused AI assistant running directly on your computer or hosted on lightweight cloud infrastructure:

1. **The Base Student:** We started with **Qwen3-1.7B** by Alibaba — a foundational model with 1.7 billion parameters containing vast general knowledge, code syntax, and reasoning fundamentals.
2. **The Specialization (QLoRA Fine-Tuning):** We adapted the model specifically for focused conversational intelligence, clean technical explanations, and accurate identity grounding through Parameter-Efficient Fine-Tuning (QLoRA) using curated dialogues and negative constraint examples.
3. **The Inference Engine:** Rather than relying solely on raw model outputs, VYREX AI employs an inference controller that stops run-on rambling, prevents word repetition loops, cleans trailing artifacts, and terminates each thought at natural grammatical sentence boundaries.

The resulting model is **Vyrex Hyperion 3**.

---

## Model Architecture & Specifications

| Property | Value | Source / Verification |
|---|---|---|
| **Model Name** | `Vyrex Hyperion 3` | System identity & checkpoint |
| **Base Model** | `Qwen/Qwen3-1.7B` | `pretrained/qwen3_1.7b/original` |
| **Fine-Tuning Method** | QLoRA (Quantized Low-Rank Adaptation) | `adapter_config.json` |
| **Active Checkpoint** | `vyrex_1_7b_qlora_v3 / best_checkpoint` | LoRA Adapter ($r=16, \alpha=32$) |
| **Target Modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Full attention + MLP projections |
| **Quantization** | 4-bit Normal Float (NF4) with Double Quantization | BitsAndBytes Config |
| **Compute Precision** | BF16 (`bfloat16`) | Native Ampere+ tensor cores |
| **Stop Token ID** | `151645` (`<|im_end|>`) | Qwen3 chat template EOS |
| **VRAM Footprint** | ~2.7 GB allocated (~3.3 GB total overhead) | Validated on NVIDIA RTX 3050 6GB Laptop GPU |
| **Inference Speed** | **7.0 – 7.9 tokens/sec** | High-throughput streaming decode |

---

## 🛠️ Deep-Dive: Inference Challenges & Step-by-Step Engineering Solutions

During real-world deployment and evaluation of the 1.7B parameter model on consumer hardware, several behavioral challenges emerged. Below is the full engineering post-mortem and step-by-step resolution.

### 1. Problem Diagnosis & Symptoms

```
┌────────────────────────────────────────────────────────────────────────┐
│ SYMPTOM 1: "HEAVILY HEAVILY..." DEGENERATION LOOP                      │
│ Output: "...depending largely heavily heavily heavily heavily heavily..."│
│ Root Cause: Low-entropy greedy decoding attractor basin where repeated │
│ tokens reinforce their own self-attention weights indefinitely.        │
├────────────────────────────────────────────────────────────────────────┤
│ SYMPTOM 2: RUN-ON CONJUNCTION CHAINING                                 │
│ Output: "...which allows x, where y depends on z, thus we see..."      │
│ Root Cause: Connective conjunctions ('and', 'which', 'where', 'thus')  │
│ delayed the end-of-turn token, causing generations to hit max_tokens.  │
├────────────────────────────────────────────────────────────────────────┤
│ SYMPTOM 3: DECODE SPEED COLLAPSE (2.3 TOKENS/SEC)                      │
│ Root Cause: Naive usage of `no_repeat_ngram_size=4` incurred an O(n)   │
│ hash lookup at every token step, causing a 60% GPU slowdown.           │
├────────────────────────────────────────────────────────────────────────┤
│ SYMPTOM 4: CHINESE (CJK) TOKEN HALLUCINATION AFTER SHORT ANSWERS       │
│ Output: "I was created by Suborno Roy Chowdhury. 具有战士\nWhat do I..."│
│ Root Cause: Base model pre-training artifact emitting stray CJK tokens │
│ when generation fails to terminate immediately after identity phrases. │
├────────────────────────────────────────────────────────────────────────┤
│ SYMPTOM 5: MID-SENTENCE CUT-OFFS                                       │
│ Output: "...within parameter space representing best" [Cut off]        │
│ Root Cause: Blunt token threshold caps terminating output mid-phrase   │
│ rather than at natural grammatical punctuation boundaries.             │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Step-by-Step Engineering Strategy

Instead of computationally expensive retraining, we engineered an inference-time solution combining custom PyTorch logits processors, streaming circuit breakers, and post-generation boundary enforcement:

#### Step 1: Remove Costly N-Gram Constraints & Restore GPU Throughput
- **Diagnosis:** `no_repeat_ngram_size=4` was degrading decode throughput from **~7.5 tps down to 2.3 tps** on NVIDIA RTX 3050.
- **Action:** Removed `no_repeat_ngram_size` entirely.
- **Alternative:** Applied `repetition_penalty=1.18` directly in model generation kwargs. This penalizes already-generated token logits with zero $O(n)$ search penalty, immediately restoring speed to **7.0–7.9 tps**.

#### Step 2: Implement Progressive Boundary-Aware Logits Processor
To solve run-on sentences without cutting the model off prematurely, we authored `CalibratedStopLogitsProcessor`:
- **Boundary Token Detection:** Pre-tokenizes sentence-closing punctuation (`.`, `!`, `?`, `\n\n`, ```` `).
- **Progressive Bias Schedule:** At each token step, if the prior token is a sentence boundary, it dynamically boosts `<|im_end|>` (token `151645`):
  $$\text{Bias}(t) = \begin{cases} +10.0 & \text{if } t > 25 \\ +25.0 & \text{if } t > 50 \\ +50.0 & \text{if } t > 80 \end{cases}$$
- **Conjunction Suppression:** After 40 tokens, connective conjunction tokens (` and`, ` which`, ` where`, ` so`, ` thus`, ` therefore`, ` etc`) receive a $-6.0$ penalty, discouraging recursive clause expansion.

#### Step 3: Multi-Tier Streaming Circuit Breaker
In `hf_engine.py`, the async `TextIteratorStreamer` generator runs through a multi-tier safety filter:
1. **Word-Level Repetition Breaker:** Tracks the last 12 words; if 3 identical words appear in consecutive succession, streaming terminates immediately.
2. **CJK Hallucination Shield:** Scans incoming tokens for Unicode range `\u4e00`–`\u9fff`. If any CJK character appears, generation halts instantly before stray characters reach the user.
3. **Clean Boundary Cap:** If the stream exceeds 90 tokens and finishes any sentence boundary (`.`, `!`, `?`, ```` `), it halts cleanly.

#### Step 4: Post-Generation Sentence Trimmer
In synchronous batch calls, if an answer reaches the token limit without terminating on closing punctuation, the post-processor locates the last complete sentence boundary (`.`, `!`, `?`) and trims trailing sentence fragments. Code blocks with odd fence counts are closed automatically with `\n``` `.

#### Step 5: System Prompt Tightening
Configured `DEFAULT_SYSTEM_PROMPT` to enforce sharp conciseness, strict attribution to creator **Suborno Roy Chowdhury**, and immediate stop behavior upon answer completion.

---

### 3. Verified Benchmark Results (10/10 Prompts Passed)

All 10 prompts from the floating website marquee were tested live against the running backend on local CUDA:

| Prompt ID & Title | Latency | Tokens | Speed | Output Quality & Termination | Status |
|---|---|---|---|---|---|
| **`p10` Who Created You** | **3.10s** | 24 | **7.7 tps** | Exact attribution to Suborno Roy Chowdhury, zero CJK artifacts, clean period stop. | **PASS** |
| **`p2` Code & Architecture** | **6.81s** | 54 | **7.9 tps** | Clean Python implementation, docstring, terminates cleanly. | **PASS** |
| **`p6` Agentic AI UX** | **8.29s** | 63 | **7.6 tps** | Crisp bullet points (UCD, NLP integration), direct and to the point. | **PASS** |
| **`p8` QLoRA Article Intro** | **9.75s** | 65 | **6.7 tps** | Highly engaging 2-paragraph technical introduction, no sign-off fluff. | **PASS** |
| **`p7` Transformer Attention** | **10.66s** | 77 | **7.2 tps** | Accurate multi-head dot product explanation, concise 3 bullets. | **PASS** |
| **`p3` Write & Refine** | **11.12s** | 77 | **6.9 tps** | Persuasive product announcement, terminates cleanly on exclamation mark. | **PASS** |
| **`p1` Startup Ideas** | **14.69s** | 108 | **7.4 tps** | 2 structured edge AI ideas for 2026, down from 99s previously. | **PASS** |
| **`p5` FastAPI Backend** | **20.23s** | 155 | **7.7 tps** | Valid async route with dependencies, router, and validation. | **PASS** |
| **`p9` Python Duplicates** | **24.59s** | 172 | **7.0 tps** | Proper `seen` set duplicate tracker, explanations, clean stop. | **PASS** |
| **`p4` Neural Network Analogy** | **25.02s** | 173 | **6.9 tps** | Intuitive gear/neuron analogy, stops cleanly on `!`, no rambling. | **PASS** |

**Summary of Improvements:**
- ⚡ **Decode Latency:** Reduced by **up to 75%** on complex queries.
- 🔁 **Word Repetitions:** **0.0%** (100% eliminated).
- 🈲 **CJK Hallucinations:** **0.0%** (100% eliminated).
- 🎯 **Sentence Boundary Completeness:** **100%**.

---

## Key Capabilities

- **Natural Sentence Boundary Termination:** Custom calibrated logits processor boosts the stop token at grammatical boundaries (`.`, `!`, `?`, `\n`, ```` `).
- **Fast 4-Bit NF4 Inference:** Double quantization and BF16 compute fit the model within **~2.7 GB of VRAM**, running at **~7.3 tps** on gaming laptops.
- **Multi-Turn Context Tracking:** Retains conversation history across turns for coherent, contextual dialogues.
- **Creator & Provenance Grounding:** Answers identity and architectural questions with accurate attribution to creator **Suborno Roy Chowdhury**.
- **OpenAI-Compatible REST API:** Exposes `/v1/chat/completions` with Server-Sent Events (SSE) token streaming.

---

## 🖥️ Live Terminal Demos (Real Outputs)

### 1. Identity & Architecture Grounding

* **Prompt:** `"Who created you and what is your architecture?"`
* **Response (24 tokens | 3.1s | 7.7 tps):**
  > *"I was created by Suborno Roy Chowdhury, my model's name is Vyrex Hyperion 3."*
* **Key Insight:** Clean, direct, proud attribution to its creator and model identity without hallucination.

### 2. Technical Explanation with Negative Constraints

* **Prompt:** `"Explain how 4-bit NF4 quantization works in exactly two sentences. Do not include an introduction or conclusion."`
* **Response (48 tokens):**
  > *"4-bit NF4 quantization compresses floating-point numbers into fixed-point representations that use four bits per number, reducing memory usage while maintaining accuracy. This technique enables efficient storage and computation of large datasets by minimizing data size without sacrificing precision."*
* **Key Insight:** Adheres strictly to the two-sentence constraint without introductory preambles or concluding sign-offs.

---

## What's Inside the Project?

```
vyrex-ai-phase1/
├── backend/                            ← Render-ready FastAPI microservice
│   ├── config.py                       ← Environment & CORS configuration
│   ├── engine.py                       ← Multi-mode inference controller (GPU/Proxy/Edge)
│   ├── main.py                         ← FastAPI endpoints (/health, /v1/chat/completions)
│   ├── schemas.py                      ← Pydantic v2 schemas
│   ├── requirements.txt                ← Lightweight production dependencies
│   ├── Dockerfile                      ← Container configuration
│   └── render.yaml                     ← Render blueprint definition
│
├── frontend/                           ← Vercel-ready Next.js 14 frontend
│   ├── src/app/page.tsx                ← Chat interface, marquee cards, sidebar
│   ├── src/components/                 ← ChatMessage, Sidebar, ThemeToggle
│   ├── tailwind.config.ts              ← Tailwind CSS theme configuration
│   ├── vercel.json                     ← Vercel build & routing configuration
│   └── package.json                    ← Next.js dependencies
│
├── src/vyrex_ai/serving/               ← Core inference engine
│   ├── hf_engine.py                    ← NF4 QLoRA engine, stop logits processor, streamer
│   ├── schemas.py                      ← Request/response schemas
│   └── engine_base.py                  ← Abstract engine interface
│
├── pretrained/qwen3_1.7b/
│   ├── original/                       ← Base Qwen3-1.7B weights (3.22 GB)
│   └── adapters/vyrex_1_7b_qlora_v3/   ← Active Vyrex Hyperion 3 QLoRA adapter
│
├── scripts/                            ← Benchmark, test, and training scripts
│   ├── benchmark_floating_prompts.py   ← 10-prompt website marquee test suite
│   ├── benchmark_v3_quality.py         ← 4-prompt quality benchmark
│   └── pretrained/                     ← Training and dataset preparation scripts
│
├── render.yaml                         ← Root Render Blueprint specification
└── README.md                           ← Project documentation
```

---

## System Requirements

| Requirement | Details |
|---|---|
| **Operating System** | Windows 10/11 or Linux (Ubuntu 20.04+) |
| **Python** | 3.10, 3.11, or 3.12 |
| **GPU** | NVIDIA GPU with Ampere+ architecture supporting BF16 (RTX 3050/3060/4060, A100, etc.) |
| **VRAM** | ~3–4 GB for inference; ~8–12 GB for fine-tuning |
| **CUDA** | CUDA 12.1+ |

---

## Quickstart: Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/subornoroychowdhury7-afk/Vyrex-AI.git
cd Vyrex-AI
```

### 2. Setup Virtual Environment
```bash
# Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install inference and fine-tuning libraries
pip install transformers peft bitsandbytes accelerate fastapi uvicorn sse-starlette httpx pydantic
```

---

## 💬 Interactive Terminal Chat

To chat directly with Vyrex Hyperion 3 inside your terminal:

```powershell
.\.venv\Scripts\python scripts/pretrained/vyrex_1_7b_chat.py
```

Commands available in-chat:
- `/reset` — Clears conversation history
- `/exit` — Exits the interactive session

---

## 🌐 Full-Stack Web App: Vercel Frontend & Render Backend

VYREX AI features a modern full-stack web application designed for cloud deployment:
- **Frontend (`frontend/`):** Next.js 14 (App Router) + Tailwind CSS + Lucide Icons. Features dark/bright mode toggling, real-time SSE token streaming, left chat history sidebar, and responsive marquee prompt cards.
- **Backend (`backend/`):** Asynchronous FastAPI REST API with CORS headers, health checks, multi-mode inference fallback, and OpenAI `/v1/chat/completions` compatibility.

```
┌──────────────────────────────┐              ┌──────────────────────────────┐
│      FRONTEND (Vercel)       │   HTTPS/SSE  │       BACKEND (Render)       │
│  - Next.js 14 (App Router)   │ ───────────> │  - FastAPI (Python 3.11)     │
│  - Bright / Dark Mode        │ <─────────── │  - OpenAI /v1/chat API       │
│  - SSE Token Streaming       │              │  - Multi-Mode Inference      │
└──────────────────────────────┘              └──────────────────────────────┘
```

### 1. Deploying Backend on Render

1. Go to **[dashboard.render.com](https://dashboard.render.com/)** and click **New +** → **Web Service**.
2. Connect your repository: `https://github.com/subornoroychowdhury7-afk/Vyrex-AI`.
3. Configure settings:
   - **Root Directory:** `backend`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path:** `/health`
4. Set Environment Variables:
   - `PYTHON_VERSION` = `3.11.9`
   - `ENVIRONMENT` = `production`
   - `ALLOWED_ORIGINS` = `*`
   - *(Optional)* `FALLBACK_API_KEY` = Your upstream API key (e.g. Groq/OpenRouter) for zero-latency serverless responses on Render CPU.
   - *(Optional)* `REMOTE_GPU_URL` = Tunnel URL pointing to your local GPU.
5. Deploy and copy your service URL (e.g., `https://vyrex-ai-backend.onrender.com`).

### 2. Deploying Frontend on Vercel

1. Go to **[vercel.com/new](https://vercel.com/new)** and import `subornoroychowdhury7-afk/Vyrex-AI`.
2. Configure settings:
   - **Framework Preset:** `Next.js`
   - **Root Directory:** `frontend` *(Important!)*
3. Add Environment Variable:
   - `NEXT_PUBLIC_API_URL` = `https://vyrex-ai-backend.onrender.com` *(Your Render URL)*
4. Click **Deploy**. Your AI web app is live worldwide!

---

## 🚀 Production Serving: OpenAI-Compatible REST API

Start the local backend server:

```powershell
cd backend
..\.venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Endpoints Available:
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`
- **Models List:** `http://localhost:8000/v1/models`
- **Chat Completions:** `http://localhost:8000/v1/chat/completions`

### Query with Official OpenAI SDK:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="vyrex-local")

response = client.chat.completions.create(
    model="Vyrex Hyperion 3",
    messages=[
        {"role": "user", "content": "Explain how transformers process tokens in parallel."}
    ],
    temperature=0.0,
    max_tokens=180,
)

print(response.choices[0].message.content)
```

---

## 🏎️ Turbocharged Inference with vLLM

For high-concurrency production deployments with continuous batching and PagedAttention:

```bash
# Serve with vLLM
python scripts/serve_vllm.py --port 8000

# Or merge the LoRA adapter directly into the base weights
python scripts/merge_lora_weights.py --output-dir pretrained/vyrex_1_7b_merged
vllm serve pretrained/vyrex_1_7b_merged --port 8000 --max-model-len 2048
```

---

## ☁️ Cloud Deployment Blueprints

Pre-configured deployment scripts are available in [`deploy/`](deploy/):

| Platform | Deployment Target | Command |
|---|---|---|
| **Modal** | Serverless Auto-Scaling GPU | `modal deploy deploy/modal_app.py` |
| **RunPod** | Serverless Endpoint Handler | `python deploy/runpod_handler.py` |
| **Kubernetes** | Autoscaling GPU Pods | `kubectl apply -f deploy/k8s-deployment.yaml` |
| **AWS SageMaker** | Dedicated Endpoint Deployment | `python deploy/sagemaker_deploy.py --role-arn <ARN>` |

---

## Training Methodology & Dataset Preparation

If you wish to inspect or extend the QLoRA fine-tuning pipeline:

1. **Build Dataset:**
   ```bash
   python scripts/pretrained/build_vyrex_v3_dataset.py
   ```
2. **Execute QLoRA Training:**
   ```bash
   python scripts/pretrained/train_vyrex_qwen3_1_7b_qlora_v3.py --epochs 2 --run-name vyrex_1_7b_qlora_v3
   ```
   - Stop Token Weighted Loss: End-of-turn token (`<|im_end|>`) weighted at **$5\times$** loss during cross-entropy calculation to enforce early stopping.
   - Validation Loss: Improved from `1.5718` down to `1.3671` over 806 optimization steps.

---

## About the Creator

**VYREX AI** was created, architected, and developed by **Suborno Roy Chowdhury** — an AI/ML Engineer and Researcher based in Kolkata, India.

- 🎓 Pursuing B.Tech in Artificial Intelligence & Machine Learning at St. Thomas' College of Engineering & Technology, Kolkata (YGPA: 8.71)
- 📜 Enrolled in the Certificate Programme in AI for Real-World Applications at **IIT Kharagpur**
- 🔧 Core expertise: PyTorch, LLM Fine-Tuning (LoRA / QLoRA), CUDA Inference Optimization, FastAPI, Next.js, and Full-Stack AI Platforms
- 🚀 Portfolio Projects:
  - **VYREX AI (Vyrex Hyperion 3)** — Independent conversational AI with calibrated natural sentence boundary termination.
  - **DocQ&A** — Enterprise Multi-Modal RAG Platform with OCR, Vector Retrieval (ChromaDB), FastAPI, and Next.js.

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
