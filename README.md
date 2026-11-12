# VYREX AI — Phase 1 Project Foundation

**Project intent:** build an English-language, hybrid (general knowledge + education + coding) decoder-only Transformer LLM, starting from randomly initialized weights. This repository is a foundation, not a pretrained model or working chatbot yet.

## Phase 1 status

Implemented:
- Cloud-provider-neutral Python project layout.
- Typed configuration loading and validation.
- Runtime diagnostics for Python, PyTorch, CUDA, and visible NVIDIA GPUs.
- Console/file logging.
- CLI commands for diagnostics, configuration validation, and directory initialization.
- Unit tests for configuration validation.

Not implemented yet: tokenizer training, Transformer layers, dataset ingestion, pretraining, instruction tuning, inference, or a chat UI. Those are later phases.

## Architecture plan

```text
Public / licensed / owned English text sources
                    |
          Data inventory & license audit
                    |
       Clean -> deduplicate -> quality filter
                    |
         Train / validation / test splits
                    |
        Custom English BPE tokenizer (Phase 2)
                    |
       Token IDs + packed training sequences
                    |
     Decoder-only Transformer (Phase 3)
       RoPE + RMSNorm + causal attention
                + SwiGLU MLP
                    |
       Next-token pretraining (Phase 4)
                    |
        Instruction / chat fine-tuning
                    |
       Evaluation, safety and regression tests
                    |
        Local inference API + chat UI
```

### Initial model design (a starting experiment, not a permanent commitment)

| Setting | Initial value |
|---|---:|
| Architecture | Decoder-only Transformer |
| Vocabulary | 16,000 BPE tokens |
| Hidden width | 384 |
| Transformer blocks | 6 |
| Attention heads | 6 |
| SwiGLU intermediate width | 1,024 |
| Context length | 1,024 tokens |
| Position handling | RoPE |
| Normalization | RMSNorm |
| Embeddings | Tied token embedding / output head |
| Dropout | 0.10 |

With tied embeddings and the planned block dimensions, the parameter count is approximately 17 million. Exact count will be reported by the implemented model in Phase 3.

### Hybrid training objective

The initial proposed *source mixture* is 60% general text, 25% educational content, and 15% code/programming material. These are hypotheses for the first controlled experiment, not fixed requirements. We will adjust them only after measuring source quality, token counts, licensing, duplicates, and validation performance by domain.

Pretraining teaches language patterns and broad representations. A separate instruction-tuning stage teaches the model to answer questions, explain concepts, and assist with coding. A small Q&A dataset alone is not enough to pretrain a general language model.

## Cloud training strategy

The code is provider-neutral and expects a Linux environment with Python and a compatible accelerator runtime. The first runs should be small experiments on a rented NVIDIA GPU; the same repository can be used in a Colab notebook for setup checks. Keep datasets and checkpoints on persistent object storage or a mounted persistent volume, not only on an ephemeral VM disk.

Do not pick a CUDA wheel by guesswork. Install PyTorch using the official selector for the VM's OS, Python, and CUDA/ROCm environment: <https://pytorch.org/get-started/locally/>. PyTorch's install page recommends selecting the compute platform matching the hardware and verifying with `torch.cuda.is_available()`.

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

### 2. Install the correct PyTorch build

Use the official PyTorch install selector: <https://pytorch.org/get-started/locally/>. Select the VM's OS, Pip, Python, and the CUDA version supported by the installed NVIDIA driver. For a CPU-only configuration check, select CPU.

### 3. Install VYREX AI and development tools

```bash
pip install -e ".[dev]"
```

### 4. Run Phase 1 checks

```bash
vyrex doctor
vyrex show-config --config configs/base.yaml
vyrex init --config configs/base.yaml
pytest -q
```

Equivalent CLI without installation: `python -m vyrex_ai.cli doctor` from the repository after setting `PYTHONPATH=src`.

## Repository layout

```text
vyrex-ai/
├── configs/              # Versioned experiment configurations
├── data/                 # Dataset staging; contents excluded from Git
│   ├── raw/
│   ├── processed/
│   └── splits/
├── checkpoints/          # Model checkpoints; contents excluded from Git
├── artifacts/            # Tokenizer and exported model artifacts
├── docs/                 # Architecture, data governance, roadmap
├── logs/                 # Local logs; contents excluded from Git
├── scripts/              # Reproducible operational scripts
├── src/vyrex_ai/         # Installable Python package
└── tests/                # Automated tests
```

## Engineering rules

1. Never describe a random-weight checkpoint as a pretrained or capable assistant model.
2. Every experiment records config, seed, code revision, data manifest, tokenizer version, and hardware details.
3. Keep a held-out test set untouched until final evaluation; do not train on it.
4. Track training and validation loss separately, plus per-domain metrics for general, education, and code.
5. Use only data with a documented lawful basis and compatible license; remove secrets and personal data where possible.
6. Do not commit raw datasets, model checkpoints, credentials, or personal access tokens to Git.
7. Checkpoint atomically and test restoration before long runs.
8. Do not scale model size until a small run is stable and the compute budget is understood.

## Phase gates

- **Phase 1 (current):** project foundation and runtime checks.
- **Phase 2:** custom BPE tokenizer, tokenizer tests, and data manifest format.
- **Phase 3:** model implementation, parameter-count report, shape tests, causal-mask tests, and a tiny overfit test.
- **Phase 4:** pretraining data pipeline, checkpoints, resume, validation, and experiment logging.
- **Phase 5:** instruction data formatting and supervised fine-tuning.
- **Phase 6:** benchmark and human-review evaluation, safety, and regression suite.
- **Phase 7:** inference interface and deployment.
