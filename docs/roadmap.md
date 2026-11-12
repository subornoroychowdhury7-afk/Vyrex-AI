# Roadmap and acceptance criteria

## Phase 1 — Foundation (implemented in this starter)
- Project package installs cleanly.
- Base and smoke YAML configs load and validate.
- Runtime doctor reports Python/PyTorch/CUDA/GPU state without crashing when PyTorch is absent.
- Tests pass.

## Phase 2 — Tokenizer and corpus manifest
- Train a fresh English BPE tokenizer from approved project data.
- Define special tokens and freeze their IDs for v1.
- Verify encode/decode, unknown/Unicode handling, and serialization round trips.
- Produce source manifests and deduplication/split reports.

## Phase 3 — Model
- Implement and test causal attention, RoPE, RMSNorm, SwiGLU, residual blocks, tied embeddings, and causal language-model loss.
- Validate tensor shapes, no future-token leakage, finite gradients, deterministic seed behavior, and exact parameter count.
- Run a tiny-batch overfit test before scaling.

## Phase 4 — Pretraining
- Add streaming/packed data loaders, mixed precision, gradient accumulation, clipping, LR schedule, evaluation, atomic checkpoints, and resume.
- Complete a small pilot and verify validation loss improves before a longer run.

## Phase 5 — Instruction tuning
- Define a stable conversation format and quality rubric.
- Fine-tune on reviewed English instructions across all three domains.
- Evaluate held-out instruction examples.

## Phase 6 — Evaluation and safety
- Evaluate general, education, and coding slices.
- Add factuality, code execution tests in a sandbox, prompt robustness, safety/refusal checks, and regression tests.

## Phase 7 — Inference and deployment
- Implement streaming generation, context management, API, UI, monitoring, and model card.
