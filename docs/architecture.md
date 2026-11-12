# VYREX AI technical architecture

## Product scope

An English-only hybrid assistant with three initial competency areas: general knowledge and reasoning, educational explanations/aptitude, and programming assistance. The first release is deliberately small and research-oriented.

## Model family

Decoder-only causal Transformer. Each input position may attend only to itself and earlier positions. The training target is the next token. The initial block design is pre-norm RMSNorm, multi-head causal self-attention with RoPE, residual connection, RMSNorm, SwiGLU feed-forward network, and residual connection. Embedding/output weights are tied in the initial experiment.

## Model boundary

The tokenizer is trained on approved English corpus material. Tokenizer vocabulary size and special-token IDs are immutable for a checkpoint family. The model config stores all architecture dimensions. A checkpoint loader must fail closed on incompatible vocabulary, dimensions, or tokenizer fingerprints.

## Training stages

1. **Pretraining:** document-level shuffled/streamed text, causal next-token loss, deduplicated source manifests, held-out validation data.
2. **Supervised fine-tuning:** high-quality instruction/response examples with explicit turn boundaries and loss masking on prompt tokens where appropriate.
3. **Optional preference optimization:** only after a reliable supervised baseline and a reviewed preference dataset exist. Not in the initial scope.

## Evaluation

Track global validation loss/perplexity and domain-specific held-out loss for general, education, and coding slices. Add task-level evaluations for instruction following, factuality, explanation quality, code correctness, and refusal/safety behavior before public release. Benchmark results must include the exact model/checkpoint, tokenizer, dataset version, prompt template, and decoding settings.

## Cloud topology

- GPU worker: training and evaluation jobs.
- Persistent object storage: approved data shards, tokenizer files, checkpoints, manifests, and logs.
- Experiment metadata: config + seed + commit hash + data manifest + runtime diagnostics.
- Optional later: inference VM/container and API gateway. Never expose training credentials in a notebook or repository.
