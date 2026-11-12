"""VYREX AI decoder-only Transformer architecture."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class VyrexConfig:
    vocab_size: int = 16000
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    intermediate_size: int = 1024
    max_seq_len: int = 1024
    dropout: float = 0.1
    tie_embeddings: bool = True

    def __post_init__(self):
        integer_fields = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "max_seq_len": self.max_seq_len,
        }

        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        if (self.hidden_size // self.num_heads) % 2 != 0:
            raise ValueError("Attention head dimension must be even for RoPE")

        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1)")

        if not isinstance(self.tie_embeddings, bool):
            raise TypeError("tie_embeddings must be a boolean")
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Applies rotary positional embeddings to query and key tensors."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inverse_frequency)
        return (
            frequencies.cos().to(dtype)[None, None, :, :],
            frequencies.sin().to(dtype)[None, None, :, :],
        )

    @staticmethod
    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        paired = x.reshape(*x.shape[:-1], -1, 2)
        even = paired[..., 0]
        odd = paired[..., 1]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        return x * cos.repeat_interleave(2, dim=-1) + (
            rotated * sin.repeat_interleave(2, dim=-1)
        )


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using PyTorch scaled dot-product attention."""

    def __init__(self, config: VyrexConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        cos, sin = self.rope(seq_len, x.device, query.dtype)
        query = self.rope.rotate(query, cos, sin)
        key = self.rope.rotate(key, cos, sin)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        return self.output(attended)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: VyrexConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer decoder block."""

    def __init__(self, config: VyrexConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.attention(self.attention_norm(x)))
        x = x + self.residual_dropout(self.mlp(self.mlp_norm(x)))
        return x


class VyrexTransformer(nn.Module):
    """Decoder-only language model for next-token prediction."""

    def __init__(self, config: VyrexConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("Input sequence exceeds max_seq_len")

        x = self.embedding_dropout(self.token_embeddings(input_ids))

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.final_norm(x))
        loss = None

        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss
