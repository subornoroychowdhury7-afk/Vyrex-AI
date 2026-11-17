"""Streaming fixed-length token blocks for VYREX AI training."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info


class TokenBlockDataset(IterableDataset[torch.Tensor]):
    """Stream JSONL documents into fixed-length causal-LM token blocks.

    Each document is encoded independently with the tokenizer's special
    tokens enabled. Blocks may cross document boundaries, but every
    boundary is explicitly marked by EOS followed by BOS.

    Incomplete final blocks are discarded. No padding is introduced.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Any,
        sequence_length: int,
    ) -> None:
        super().__init__()

        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
        ):
            raise TypeError("sequence_length must be an integer")

        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.block_size = sequence_length + 1

        if not self.jsonl_path.is_file():
            raise FileNotFoundError(
                f"Dataset JSONL file does not exist: {self.jsonl_path}"
            )

        for token_name in ("bos_token_id", "eos_token_id"):
            token_id = getattr(tokenizer, token_name, None)
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError(
                    f"Tokenizer must expose an integer {token_name}"
                )

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            raise RuntimeError(
                "TokenBlockDataset currently requires num_workers=0. "
                "Multi-worker token-stream sharding is not implemented."
            )

        pending: list[int] = []

        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(
                        f"Blank JSONL record at line {line_number}"
                    )

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_number}"
                    ) from exc

                if not isinstance(record, dict):
                    raise TypeError(
                        f"JSONL record at line {line_number} "
                        "must be an object"
                    )

                text = record.get("text")

                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"Record at line {line_number} "
                        "must contain non-empty text"
                    )

                token_ids = self.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                )

                if not token_ids:
                    continue

                pending.extend(token_ids)

                while len(pending) >= self.block_size:
                    block = pending[:self.block_size]
                    del pending[:self.block_size]
                    yield torch.tensor(block, dtype=torch.long)

        # Each worker discards only its own incomplete final block.
