from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

TOKENIZER_NAME = "vyrex-bpe-en-v1"
DEFAULT_VOCAB_SIZE = 16_000

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]


class VyrexTokenizer:
    """Byte-level BPE tokenizer for VYREX AI."""

    def __init__(
        self,
        backend: Tokenizer,
        *,
        requested_vocab_size: int | None = None,
        min_frequency: int | None = None,
    ) -> None:
        self.backend = backend
        self.requested_vocab_size = requested_vocab_size
        self.min_frequency = min_frequency
        self.pad_token_id = self._required_token_id(PAD_TOKEN)
        self.bos_token_id = self._required_token_id(BOS_TOKEN)
        self.eos_token_id = self._required_token_id(EOS_TOKEN)

        expected_ids = (0, 1, 2)
        actual_ids = (
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
        )

        if actual_ids != expected_ids:
            raise ValueError(
                "Special token IDs are incompatible: "
                f"expected {expected_ids}, got {actual_ids}"
            )

    def _required_token_id(self, token: str) -> int:
        token_id = self.backend.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Missing required special token: {token}")
        return token_id

    @property
    def vocab_size(self) -> int:
        """Return the actual vocabulary size."""
        return self.backend.get_vocab_size()

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        min_frequency: int = 2,
    ) -> VyrexTokenizer:
        """Train byte-level BPE from training texts."""
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer")
        if isinstance(min_frequency, bool) or not isinstance(min_frequency, int):
            raise TypeError("min_frequency must be an integer")

        if vocab_size < 259:
            raise ValueError("vocab_size must be at least 259")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")

        iterator = iter(texts)

        try:
            first_text = next(iterator)
        except StopIteration as exc:
            raise ValueError("Training text collection cannot be empty") from exc

        if not isinstance(first_text, str):
            raise TypeError("All training documents must be strings")

        def validated_texts() -> Iterable[str]:
            for text in chain([first_text], iterator):
                if not isinstance(text, str):
                    raise TypeError("All training documents must be strings")
                yield text

        backend = Tokenizer(models.BPE(unk_token=None))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=True,
        )
        backend.decoder = decoders.ByteLevel()
        backend.post_processor = processors.TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            special_tokens=[
                (BOS_TOKEN, 1),
                (EOS_TOKEN, 2),
            ],
        )

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        backend.train_from_iterator(
            validated_texts(),
            trainer=trainer,
        )

        actual_vocab_size = backend.get_vocab_size()

        if actual_vocab_size > vocab_size:
            raise RuntimeError(
                "Trained tokenizer exceeded its vocabulary target: "
                f"{actual_vocab_size} > {vocab_size}"
            )

        return cls(
            backend,
            requested_vocab_size=vocab_size,
            min_frequency=min_frequency,
        )

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Convert text into token IDs."""
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")

        return self.backend.encode(
            text,
            add_special_tokens=add_special_tokens,
        ).ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Convert token IDs back into text."""
        return self.backend.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def save(
        self,
        directory: str | Path,
        overwrite: bool = False,
    ) -> Path:
        """Save the tokenizer safely as tokenizer.json."""
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "tokenizer.json"
        temporary_path = output_dir / "tokenizer.json.tmp"

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Tokenizer already exists: {output_path}. "
                "Pass overwrite=True to replace it intentionally."
            )

        if temporary_path.exists():
            raise FileExistsError(
                f"Temporary tokenizer file already exists: {temporary_path}"
            )

        try:
            self.backend.save(str(temporary_path))

            if overwrite:
                temporary_path.replace(output_path)
            else:
                temporary_path.rename(output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        return output_path

    @classmethod
    def load(cls, path: str | Path) -> VyrexTokenizer:
        """Load a previously saved tokenizer."""
        tokenizer_path = Path(path)

        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file does not exist: {tokenizer_path}"
            )

        return cls(Tokenizer.from_file(str(tokenizer_path)))