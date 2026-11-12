"""Source-group-aware dataset splitting for VYREX AI."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .readers import Document


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[Document, ...]
    validation: tuple[Document, ...]
    test: tuple[Document, ...]


def _select_groups(
    grouped: dict[str, list[Document]],
    candidates: list[str],
    target: int,
) -> set[str]:
    """Select whole groups as close as possible to target without exceeding it."""
    if target <= 0 or not candidates:
        return set()

    # Subset-sum dynamic programming.
    # Each reachable total stores one deterministic group combination.
    reachable: dict[int, tuple[str, ...]] = {0: ()}

    for group_name in candidates:
        group_size = len(grouped[group_name])

        if group_size > target:
            continue

        for current_total in sorted(reachable, reverse=True):
            new_total = current_total + group_size

            if new_total <= target and new_total not in reachable:
                reachable[new_total] = (
                    *reachable[current_total],
                    group_name,
                )

        if target in reachable:
            break

    selected = reachable[max(reachable)]
    return set(selected)


def split_documents(
    documents: list[Document] | tuple[Document, ...],
    source_groups: dict[str, str],
    train_ratio: float = 0.98,
    validation_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 42,
) -> DatasetSplits:
    """Split documents by source group to prevent cross-split leakage."""
    ratios = (train_ratio, validation_ratio, test_ratio)

    if any(ratio < 0 or ratio > 1 for ratio in ratios):
        raise ValueError("Split ratios must be between 0 and 1.")

    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.")

    if not documents:
        raise ValueError("Cannot split an empty document collection.")

    grouped: dict[str, list[Document]] = {}

    for document in documents:
        group = source_groups.get(document.source_path)

        if not group or not group.strip():
            raise ValueError(
                f"Missing source group for: {document.source_path}"
            )

        grouped.setdefault(group, []).append(document)

    rng = random.Random(seed)
    group_names = sorted(grouped)
    rng.shuffle(group_names)

    total_documents = len(documents)
    validation_target = round(total_documents * validation_ratio)
    test_target = round(total_documents * test_ratio)

    validation_groups = _select_groups(
        grouped,
        group_names,
        validation_target,
    )

    remaining_groups = [
        name for name in group_names
        if name not in validation_groups
    ]

    test_groups = _select_groups(
        grouped,
        remaining_groups,
        test_target,
    )

    train_groups = [
        name for name in remaining_groups
        if name not in test_groups
    ]

    def collect(names: list[str] | set[str]) -> tuple[Document, ...]:
        return tuple(
            document
            for name in sorted(names)
            for document in grouped[name]
        )

    return DatasetSplits(
        train=collect(train_groups),
        validation=collect(validation_groups),
        test=collect(test_groups),
    )
