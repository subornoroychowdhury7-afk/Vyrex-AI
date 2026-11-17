"""Leakage-aware grouped splitting for English-level ML datasets."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict
from typing import Sequence

from sklearn.model_selection import StratifiedGroupKFold


def normalize_sentence(text: str) -> str:
    """Normalize text for duplicate detection."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Sentence text must be non-empty")

    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.casefold().split())


def sentence_group_id(text: str) -> str:
    """Return a stable SHA-256 group ID for normalized sentence text."""

    normalized = normalize_sentence(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_group_ids(examples: Sequence) -> list[str]:
    """Build duplicate-safe group IDs for labelled examples."""

    return [
        sentence_group_id(example.text)
        for example in examples
    ]


def create_stratified_group_splits(
    examples: Sequence,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Create stratified folds without allowing duplicate-text leakage."""

    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise ValueError("n_splits must be an integer")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
        or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer")

    if not examples:
        raise ValueError("At least one labelled example is required")

    labels = []
    groups = build_group_ids(examples)
    group_labels = defaultdict(set)

    for example, group in zip(examples, groups):
        label = example.label

        if not isinstance(label, str) or not label.strip():
            raise ValueError("Every example must have a valid label")

        labels.append(label)
        group_labels[group].add(label)

    conflicting = [
        group for group, assigned_labels in group_labels.items()
        if len(assigned_labels) > 1
    ]

    if conflicting:
        raise ValueError(
            "Identical normalized sentences have conflicting labels"
        )

    class_groups = defaultdict(set)

    for label, group in zip(labels, groups):
        class_groups[label].add(group)

    insufficient = {
        label: len(class_group_ids)
        for label, class_group_ids in class_groups.items()
        if len(class_group_ids) < n_splits
    }

    if insufficient:
        raise ValueError(
            f"Insufficient independent groups per class for "
            f"{n_splits} folds: {insufficient}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits = []

    for train_indices, validation_indices in splitter.split(
        X=[[0.0]] * len(examples),
        y=labels,
        groups=groups,
    ):
        train_indices = train_indices.tolist()
        validation_indices = validation_indices.tolist()

        train_groups = {groups[i] for i in train_indices}
        validation_groups = {groups[i] for i in validation_indices}

        if train_groups & validation_groups:
            raise RuntimeError("Duplicate-text leakage detected")

        splits.append((train_indices, validation_indices))

    return splits
