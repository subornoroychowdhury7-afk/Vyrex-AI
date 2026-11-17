"""Experimental V2.1 diagnostic linguistic feature extraction.

This module is additive. It does not change the V2 classifier,
scoring engine, or 20-feature ML schema.
"""

from __future__ import annotations

import re
from collections import Counter

from vyrex_ai.data.english_level_classifier_v2 import (
    WORD_PATTERN,
    analyze_sentence,
)


MODALS = {
    "can", "could", "may", "might", "must",
    "shall", "should", "will", "would",
}

COORDINATORS = {"and", "but", "or", "nor", "yet", "so"}

SUBORDINATE_MARKERS = {
    "although", "because", "before", "if", "once",
    "since", "though", "unless", "until", "when",
    "whenever", "whereas", "wherever", "while",
}

PREPOSITIONS = {
    "about", "above", "across", "after", "against",
    "among", "around", "at", "before", "behind",
    "below", "beneath", "beside", "between", "beyond",
    "by", "despite", "during", "for", "from", "in",
    "inside", "into", "near", "of", "off", "on",
    "onto", "over", "through", "throughout", "to",
    "toward", "under", "until", "upon", "with", "within",
    "without",
}

NOMINALIZATION_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity",
    "ance", "ence", "ism",
)

AUXILIARIES = {
    "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did",
    "can", "could", "may", "might", "must", "shall",
    "should", "will", "would",
}


COMMON_FINITE_VERBS = {
    "work", "go", "know", "think", "believe", "use",
    "make", "take", "see", "do", "say", "become",
    "remain", "continue", "study", "live", "read",
    "write", "learn", "cause", "lead", "affect",
    "influence", "require", "need", "want", "prefer",
    "produce", "provide", "include", "show", "explain",
    "consider", "argue", "suggest", "demonstrate",
    "establish", "evaluate", "compare", "depend",
    "occur", "exist", "matter", "seem",
}

IRREGULAR_PARTICIPLES = {
    "been", "built", "bought", "brought", "caught",
    "done", "felt", "found", "given", "gone", "held",
    "kept", "known", "left", "lost", "made", "paid",
    "read", "sent", "seen", "shown", "sold", "spent",
    "taken", "taught", "thought", "told", "won",
}

RELATIVE_MARKERS = {"that", "who", "whom", "which", "whose"}

RELATIVE_SUBJECTS = {
    "i", "you", "he", "she", "it", "we", "they",
    "who", "which", "someone", "somebody",
}

CONTENT_CLAUSE_NOUNS = {
    "idea", "fact", "belief", "claim", "thought",
    "possibility", "assumption", "notion", "argument",
    "statement", "impression", "suggestion", "hope",
    "knowledge", "understanding", "news", "truth",
}

NOMINALIZATION_EXCLUSIONS = {
    "experiment",
    "question",
    "multilingual",
    "tournament",
    "sucker",
    "eternity",
    "sentence",
    "city",
    "romance",
    "chance",
    "testament",
    "apartment",
    "government",
    "cement",
    "station",
    "serendipity",
}


def _looks_like_finite_verb(word: str) -> bool:
    return (
        word in AUXILIARIES
        or word in COMMON_FINITE_VERBS
        or word.endswith(("ed", "ies"))
        or (word.endswith("s") and len(word) > 3)
    )


def _count_relative_clauses(words: list[str], text: str = '') -> int:
    count = 0

    for index, word in enumerate(words):
        if word not in RELATIVE_MARKERS or index == 0:
            continue

        previous = words[index - 1]

        if word == "which" and previous == "that":
            continue

        following = words[index + 1:index + 5]
        if not following:
            continue

        if word == "that" and following[0] == "what":
            continue

        # Demonstrative "that" after a coordinator.
        if (
            word == "that"
            and previous in COORDINATORS
            and _looks_like_finite_verb(following[0])
        ):
            continue

        # Demonstrative "that" after explicit clause punctuation.
        if (
            word == "that"
            and _looks_like_finite_verb(following[0])
            and __import__("re").search(
                r"[-??;:]\s*that\b",
                text,
                flags=__import__("re").IGNORECASE,
            )
        ):
            continue

        # Content clauses are not relative clauses.
        if (
            previous in AUXILIARIES
            or previous in COMMON_FINITE_VERBS
            or previous in CONTENT_CLAUSE_NOUNS
        ):
            continue

        # Subject relative: "the person who works here"
        if _looks_like_finite_verb(following[0]):
            count += 1
            continue

        # Object relative: "the book that I purchased"
        if (
            following[0] in RELATIVE_SUBJECTS
            or (
                following[0] not in AUXILIARIES
                and len(following[0]) > 2
            )
        ):
            if any(
                _looks_like_finite_verb(token)
                for token in following[1:]
            ):
                count += 1

    return count

def _count_subordinate_clauses(words: list[str]) -> int:
    count = 0

    for index, word in enumerate(words):
        if word not in SUBORDINATE_MARKERS:
            continue

        following = words[index + 1:index + 6]
        if len(following) < 2:
            continue

        if any(
            token in AUXILIARIES
            or token in COMMON_FINITE_VERBS
            or token.endswith(("ed", "ies"))
            or (token.endswith("s") and len(token) > 3)
            for token in following[1:]
        ):
            count += 1

    return count


def _count_nominalization_candidates(words: list[str]) -> int:
    count = 0

    for word in words:
        candidates = [word]

        if word.endswith("ies") and len(word) > 4:
            candidates.append(word[:-3] + "y")
        elif word.endswith("s") and not word.endswith("ss"):
            candidates.append(word[:-1])

        if any(
            candidate not in NOMINALIZATION_EXCLUSIONS
            and candidate.endswith(NOMINALIZATION_SUFFIXES)
            for candidate in candidates
        ):
            count += 1

    return count


ADVERBS_BETWEEN_AUXILIARIES = {
    "already", "always", "carefully", "certainly", "currently",
    "fully", "just", "never", "not", "often", "probably", "quickly",
    "recently", "silently", "still", "usually",
}

BASE_VERBS = COMMON_FINITE_VERBS | {
    "arrive", "complete", "evaluate", "finish", "run",
    "examine", "study", "publish", "replace", "develop",
}


def _is_participle(word: str) -> bool:
    return (
        word.endswith(("ed", "en", "ing"))
        or word in IRREGULAR_PARTICIPLES
    )


def _next_verb_index(words: list[str], index: int) -> int | None:
    """Find a following verb while allowing a few intervening adverbs."""

    cursor = index + 1
    skipped = 0

    while (
        cursor < len(words)
        and words[cursor] in ADVERBS_BETWEEN_AUXILIARIES
        and skipped < 3
    ):
        cursor += 1
        skipped += 1

    return cursor if cursor < len(words) else None


def _is_base_verb(word: str) -> bool:
    return word in BASE_VERBS


def _is_auxiliary_at(words: list[str], index: int) -> bool:
    word = words[index]
    next_index = _next_verb_index(words, index)

    if next_index is None:
        return False

    following = words[next_index]

    # In this context, "that's been" means "that has been".
    if word == "that's" and _is_participle(following):
        return True

    if word in MODALS:
        return (
            following in AUXILIARIES
            or _is_base_verb(following)
        )

    if word in {"am", "is", "are", "was", "were", "be", "been", "being"}:
        return _is_participle(following)

    if word in {"have", "has", "had"}:
        return _is_participle(following)

    if word in {"do", "does", "did"}:
        return _is_base_verb(following)

    return False


def _count_auxiliary_chains(words: list[str]) -> int:
    auxiliary_indices = [
        index
        for index in range(len(words))
        if _is_auxiliary_at(words, index)
    ]

    count = 0
    group_size = 0
    previous_index = None

    for index in auxiliary_indices:
        if previous_index is None:
            group_size = 1
        else:
            gap = words[previous_index + 1:index]

            if (
                len(gap) <= 3
                and all(
                    token in ADVERBS_BETWEEN_AUXILIARIES
                    for token in gap
                )
            ):
                group_size += 1
            else:
                if group_size >= 2:
                    count += 1
                group_size = 1

        previous_index = index

    if group_size >= 2:
        count += 1

    return count


def analyze_sentence_v2_1(text: str) -> dict:
    """Return additive experimental diagnostic features."""

    baseline = analyze_sentence(text)
    words = WORD_PATTERN.findall(text.lower())
    counts = Counter(words)
    word_count = len(words)

    modal_count = sum(word in MODALS for word in words)
    coordinator_count = sum(
        word in COORDINATORS for word in words
    )
    subordinate_count = _count_subordinate_clauses(words)
    preposition_count = sum(
        word in PREPOSITIONS for word in words
    )
    nominalization_count = _count_nominalization_candidates(words)
    auxiliary_count = sum(
        _is_auxiliary_at(words, index)
        for index in range(len(words))
    )

    auxiliary_chain_count = _count_auxiliary_chains(words)

    sentence_count = len(
        re.findall(r"[.!?]+", text)
    )

    return {
        "baseline_feature_count": len(baseline),
        "word_count": word_count,
        "modal_auxiliary_count": modal_count,
        "coordinating_conjunction_count": coordinator_count,
        "subordinate_marker_count": subordinate_count,
        "relative_clause_count": _count_relative_clauses(words, text),
        "preposition_count": preposition_count,
        "nominalization_candidate_count": nominalization_count,
        "auxiliary_count": auxiliary_count,
        "auxiliary_chain_count": auxiliary_chain_count,
        "sentence_count": max(sentence_count, 1),
        "lexical_diversity": round(
            len(counts) / word_count, 4
        ),
        "long_word_ratio": round(
            sum(len(word) >= 9 for word in words) / word_count,
            4,
        ),
        "preposition_ratio": round(
            preposition_count / word_count, 4
        ),
        "nominalization_ratio": round(
            nominalization_count / word_count, 4
        ),
    }



