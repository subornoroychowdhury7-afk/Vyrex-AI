"""Interpretable linguistic feature extraction for VYREX AI V2."""

from __future__ import annotations

import re
from collections import Counter

LEVELS = (
    "basic_english",
    "elementary_intermediate_english",
    "upper_intermediate_english",
    "advanced_english",
)

WORD_PATTERN = re.compile(
    r"[A-Za-z]+(?:[\u0027\u2019-][A-Za-z]+)*"
)

ACADEMIC_VOCABULARY = {
    "abstract", "analysis", "assumption", "concept",
    "consequently", "constitute", "criterion", "derive",
    "distribution", "establish", "evaluate", "evidence",
    "fundamental", "hypothesis", "implication", "inherent",
    "interpretation", "methodology", "phenomenon",
    "principle", "proportion", "quantify", "significant",
    "substantial", "theoretical", "variable",
    "attainment", "comprehensive", "considerable",
    "controversial", "debate", "demonstrate", "extent",
    "influence", "investigation", "socioeconomic",
    "subsequently", "underlying",
    "a priori", "a posteriori",
}

TECHNICAL_VOCABULARY = {
    "carnivorous", "herbivorous", "scavenging",
    "trigonometry", "prosperity", "unemployment",
    "photosynthesis", "biodiversity", "ecosystem",
    "morphology", "neurological", "metabolism",
}

INTERMEDIATE_VOCABULARY = {
    "although", "circumstance", "comparison",
    "development", "effective", "environment",
    "experience", "independent", "opportunity",
    "relationship", "responsibility",
}

CONNECTORS = {
    "however", "nevertheless", "therefore",
    "furthermore", "moreover", "consequently",
    "nonetheless", "accordingly", "hence",
}

SUBORDINATORS = {
    "although", "because", "since", "unless",
    "whereas", "while", "though", "if", "when",
    "after", "before", "until", "once", "wherever",
}

PHRASAL_VERBS = {
    "carry out", "find out", "give up", "look into",
    "put off", "run into", "set up", "take over",
    "turn down", "work out",
}

PASSIVE_PATTERN = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+"
    r"(?:\w+ed|known|given|made|written|built|taken|"
    r"seen|shown|found|held|done|told|sent|bought)\b",
    re.I,
)

RELATIVE_PATTERN = re.compile(
    r"\b(?:who|whom|whose|which|that)\s+"
    r"(?:I|you|he|she|it|we|they|"
    r"the|a|an|this|that|these|those|"
    r"[A-Za-z]+)\s+"
    r"(?:am|is|are|was|were|has|have|had|"
    r"do|does|did|can|could|will|would|"
    r"should|may|might|must|[A-Za-z]+(?:ed|ing|s))\b",
    re.I,
)

INFINITIVE_PATTERN = re.compile(
    r"\bto\s+(?:learn|study|read|write|speak|understand|"
    r"improve|develop|explain|analyze|evaluate|compare|"
    r"consider|establish|demonstrate|quantify|discover|"
    r"create|solve|help|make|do|go|come|see|know|"
    r"be|have|get|take|give|find|keep|use|become)\b",
    re.I,
)

PARTICIPIAL_PATTERN = re.compile(
    r"\b(?:having|being|given|considered|established|"
    r"explained|involving|arriving|written|published|"
    r"built|made|taken|seen|shown|found|held|done|"
    r"told|sent|bought)\b",
    re.I,
)

CONDITIONAL_PATTERN = re.compile(
    r"\b(?:if|unless|provided that|as long as|"
    r"on condition that)\b",
    re.I,
)

CONCESSIVE_PATTERN = re.compile(
    r"\b(?:although|though|even though|even if|"
    r"despite|in spite of|whereas)\b",
    re.I,
)

COMPARATIVE_PATTERN = re.compile(
    r"\b(?:more|less|fewer|greater|better|worse)\b"
    r".{1,120}\bthan\b",
    re.I,
)

PERFECT_PATTERN = re.compile(
    r"\b(?:has|have|had)\s+"
    r"(?:been\b|(?:\w+ed|known|made|written|seen|"
    r"given|found|taken|done|shown)\b)",
    re.I,
)

EMBEDDED_PATTERN = re.compile(
    r"\b(?:idea|belief|claim|fact|assumption|thought|argument|"
    r"possibility|evidence|report|suggestion|conclusion|theory|"
    r"notion)\s+that\b"
    r"|\b(?:believe|claim|argue|suggest|think|establish|"
    r"demonstrate|assume|consider)\s+(?:that|whether)\b"
    r"|\b(?:has|have|had)\s+"
    r"(?:demonstrated|established|suggested|argued|shown|"
    r"proved|confirmed|reported)\s+that\b",
    re.I,
)


def _count_phrases(text: str, phrases: set[str]) -> int:
    """Count recognized multiword expressions."""
    return sum(
        len(re.findall(
            rf"(?<!\w){re.escape(phrase)}(?!\w)",
            text,
            re.I,
        ))
        for phrase in phrases
    )


def _count_independent_participials(text: str) -> int:
    """Count participial signals excluding recognized passive verb phrases."""
    without_passives = re.sub(
        r"\b(?:am|is|are|was|were|be|been|being|has|have|had)\s+"
        r"(?:\w+ed|known|given|made|written|built|taken|seen|shown|"
        r"found|held|done|told|sent|bought)\b",
        " ",
        text,
        flags=re.I,
    )
    return len(PARTICIPIAL_PATTERN.findall(without_passives))


def analyze_sentence(text: str) -> dict:
    """Extract interpretable lexical and grammatical signals."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    normalized = text.lower()
    words = WORD_PATTERN.findall(normalized)

    if not words:
        raise ValueError("No English word tokens found")

    counts = Counter(words)
    word_count = len(words)

    academic_hits = sum(
        word in ACADEMIC_VOCABULARY for word in words
    ) + _count_phrases(
        normalized, {"a priori", "a posteriori"}
    )
    technical_hits = sum(
        word in TECHNICAL_VOCABULARY for word in words
    )
    intermediate_hits = sum(
        word in INTERMEDIATE_VOCABULARY for word in words
    )

    return {
        "word_count": word_count,
        "unique_token_ratio": round(len(counts) / word_count, 3),
        "average_word_length": round(
            sum(map(len, words)) / word_count, 2
        ),
        "long_word_count": sum(len(word) >= 9 for word in words),
        "academic_vocabulary_hits": academic_hits,
        "technical_vocabulary_hits": technical_hits,
        "intermediate_vocabulary_hits": intermediate_hits,
        "connector_count": sum(
            word in CONNECTORS for word in words
        ),
        "subordinator_count": sum(
            word in SUBORDINATORS for word in words
        ),
        "passive_construction": bool(
            PASSIVE_PATTERN.search(normalized)
        ),
        "relative_clause_marker": bool(
            RELATIVE_PATTERN.search(normalized)
            and not re.search(
                r"\b(?:question|issue|matter|problem)\s+of\s+"
                r"(?:who|whom|whose|which|what)\b"
                r"|\b(?:idea|belief|claim|fact|assumption|thought|"
                r"argument|possibility|evidence|report|suggestion|"
                r"conclusion|theory|notion)\s+that\b"
                r"|\b(?:(?:has|have|had)\s+)?"
                r"(?:demonstrat(?:e|es|ed|ing)|"
                r"establish(?:es|ed|ing)?|"
                r"suggest(?:s|ed|ing)?|"
                r"argu(?:e|es|ed|ing)|"
                r"show(?:s|ed|ing)|"
                r"prov(?:e|es|ed|ing)|"
                r"confirm(?:s|ed|ing)|"
                r"report(?:s|ed|ing))\s+that\b",
                normalized,
                re.I,
            )
        ),
        "embedded_clause_marker": bool(
            EMBEDDED_PATTERN.search(normalized)
        ),
        "infinitive_construction_count": len(
            INFINITIVE_PATTERN.findall(normalized)
        ),
        "participial_construction_count": _count_independent_participials(normalized),
        "conditional_construction": bool(
            CONDITIONAL_PATTERN.search(normalized)
        ),
        "concessive_construction": bool(
            CONCESSIVE_PATTERN.search(normalized)
        ),
        "comparative_construction": bool(
            COMPARATIVE_PATTERN.search(normalized)
        ),
        "perfect_construction": bool(
            PERFECT_PATTERN.search(normalized)
        ),
        "phrasal_verb_hits": _count_phrases(
            normalized, PHRASAL_VERBS
        ),
        "punctuation_count": len(
            re.findall(r"[.!?;:,]", text)
        ),
    }

