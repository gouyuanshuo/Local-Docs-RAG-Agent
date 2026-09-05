"""Reciprocal Rank Fusion for combining rankings that share no common scale.

The original local scoring blends its dense and lexical signals with ``max()``, which
is only meaningful if the two numbers mean the same thing — and they do not. A cosine
similarity and a term-overlap ratio both land in [0, 1] while measuring different
things, so whichever happens to be numerically larger wins regardless of how confident
either signal actually is. Worse, the winning branch's scale then sets the scale of the
whole result, and two chunks picked by different branches are no longer comparable.

Reciprocal Rank Fusion sidesteps calibration entirely by throwing the scores away and
fusing positions instead. A chunk both signals place near the top outranks one that a
single signal loves, which is the behaviour hybrid retrieval is actually after.

`k` damps the advantage of the very top positions: without it, one signal's first place
would dominate any amount of agreement further down. 60 is the value from the original
paper and remains the common default.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked key lists into one ranking, best first.

    Each ranking contributes `1 / (rrf_k + position)` to every key it contains, so a key
    absent from a ranking is simply not scored by it rather than being penalised with a
    zero — which matters because the branches rank different candidate sets.
    """

    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + position)
    # Ties break on the key so a fused ranking is reproducible across runs.
    return sorted(fused.items(), key=lambda entry: (-entry[1], entry[0]))
