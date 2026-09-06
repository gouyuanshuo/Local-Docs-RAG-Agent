"""Okapi BM25 ranking over a chunk corpus.

The lexical signal in :mod:`local_docs_rag_agent.rag.scoring` is a bare set
intersection: it counts how many distinct query terms a chunk contains. That
treats every term as equally informative, so a chunk matching only "the" and
"model" scores the same as one matching the query's rarest and most identifying
term, and it ignores how often a term appears at all.

BM25 repairs both halves. Inverse document frequency makes a term that occurs in
few chunks worth more than one that occurs in most of them, and term-frequency
saturation means the fifth occurrence of a word adds far less than the first, so
a long chunk cannot win by repetition alone.

The index is rebuilt from the corpus it ranks rather than persisted, because
document frequencies are a property of the corpus as it stands: a stale index
would weight terms by documents that are no longer in it.
"""

from __future__ import annotations

import collections
import dataclasses
import math

from local_docs_rag_agent.rag import scoring

# Saturation and length-normalization constants. These are the values BM25 is
# almost always deployed with, and they are named here so a future tuning sweep
# has something to vary rather than a literal buried in the scoring expression.
K1 = 1.5
B = 0.75


@dataclasses.dataclass(slots=True)
class Bm25Index:
    """Term statistics for one corpus, used to rank it against a query."""

    term_counts: list[collections.Counter[str]]
    document_lengths: list[int]
    document_frequencies: collections.Counter[str]
    average_length: float

    @classmethod
    def build(cls, texts: list[str]) -> Bm25Index:
        """Index `texts` positionally, so a rank refers back by index.

        Args:
          texts: The corpus to index.

        Returns:
          The index, holding this corpus's term statistics.
        """
        term_counts = [
            collections.Counter(scoring.tokenize_terms(text)) for text in texts
        ]
        document_lengths = [sum(counts.values()) for counts in term_counts]
        document_frequencies: collections.Counter[str] = collections.Counter()
        for counts in term_counts:
            document_frequencies.update(counts.keys())
        total_length = sum(document_lengths)
        average_length = total_length / len(texts) if texts else 0.0
        return cls(
            term_counts=term_counts,
            document_lengths=document_lengths,
            document_frequencies=document_frequencies,
            average_length=average_length,
        )

    def score(self, query_terms: list[str], document_index: int) -> float:
        """Return one indexed document's BM25 score for `query_terms`."""
        if self.average_length <= 0:
            return 0.0
        counts = self.term_counts[document_index]
        length = self.document_lengths[document_index]
        total = 0.0
        for term in query_terms:
            term_frequency = counts.get(term, 0)
            if term_frequency == 0:
                continue
            normalized_length = 1 - B + B * (length / self.average_length)
            saturated = (term_frequency * (K1 + 1)) / (
                term_frequency + K1 * normalized_length
            )
            total += self._inverse_document_frequency(term) * saturated
        return total

    def rank(self, query: str) -> list[tuple[int, float]]:
        """Return `(index, score)` for every scoring document, best first."""
        query_terms = scoring.tokenize_terms(query)
        if not query_terms:
            return []
        scored = [
            (index, self.score(query_terms, index))
            for index in range(len(self.term_counts))
        ]
        # Ties break on the document index so the same corpus always ranks the
        # same way, which is what lets an eval run be compared against an
        # earlier one.
        return sorted(
            (entry for entry in scored if entry[1] > 0),
            key=lambda entry: (-entry[1], entry[0]),
        )

    def _inverse_document_frequency(self, term: str) -> float:
        # The `1 +` inside the logarithm is the Lucene variant: it keeps the
        # weight of a term appearing in more than half the corpus small but
        # never negative, so a common term can dilute a ranking without actively
        # reversing it.
        num_documents = len(self.term_counts)
        frequency = self.document_frequencies.get(term, 0)
        return math.log(
            1 + (num_documents - frequency + 0.5) / (frequency + 0.5)
        )
