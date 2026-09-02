"""Answer runtimes and the dispatcher that chooses between them.

Callers use `answer_question`; which runtime serves the request, and whether it had to
delegate to another one, is reported in the answer's diagnostics rather than decided by
the caller.
"""

from local_docs_rag_agent.runtime.dispatch import answer_question

__all__ = ["answer_question"]
