"""Chat and embedding providers behind a common interface.

Import `build_chat_provider` and `build_embedding_provider` from here; the
concrete classes are implementation detail and may be replaced per provider.
"""

from local_docs_rag_agent.providers.factory import (
    build_chat_provider,
    build_embedding_provider,
)

__all__ = ["build_chat_provider", "build_embedding_provider"]
