"""Read-only connectivity check for the configured Qdrant collection.

Reports whether the collection already exists without writing to it, so it is safe
to run against a shared deployment.
This is not a pytest module; run it directly with `python scripts/check_qdrant.py`.
"""

from __future__ import annotations

import sys

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.exceptions import ConfigurationError, LocalDocsError
from local_docs_rag_agent.rag import QdrantChunkStore, build_store


def main() -> int:
    try:
        config = AppConfig.from_env()
        store = build_store(config)
        if not isinstance(store, QdrantChunkStore):
            raise ConfigurationError("Set VECTOR_BACKEND=qdrant before running this check")
        print(f"collection={config.qdrant_collection}")
        collection_exists = store.collection_exists()
    except LocalDocsError as exc:
        print(f"status=failed error={exc}", file=sys.stderr)
        return 1

    print(f"collection_exists={collection_exists}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
