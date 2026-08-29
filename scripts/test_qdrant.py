from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.rag.ingest import build_store
from local_docs_rag_agent.rag.store import QdrantChunkStore


def main() -> None:
    config = AppConfig.from_env()
    store = build_store(config)
    if not isinstance(store, QdrantChunkStore):
        raise RuntimeError("Set VECTOR_BACKEND=qdrant before running this check.")
    print(f"collection={config.qdrant_collection}")
    try:
        collection_exists = store.collection_exists()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    print(f"collection_exists={collection_exists}")


if __name__ == "__main__":
    main()
