"""Incremental ingest pipeline: discover, plan, chunk, embed, and commit.

The step order is deliberate and is what makes a partial or repeated run safe:

1. validate and bind the canonical manifest and storage targets
2. acquire both reentrant thread and cross-process lifecycle resources
3. read every source document, so an unreadable file aborts before mutation
4. load the typed manifest recorded by the previous run
5. verify storage ownership, then compare the scope/retrieval fingerprint and
   work out removed and changed sources
6. chunk only the sources that actually need reindexing
7. attach embeddings and reject empty or mismatched vectors
8. durably publish repair intent, then commit the configured store
9. atomically publish the clean manifest

The fingerprint covers document scope plus every setting that changes what a
stored vector *means*. A separate non-secret hash binds source ownership to the
local index path or Qdrant endpoint and collection, so a target change cannot
replay another target's deletion list.
"""

from __future__ import annotations

import dataclasses
import hashlib
import ipaddress
import json
import pathlib
import re
from urllib import parse

import idna
from urllib3 import util as urllib3_util

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.core import exceptions, models
from local_docs_rag_agent.providers import base as provider_base
from local_docs_rag_agent.providers import factory as provider_factory
from local_docs_rag_agent.rag import (
    chunker,
    discovery,
    index_lock,
    manifest,
    store_factory,
)


@dataclasses.dataclass(frozen=True, slots=True)
class IngestPlan:
    """The work one ingest run has to do, decided before anything is written."""

    removed_sources: tuple[str, ...]
    changed_sources: tuple[str, ...]
    sources_to_index: tuple[str, ...]


def ingest_documents(
    config: app_config.AppConfig,
) -> list[models.DocumentChunk]:
    """Bring the index up to date and return the chunks written.

    Args:
      config: The settings to ingest under.

    Returns:
      Every chunk written by this run, which for an incremental
      backend is only the chunks of the documents that changed.

    Raises:
      ConfigurationError: If the documents cannot be read, the storage target
        is invalid, or the manifest cannot safely own the configured target.
      ProviderUnavailableError: If the provider returns the wrong number
        of vectors or an empty one, or if a store that requires live
        embeddings receives fallback vectors.
      VectorStoreError: If the store rejects the write.
    """
    with index_lock._bound_index_guard(config) as targets:
        return _ingest_documents_locked(targets)


def _ingest_documents_locked(
    targets: index_lock._BoundIndexTargets,
) -> list[models.DocumentChunk]:
    config = targets.config
    manifest_path = targets.manifest_path
    source_texts = discovery.read_source_texts(
        config.docs_dir, config.docs_exclude_patterns
    )
    source_checksums = {
        source_path: source_checksum(text)
        for source_path, text in source_texts.items()
    }
    previous_manifest = manifest.IngestManifest.load(manifest_path)
    desired_storage_identity = targets.storage_identity
    legacy_rebuild = _validate_manifest_storage(
        config,
        previous_manifest,
        desired_storage_identity,
    )
    embedding_provider = provider_factory.build_embedding_provider(config)
    store = store_factory._build_bound_store(
        targets, embedding_provider=embedding_provider
    )
    desired_fingerprint = index_fingerprint(
        config, embedding_mode=_expected_embedding_mode(config)
    )
    plan = _build_ingest_plan(
        source_checksums=source_checksums,
        previous_manifest=previous_manifest,
        index_all=(
            not store.supports_incremental_updates
            or legacy_rebuild
            or previous_manifest.repair_required
            or not store.exists()
            or previous_manifest.index_fingerprint != desired_fingerprint
        ),
    )

    chunks = _chunk_sources(config, source_texts, plan.sources_to_index)
    _attach_embeddings(chunks, embedding_provider)
    if store.requires_live_embeddings and chunks:
        _require_live_embeddings(embedding_provider)

    dirty_manifest = previous_manifest.marked_needs_reindex(
        plan.sources_to_index,
        storage_identity=desired_storage_identity,
    )
    dirty_manifest.save(manifest_path)
    store.save(
        chunks,
        removed_source_paths=(
            list(plan.removed_sources)
            if store.supports_incremental_updates
            else None
        ),
        # Include changed-to-empty sources so an incremental store deletes
        # their stale records.
        replaced_source_paths=(
            list(plan.sources_to_index)
            if store.supports_incremental_updates
            else None
        ),
    )
    dirty_manifest.updated(
        source_checksums=source_checksums,
        indexed_sources=set(plan.sources_to_index),
        chunks=chunks,
        index_fingerprint=index_fingerprint(
            config,
            # With no chunks the provider was never exercised, so record the
            # intent.
            embedding_mode=(
                embedding_provider.status.mode
                if chunks
                else _expected_embedding_mode(config)
            ),
        ),
        storage_identity=desired_storage_identity,
    ).save(manifest_path)
    return chunks


def ensure_index(config: app_config.AppConfig) -> None:
    """Ingest only if the index is missing, stale, or marked dirty.

    Ask and eval call this rather than `ingest_documents`. An incremental
    save that deleted records and then failed to replace them records those
    sources in `needs_reindex`; skipping that flag would leave Ask serving
    an index the manifest still calls complete.

    Args:
      config: The settings whose index should be present.

    Raises:
      ConfigurationError: If the storage target is invalid, differs from the
        manifest, or a legacy Qdrant manifest needs an ownership decision.
    """
    with index_lock._bound_index_guard(config) as targets:
        _ensure_index_locked(targets)


def _ensure_index_locked(
    targets: index_lock._BoundIndexTargets,
) -> None:
    config = targets.config
    manifest_path = targets.manifest_path
    stored = manifest.IngestManifest.load(manifest_path)
    desired_storage_identity = targets.storage_identity
    legacy_rebuild = _validate_manifest_storage(
        config,
        stored,
        desired_storage_identity,
    )
    configuration_changed = (
        legacy_rebuild
        or stored.index_fingerprint
        != index_fingerprint(
            config,
            embedding_mode=_expected_embedding_mode(config),
        )
    )
    needs_restore = stored.repair_required or bool(stored.needs_reindex)
    store = store_factory._build_bound_store(targets)
    if configuration_changed or needs_restore or not store.exists():
        _ingest_documents_locked(targets)


def source_checksum(text: str) -> str:
    """Return the content hash used to detect a changed source document."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_fingerprint(
    config: app_config.AppConfig, *, embedding_mode: str
) -> str:
    """Return a hash of document scope and vector-defining settings."""
    payload = {
        "docs_dir": str(config.docs_dir.resolve()),
        "docs_exclude_patterns": sorted(set(config.docs_exclude_patterns)),
        "vector_backend": config.vector_backend,
        "embedding_provider": config.embedding_provider,
        "embedding_base_url": config.embedding_base_url,
        "embedding_model": config.embedding_model,
        "embedding_dimensions": config.embedding_dimensions,
        "embedding_mode": embedding_mode,
        "chunk_strategy": config.chunk_strategy,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def storage_identity(config: app_config.AppConfig) -> str:
    """Return a non-secret hash identifying the configured chunk store.

    Local storage is identified by its resolved index path. Qdrant storage is
    identified by a canonical endpoint and collection; URL user information,
    query parameters, fragments, and the API key are deliberately excluded.

    Args:
      config: The settings selecting the storage target.

    Returns:
      A SHA-256 digest of canonical, non-secret target data.

    Raises:
      ConfigurationError: If target semantics are invalid, including an
        unsupported backend or a Qdrant URL that is missing or malformed.
    """
    return _storage_identity_from_payload(_storage_target_payload(config))


def _bound_local_storage_identity(index_path: pathlib.Path) -> str:
    payload = _local_storage_target_payload(index_path)
    return _storage_identity_from_payload(payload)


def _storage_identity_from_payload(payload: dict[str, str]) -> str:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_storage_target(config: app_config.AppConfig) -> None:
    """Validate storage semantics without mutation or client construction.

    The check performs no filesystem inspection and does not acquire lifecycle
    locks, create providers or stores, contact Qdrant, or write local paths. A
    missing Qdrant URL is invalid here; callers that classify that
    configuration as skipped must decide before invoking this validator.

    Args:
      config: Settings selecting the storage and manifest targets.

    Raises:
      ConfigurationError: If the backend is unsupported or a Qdrant URL is
        missing or malformed.
    """
    if config.vector_backend == "local":
        return
    if config.vector_backend == "qdrant":
        if not config.qdrant_url:
            raise exceptions.ConfigurationError(
                "QDRANT_URL must be set when VECTOR_BACKEND=qdrant"
            )
        _canonical_qdrant_url(config.qdrant_url)
        return
    raise exceptions.ConfigurationError(
        f"VECTOR_BACKEND is unsupported: {config.vector_backend!r}"
    )


def _storage_target_payload(
    config: app_config.AppConfig,
) -> dict[str, str]:
    validate_storage_target(config)
    if config.vector_backend == "local":
        return _local_storage_target_payload(config.index_path.resolve())
    if config.vector_backend == "qdrant":
        if not config.qdrant_url:
            raise exceptions.ConfigurationError(
                "QDRANT_URL must be set when VECTOR_BACKEND=qdrant"
            )
        return {
            "vector_backend": "qdrant",
            "qdrant_url": _canonical_qdrant_url(config.qdrant_url),
            "qdrant_collection": config.qdrant_collection,
        }
    raise exceptions.ConfigurationError(
        f"VECTOR_BACKEND is unsupported: {config.vector_backend!r}"
    )


def _canonical_qdrant_url(url: str) -> str:
    try:
        if any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in url
        ):
            raise ValueError
        parsed = urllib3_util.parse_url(url)
        scheme = parsed.scheme.lower() if parsed.scheme else ""
        hostname = parsed.host
        port = parsed.port
        if scheme not in {"http", "https"} or hostname is None or port == 0:
            raise ValueError
        _reject_percent_encoded_dns_host(url, hostname)
        normalized_host = _canonical_qdrant_host(hostname)
        normalized_path = _canonical_qdrant_path(parsed.path or "")
    except (TypeError, UnicodeError, ValueError):
        raise exceptions.ConfigurationError(
            "QDRANT_URL must be a valid HTTP(S) endpoint"
        ) from None
    effective_port = port if port is not None else 6333
    netloc = f"{normalized_host}:{effective_port}"
    return parse.urlunsplit((scheme, netloc, normalized_path, "", ""))


def _canonical_qdrant_path(path: str) -> str:
    if path.startswith("//"):
        raise ValueError
    joined = parse.urljoin("https://identity.invalid/", path)
    normalized = parse.urlsplit(joined).path
    normalized = re.sub(r"/{2,}", "/", normalized)
    escapes = re.findall(r"%[0-9A-Fa-f]{2}", normalized)
    normalized = re.sub(
        r"%[0-9A-Fa-f]{2}",
        lambda match: match.group(0).upper(),
        normalized,
    )
    percent_is_encoded = len(escapes) == normalized.count("%")
    safe = "/!$&'()*+,-.:;=@_~"
    if percent_is_encoded:
        safe += "%"
    return parse.quote(normalized, safe=safe).rstrip("/")


def _reject_percent_encoded_dns_host(url: str, parsed_host: str) -> None:
    if parsed_host.startswith("["):
        return
    # urllib3 decodes percent escapes in DNS labels. Keep the project's
    # stricter raw-host policy without using this check to derive identity.
    _, separator, remainder = url.partition("://")
    if not separator:
        raise ValueError
    raw_authority = re.split(r"[/\\?#]", remainder, maxsplit=1)[0]
    raw_host_port = raw_authority.rpartition("@")[2]
    if "%" in raw_host_port:
        raise ValueError


def _local_storage_target_payload(
    index_path: pathlib.Path,
) -> dict[str, str]:
    return {
        "vector_backend": "local",
        "index_path": str(index_path),
    }


def _canonical_qdrant_host(hostname: str) -> str:
    if not hostname:
        raise ValueError
    if hostname.startswith("[") and hostname.endswith("]"):
        hostname = hostname[1:-1]
    if ":" in hostname:
        address = ipaddress.IPv6Address(hostname)
        return f"[{address.compressed}]"

    lowercase_hostname = hostname.lower()
    ascii_hostname = (
        lowercase_hostname
        if lowercase_hostname.isascii()
        else idna.encode(
            lowercase_hostname,
            strict=True,
            std3_rules=True,
        ).decode("ascii")
    )
    if not ascii_hostname or len(ascii_hostname) > 253:
        raise ValueError
    labels = ascii_hostname.split(".")
    if any(
        not label
        or len(label) > 63
        or not label[0].isalnum()
        or not label[-1].isalnum()
        or any(
            not character.isalnum() and character != "-" for character in label
        )
        for label in labels
    ):
        raise ValueError
    if all(
        character.isdigit() or character == "." for character in ascii_hostname
    ):
        return str(ipaddress.IPv4Address(ascii_hostname))
    return ascii_hostname


def _validate_manifest_storage(
    config: app_config.AppConfig,
    previous_manifest: manifest.IngestManifest,
    desired_storage_identity: str,
) -> bool:
    if previous_manifest.version == manifest.LEGACY_VERSION:
        if config.vector_backend == "qdrant":
            raise exceptions.ConfigurationError(
                "A legacy Qdrant manifest cannot prove target ownership",
                action_hint=(
                    "Verify ownership of the configured collection. To adopt "
                    "it, clear the confirmed-owned collection, move the "
                    "legacy manifest aside, and ingest with a dedicated "
                    "INGEST_MANIFEST_PATH."
                ),
            )
        return True

    known_identity = previous_manifest.storage_identity
    if known_identity is None:
        has_owned_state = bool(
            previous_manifest.sources
            or previous_manifest.index_fingerprint
            or previous_manifest.needs_reindex
            or previous_manifest.repair_required
        )
        if config.vector_backend == "qdrant" and has_owned_state:
            raise exceptions.ConfigurationError(
                "The Qdrant manifest has no storage identity",
                action_hint=(
                    "Verify target ownership, then use a dedicated "
                    "INGEST_MANIFEST_PATH for the configured collection."
                ),
            )
        return has_owned_state
    if known_identity != desired_storage_identity:
        raise exceptions.ConfigurationError(
            "The ingest manifest belongs to a different storage target",
            action_hint=(
                "Use a dedicated INGEST_MANIFEST_PATH for this target. "
                "To adopt it instead, first verify ownership and clear the "
                "target, then move the existing manifest aside."
            ),
        )
    return False


def _expected_embedding_mode(config: app_config.AppConfig) -> str:
    return "live" if config.embedding_api_key else "fallback"


def _build_ingest_plan(
    *,
    source_checksums: dict[str, str],
    previous_manifest: manifest.IngestManifest,
    index_all: bool,
) -> IngestPlan:
    previous_sources = previous_manifest.sources
    dirty_sources = set(previous_manifest.needs_reindex)
    owned_sources = set(previous_sources) | dirty_sources
    removed_sources = tuple(sorted(owned_sources - set(source_checksums)))
    changed_sources = tuple(
        sorted(
            source_path
            for source_path, checksum in source_checksums.items()
            if source_path not in previous_sources
            or previous_sources[source_path].checksum != checksum
            or source_path in dirty_sources
        )
    )
    sources_to_index = (
        tuple(sorted(source_checksums)) if index_all else changed_sources
    )
    return IngestPlan(
        removed_sources=removed_sources,
        changed_sources=changed_sources,
        sources_to_index=sources_to_index,
    )


def _chunk_sources(
    config: app_config.AppConfig,
    source_texts: dict[str, str],
    source_paths: tuple[str, ...],
) -> list[models.DocumentChunk]:
    chunks: list[models.DocumentChunk] = []
    for source_path in source_paths:
        chunks.extend(
            chunker.chunk_text(
                source_path=pathlib.Path(source_path),
                text=source_texts[source_path],
                chunk_strategy=config.chunk_strategy,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        )
    return chunks


def _attach_embeddings(
    chunks: list[models.DocumentChunk],
    embedding_provider: provider_base.EmbeddingProvider,
) -> None:
    if not chunks:
        return
    embeddings = embedding_provider.embed_texts(
        [chunk.text for chunk in chunks]
    )
    if len(embeddings) != len(chunks):
        raise exceptions.ProviderUnavailableError(
            "Embedding provider returned a different number of vectors "
            "than input chunks",
            action_hint=(
                f"Expected {len(chunks)} vectors, received {len(embeddings)}."
            ),
        )
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        if not embedding:
            raise exceptions.ProviderUnavailableError(
                "Embedding provider returned an empty vector for "
                f"{chunk.chunk_id}"
            )
        chunk.embedding = embedding


def _require_live_embeddings(
    embedding_provider: provider_base.EmbeddingProvider,
) -> None:
    # A store may compare incoming vectors with live-provider vectors already
    # persisted there. Writing fallback vectors would quietly corrupt retrieval
    # quality for such a store.
    status = embedding_provider.status
    if status.mode == "live":
        return
    reason = status.reason or "unknown_reason"
    action_hint = (
        "Try ingest again or increase EMBEDDING_MAX_RETRIES and "
        "EMBEDDING_RETRY_BACKOFF_MS."
        if reason.startswith("provider_transient_error:")
        else "Verify EMBEDDING_API_KEY, EMBEDDING_BASE_URL, and "
        "EMBEDDING_MODEL."
    )
    raise exceptions.ProviderUnavailableError(
        "The configured vector store requires a live embedding provider",
        action_hint=(
            f"Provider {status.provider!r} is in {status.mode!r} mode "
            f"({reason}). {action_hint}"
        ),
    )
