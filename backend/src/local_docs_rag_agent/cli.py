from __future__ import annotations

import argparse
import sys

from local_docs_rag_agent.commands.ask import run_ask
from local_docs_rag_agent.commands.eval import run_eval_command
from local_docs_rag_agent.commands.eval_compare import run_eval_compare_command
from local_docs_rag_agent.commands.ingest import run_ingest
from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.exceptions import LocalDocsError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Docs RAG Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Read local docs and build the retrieval index")

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the docs index")
    ask_parser.add_argument(
        "--runtime",
        choices=["basic", "agents_sdk"],
        default=None,
        help="Override the answer runtime for this command",
    )
    ask_parser.add_argument("question", help="User question")

    eval_parser = subparsers.add_parser("eval", help="Run the sample eval harness")
    eval_parser.add_argument(
        "--runtime",
        choices=["basic", "agents_sdk"],
        default=None,
        help="Override the answer runtime for this command",
    )

    compare_parser = subparsers.add_parser(
        "eval-compare", help="Run eval across multiple retrieval/runtime configs"
    )
    compare_parser.add_argument(
        "--runtime",
        dest="runtimes",
        choices=["basic", "agents_sdk"],
        action="append",
        help="One or more runtimes to compare. Repeat the flag to compare multiple runtimes.",
    )
    compare_parser.add_argument(
        "--chunk-strategy",
        dest="chunk_strategies",
        choices=["fixed", "paragraph", "markdown"],
        action="append",
        help=(
            "One or more chunk strategies to compare. "
            "Repeat the flag to compare multiple strategies."
        ),
    )
    compare_parser.add_argument(
        "--vector-backend",
        dest="vector_backends",
        choices=["local", "qdrant"],
        action="append",
        help=(
            "One or more vector backends to compare. Repeat the flag to compare multiple backends."
        ),
    )
    compare_parser.add_argument(
        "--top-k",
        dest="top_ks",
        type=int,
        action="append",
        help="One or more top-k values to compare. Repeat the flag to compare multiple values.",
    )
    compare_parser.add_argument(
        "--chunk-size",
        dest="chunk_sizes",
        type=int,
        action="append",
        help="One or more chunk sizes to compare. Repeat the flag to compare multiple values.",
    )
    compare_parser.add_argument(
        "--chunk-overlap",
        dest="chunk_overlaps",
        type=int,
        action="append",
        help=(
            "One or more chunk overlap values to compare. "
            "Repeat the flag to compare multiple values."
        ),
    )
    compare_parser.add_argument(
        "--output",
        default="data/evals/compare_latest.json",
        help="Path to save the comparison report JSON",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        config = AppConfig.from_env()
        if args.command == "ingest":
            run_ingest(config)
            return

        if args.command == "ask":
            run_ask(config, question=args.question, runtime=args.runtime)
            return

        if args.command == "eval":
            run_eval_command(config, runtime=args.runtime)
            return

        if args.command == "eval-compare":
            run_eval_compare_command(
                config,
                runtimes=args.runtimes,
                chunk_strategies=args.chunk_strategies,
                vector_backends=args.vector_backends,
                top_ks=args.top_ks,
                chunk_sizes=args.chunk_sizes,
                chunk_overlaps=args.chunk_overlaps,
                output_path=args.output,
            )
            return

        parser.error(f"Unsupported command: {args.command}")
    except (LocalDocsError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
