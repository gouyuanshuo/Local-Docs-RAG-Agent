from __future__ import annotations

import argparse

from local_docs_rag_agent.commands.ask import run_ask
from local_docs_rag_agent.commands.eval import run_eval_command
from local_docs_rag_agent.commands.ingest import run_ingest
from local_docs_rag_agent.config import AppConfig


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

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
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

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
