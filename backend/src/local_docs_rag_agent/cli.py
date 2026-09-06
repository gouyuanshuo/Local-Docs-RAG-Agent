"""Command-line entry point for ingest, ask, eval, and eval-compare.

Each subcommand registers its own parser and binds a handler with
`parser.set_defaults(handler=...)`, so `main` never grows a branch per command:
it parses, then calls whichever handler the chosen subcommand supplied. Adding a
command means adding one `_register_*` function and listing it in
`COMMAND_REGISTRARS`.

Argument choices come from `constants`, which keeps `--runtime basic` and
`AGENT_RUNTIME=basic` describing the same set of values.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import constants, exceptions
from local_docs_rag_agent.commands import ask, eval_compare, ingest
from local_docs_rag_agent.commands import eval as eval_command

if TYPE_CHECKING:
    # argparse does not export a public alias for the object add_subparsers
    # returns.
    SubParsers = argparse._SubParsersAction[argparse.ArgumentParser]

CommandHandler = Callable[[app_config.AppConfig, argparse.Namespace], None]
CommandRegistrar = Callable[["SubParsers"], None]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse `argv`, run the chosen command, and return an exit code.

    Args:
      argv: Arguments to parse, or None to read `sys.argv`.

    Returns:
      0 on success, or 1 when the command raised an expected
      failure, which is printed to stderr rather than traced.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: CommandHandler = args.handler
    try:
        handler(app_config.AppConfig.from_env(), args)
    except (exceptions.LocalDocsError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with every subcommand attached.

    Returns:
      The parser. Every subcommand binds its handler with
      `set_defaults`, so `main` never grows a branch per command.
    """
    parser = argparse.ArgumentParser(
        prog="local-docs-rag",
        description="Local Docs RAG Agent CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for register in COMMAND_REGISTRARS:
        register(subparsers)
    return parser


def _register_ingest(subparsers: SubParsers) -> None:
    parser = subparsers.add_parser(
        "ingest",
        help="Read local docs and build the retrieval index",
    )
    parser.set_defaults(handler=_handle_ingest)


def _register_ask(subparsers: SubParsers) -> None:
    parser = subparsers.add_parser(
        "ask", help="Ask a question against the docs index"
    )
    _add_runtime_option(parser)
    parser.add_argument("question", help="User question")
    parser.set_defaults(handler=_handle_ask)


def _register_eval(subparsers: SubParsers) -> None:
    parser = subparsers.add_parser("eval", help="Run the sample eval harness")
    _add_runtime_option(parser)
    parser.set_defaults(handler=_handle_eval)


def _register_eval_compare(subparsers: SubParsers) -> None:
    parser = subparsers.add_parser(
        "eval-compare",
        help="Run eval across multiple retrieval/runtime configs",
    )
    _add_repeatable_choice_option(
        parser,
        "--runtime",
        dest="runtimes",
        choices=constants.AGENT_RUNTIMES,
        noun="runtimes",
    )
    _add_repeatable_choice_option(
        parser,
        "--chunk-strategy",
        dest="chunk_strategies",
        choices=constants.CHUNK_STRATEGIES,
        noun="chunk strategies",
    )
    _add_repeatable_choice_option(
        parser,
        "--vector-backend",
        dest="vector_backends",
        choices=constants.VECTOR_BACKENDS,
        noun="vector backends",
    )
    _add_repeatable_choice_option(
        parser,
        "--retrieval-strategy",
        dest="retrieval_strategies",
        choices=constants.RETRIEVAL_STRATEGIES,
        noun="retrieval strategies",
    )
    _add_repeatable_choice_option(
        parser,
        "--reranker",
        dest="rerankers",
        choices=constants.RERANKERS,
        noun="rerankers",
    )
    _add_repeatable_int_option(
        parser, "--top-k", dest="top_ks", noun="top-k values"
    )
    _add_repeatable_int_option(
        parser, "--chunk-size", dest="chunk_sizes", noun="chunk sizes"
    )
    _add_repeatable_int_option(
        parser,
        "--chunk-overlap",
        dest="chunk_overlaps",
        noun="chunk overlap values",
    )
    parser.add_argument(
        "--output",
        default=str(eval_compare.DEFAULT_COMPARE_OUTPUT_PATH),
        help="Path to save the comparison report JSON",
    )
    parser.set_defaults(handler=_handle_eval_compare)


COMMAND_REGISTRARS: tuple[CommandRegistrar, ...] = (
    _register_ingest,
    _register_ask,
    _register_eval,
    _register_eval_compare,
)


def _handle_ingest(
    config: app_config.AppConfig, args: argparse.Namespace
) -> None:
    del args
    ingest.run_ingest(config)


def _handle_ask(config: app_config.AppConfig, args: argparse.Namespace) -> None:
    ask.run_ask(config, question=args.question, runtime=args.runtime)


def _handle_eval(
    config: app_config.AppConfig, args: argparse.Namespace
) -> None:
    eval_command.run_eval_command(config, runtime=args.runtime)


def _handle_eval_compare(
    config: app_config.AppConfig, args: argparse.Namespace
) -> None:
    eval_compare.run_eval_compare_command(
        config,
        runtimes=args.runtimes,
        chunk_strategies=args.chunk_strategies,
        vector_backends=args.vector_backends,
        top_ks=args.top_ks,
        chunk_sizes=args.chunk_sizes,
        chunk_overlaps=args.chunk_overlaps,
        retrieval_strategies=args.retrieval_strategies,
        rerankers=args.rerankers,
        output_path=args.output,
    )


def _add_runtime_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime",
        choices=list(constants.AGENT_RUNTIMES),
        default=None,
        help="Override the answer runtime for this command",
    )


def _add_repeatable_choice_option(
    parser: argparse.ArgumentParser,
    flag: str,
    *,
    dest: str,
    choices: tuple[str, ...],
    noun: str,
) -> None:
    parser.add_argument(
        flag,
        dest=dest,
        choices=list(choices),
        action="append",
        help=(
            f"One or more {noun} to compare. "
            "Repeat the flag to compare several."
        ),
    )


def _add_repeatable_int_option(
    parser: argparse.ArgumentParser,
    flag: str,
    *,
    dest: str,
    noun: str,
) -> None:
    parser.add_argument(
        flag,
        dest=dest,
        type=int,
        action="append",
        help=(
            f"One or more {noun} to compare. "
            "Repeat the flag to compare several."
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
