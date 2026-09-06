from __future__ import annotations

import argparse

import pytest

from local_docs_rag_agent import cli, constants, exceptions
from local_docs_rag_agent import config as app_config


@pytest.mark.parametrize(
    ("argv", "expected_handler"),
    [
        (["ingest"], cli._handle_ingest),
        (["ask", "a question"], cli._handle_ask),
        (["eval"], cli._handle_eval),
        (["eval-compare"], cli._handle_eval_compare),
    ],
)
def test_every_subcommand_binds_a_handler(
    argv: list[str], expected_handler: object
) -> None:
    args = cli.build_parser().parse_args(argv)

    assert args.handler is expected_handler


def test_runtime_choices_track_the_shared_option_set() -> None:
    parser = cli.build_parser()

    for argv in (
        ["ask", "--runtime", "agents_sdk", "q"],
        ["eval", "--runtime", "agents_sdk"],
    ):
        assert parser.parse_args(argv).runtime == "agents_sdk"

    with pytest.raises(SystemExit):
        parser.parse_args(["ask", "--runtime", "telepathy", "q"])

    assert set(constants.AGENT_RUNTIMES) == {"basic", "agents_sdk"}


def test_compare_axes_are_repeatable_and_constrained() -> None:
    args = cli.build_parser().parse_args(
        [
            "eval-compare",
            "--chunk-strategy",
            "fixed",
            "--chunk-strategy",
            "markdown",
            "--vector-backend",
            "local",
            "--top-k",
            "3",
            "--top-k",
            "6",
        ]
    )

    assert args.chunk_strategies == ["fixed", "markdown"]
    assert args.vector_backends == ["local"]
    assert args.top_ks == [3, 6]
    assert set(args.chunk_strategies) <= set(constants.CHUNK_STRATEGIES)
    assert set(args.vector_backends) <= set(constants.VECTOR_BACKENDS)


def test_unset_compare_axes_stay_none_so_defaults_apply() -> None:
    args = cli.build_parser().parse_args(["eval-compare"])

    assert args.runtimes is None
    assert args.chunk_strategies is None
    assert args.vector_backends is None


def test_expected_failure_becomes_exit_code_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(config: app_config.AppConfig, args: argparse.Namespace) -> None:
        del config, args
        raise exceptions.ConfigurationError("DOCS_DIR does not exist: missing")

    monkeypatch.setattr(cli, "_handle_ingest", fail)
    monkeypatch.setattr(cli, "COMMAND_REGISTRARS", (cli._register_ingest,))

    exit_code = cli.main(["ingest"])

    assert exit_code == 1
    assert "DOCS_DIR does not exist" in capsys.readouterr().err


def test_successful_command_returns_exit_code_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def succeed(config: app_config.AppConfig, args: argparse.Namespace) -> None:
        del config, args
        calls.append("ran")

    monkeypatch.setattr(cli, "_handle_ingest", succeed)
    monkeypatch.setattr(cli, "COMMAND_REGISTRARS", (cli._register_ingest,))

    assert cli.main(["ingest"]) == 0
    assert calls == ["ran"]
