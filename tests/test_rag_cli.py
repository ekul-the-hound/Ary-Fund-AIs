"""
tests/test_rag_cli.py
=====================
Smoke tests for ``python -m rag --help`` and friends.

The CLI is a thin router: every subcommand delegates to an entry
point that already exists elsewhere in the package. These tests
verify the router's contract:

* ``--help`` exits cleanly with the documented usage
* no-args prints usage and exits 0 (not a crash)
* subcommand ``--help`` works
* missing-dependency paths exit non-zero with a readable error,
  NOT a traceback

We don't test the actual indexing / benchmarking behaviour here —
those have their own tests in the modules they delegate to.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> subprocess.CompletedProcess:
    """Spawn a subprocess so each invocation gets a fresh interpreter.

    Subprocess (not in-process import) is the right test boundary
    because the CLI's lazy-import discipline only matters when the
    process starts cold.
    """
    return subprocess.run(
        [sys.executable, "-m", "rag", *args],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestCliHelp:

    def test_help_exits_zero(self):
        r = _run(["--help"])
        assert r.returncode == 0, (
            f"--help should exit 0, got {r.returncode}\nstderr: {r.stderr}"
        )

    def test_help_mentions_all_subcommands(self):
        r = _run(["--help"])
        out = r.stdout
        for sub in ("index", "stats", "benchmark"):
            assert sub in out, f"--help missing subcommand {sub!r}"

    def test_help_shows_examples(self):
        r = _run(["--help"])
        assert "Examples" in r.stdout or "examples" in r.stdout.lower()

    def test_no_args_prints_help_and_exits_zero(self):
        r = _run([])
        assert r.returncode == 0
        # Same usage banner as --help
        assert "usage:" in r.stdout.lower()

    def test_short_help_flag_works(self):
        r = _run(["-h"])
        assert r.returncode == 0
        assert "usage:" in r.stdout.lower()


class TestSubcommandHelp:

    def test_index_help_works(self):
        r = _run(["index", "--help"])
        assert r.returncode == 0, r.stderr
        assert "--tickers" in r.stdout

    def test_stats_help_works(self):
        r = _run(["stats", "--help"])
        assert r.returncode == 0, r.stderr
        assert "--tracking-db" in r.stdout

    def test_index_without_tickers_errors(self):
        # argparse should reject this with exit code 2.
        r = _run(["index"])
        assert r.returncode == 2
        # argparse writes the error to stderr
        assert "tickers" in (r.stderr or "").lower()


class TestSubcommandRouting:
    """The CLI exists to ROUTE to existing entry points. We don't run
    the real indexer here (no Ollama/Chroma in CI), but we DO verify
    that the router fails gracefully when those deps are missing —
    exit non-zero with a readable message, never a Python traceback."""

    def test_stats_without_deps_exits_cleanly(self):
        # `rag.indexer` and friends are not in this test rig.
        r = _run(["stats"])
        # Exit non-zero is correct (the subcommand can't actually run).
        assert r.returncode != 0
        # But the error must be a readable message, not an uncaught
        # ImportError traceback dumped to stderr.
        assert "Traceback" not in r.stderr, (
            f"router leaked a traceback:\n{r.stderr}"
        )

    def test_unknown_subcommand_errors(self):
        r = _run(["bogus"])
        assert r.returncode != 0
