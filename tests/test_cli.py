"""CLI wiring (no heavy training)."""

import pytest

from compassmind.cli import main


def test_main_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
