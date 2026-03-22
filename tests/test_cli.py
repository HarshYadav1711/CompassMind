"""CLI wiring (no heavy training)."""

import pandas as pd
import pytest

from compassmind.cli import main


def test_main_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_summarize_prints_value_counts(tmp_path, capsys):
    p = tmp_path / "pred.csv"
    pd.DataFrame(
        {"predicted_intensity": [2, 3, 3, 4], "id": [1, 2, 3, 4]}
    ).to_csv(p, index=False)
    main(["summarize", "--csv", str(p)])
    out = capsys.readouterr().out
    assert "predicted_intensity" in out
    assert "2" in out and "3" in out
