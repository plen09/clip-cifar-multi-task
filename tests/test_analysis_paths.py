"""Analysis entry points must discover the project after directory cleanup."""
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", [
    "analyze_preliminary.py",
    "analyze_perturbations.py",
    "analyze_hetero.py",
])
def test_analysis_root_is_project_root(script):
    project_root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(project_root / "analysis" / script), run_name="path_audit"
    )
    assert namespace["ROOT"] == project_root
