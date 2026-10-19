"""Fail closed if estimator training leaks into the Pythia package."""
from __future__ import annotations

import ast
from pathlib import Path


PYTHIA_ROOT = Path(__file__).parents[1] / "pythia"
TRAINING_METHODS = {"fit", "fit_transform", "partial_fit", "train", "fine_tune"}
TRAINER_MODULE_PREFIXES = (
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.neighbors",
    "sklearn.pipeline",
    "xgboost",
    "lightgbm",
    "catboost",
    "torch",
    "tensorflow",
    "keras",
    "transformers",
)


def test_pythia_contains_no_model_training_calls_or_trainer_imports():
    violations: list[str] = []
    for path in sorted(PYTHIA_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in TRAINING_METHODS
            ):
                violations.append(f"{path.relative_to(PYTHIA_ROOT)}:{node.lineno} calls .{node.func.attr}()")
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                modules = []
            for module in modules:
                if module.startswith(TRAINER_MODULE_PREFIXES):
                    violations.append(
                        f"{path.relative_to(PYTHIA_ROOT)}:{node.lineno} imports trainer module {module}"
                    )
    assert not violations, "model training crossed the Pythia boundary:\n" + "\n".join(violations)
