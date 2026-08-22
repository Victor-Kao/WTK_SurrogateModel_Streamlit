"""Catalog and loaders for packaged benchmark CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"


@dataclass(frozen=True)
class BenchmarkProblem:
    id: str
    title: str
    task: str
    kind: str
    relative_path: str
    target: str
    description: str

    @property
    def path(self) -> Path:
        return BENCHMARK_DIR / self.relative_path


PROBLEMS: dict[str, tuple[BenchmarkProblem, ...]] = {
    "regression": (
        BenchmarkProblem(
            id="friedman1",
            title="Friedman #1",
            task="regression",
            kind="continuous",
            relative_path="regression/friedman1.csv",
            target="y",
            description="Classic Friedman (1991) test function: 10 inputs on [0, 1], nonlinear interactions, 800 samples.",
        ),
        BenchmarkProblem(
            id="ishigami",
            title="Ishigami function",
            task="regression",
            kind="continuous",
            relative_path="regression/ishigami.csv",
            target="y",
            description="Standard UQ / surrogate benchmark: y = sin(x1) + 7 sin²(x2) + 0.1 x3⁴ sin(x1), 800 samples.",
        ),
    ),
    "binary_classification": (
        BenchmarkProblem(
            id="breast_cancer",
            title="Wisconsin Breast Cancer (binary)",
            task="binary_classification",
            kind="binary",
            relative_path="classification/breast_cancer.csv",
            target="label",
            description="UCI / scikit-learn Wisconsin diagnostic set: 30 cell-nucleus features, 2 classes, 569 samples.",
        ),
    ),
    "multiclass_classification": (
        BenchmarkProblem(
            id="digits",
            title="Handwritten Digits (multiclass)",
            task="multiclass_classification",
            kind="multiclass",
            relative_path="classification/digits.csv",
            target="label",
            description="UCI / scikit-learn 8×8 digit images: 64 pixel features, 10 classes (0–9), 1,797 samples.",
        ),
    ),
}


def list_problems(task: str) -> tuple[BenchmarkProblem, ...]:
    if task == "classification":
        return PROBLEMS["binary_classification"] + PROBLEMS["multiclass_classification"]
    return PROBLEMS[task]


def load_problem(problem: BenchmarkProblem) -> pd.DataFrame:
    if not problem.path.is_file():
        raise FileNotFoundError(
            f"Missing benchmark file: {problem.path}. Run python benchmark/build_benchmarks.py"
        )
    return pd.read_csv(problem.path)
