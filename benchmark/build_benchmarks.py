"""Write packaged benchmark CSVs into benchmark/ (at least 500 rows each)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, make_friedman1

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark"
N_REG = 800
SEED = 42


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(frame) < 500:
        raise ValueError(f"{path.name} has only {len(frame)} rows; need at least 500.")
    frame.to_csv(path, index=False)
    print(f"wrote {path.relative_to(ROOT)}  ({len(frame)} rows × {frame.shape[1]} cols)")


def friedman1() -> pd.DataFrame:
    x, y = make_friedman1(n_samples=N_REG, n_features=10, noise=1.0, random_state=SEED)
    cols = {f"x{i}": x[:, i] for i in range(x.shape[1])}
    cols["y"] = y
    return pd.DataFrame(cols)


def ishigami() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    x1, x2, x3 = rng.uniform(-np.pi, np.pi, size=(3, N_REG))
    y = np.sin(x1) + 7.0 * np.sin(x2) ** 2 + 0.1 * x3**4 * np.sin(x1)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


def breast_cancer() -> pd.DataFrame:
    bundle = load_breast_cancer()
    cols = [name.replace(" ", "_").replace("(", "").replace(")", "") for name in bundle.feature_names]
    frame = pd.DataFrame(bundle.data, columns=cols)
    frame["label"] = bundle.target
    return frame


def digits() -> pd.DataFrame:
    bundle = load_digits()
    cols = [f"pixel_{i:02d}" for i in range(bundle.data.shape[1])]
    frame = pd.DataFrame(bundle.data, columns=cols)
    frame["label"] = bundle.target
    return frame


def main() -> None:
    _write(OUT / "regression" / "friedman1.csv", friedman1())
    _write(OUT / "regression" / "ishigami.csv", ishigami())
    _write(OUT / "classification" / "breast_cancer.csv", breast_cancer())
    _write(OUT / "classification" / "digits.csv", digits())


if __name__ == "__main__":
    main()
