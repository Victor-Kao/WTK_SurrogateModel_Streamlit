"""Design-of-experiments sampling methods."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import qmc

METHODS = [
    "Latin Hypercube",
    "Sobol",
    "Halton",
    "Random Uniform",
    "Full Factorial",
]


def default_variables() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["x1", "x2", "x3"],
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 1.0, 1.0],
            "type": ["continuous", "continuous", "continuous"],
        }
    )


def _unit_sample(method: str, n_dim: int, n_samples: int, seed: int) -> np.ndarray:
    if method == "Latin Hypercube":
        sampler = qmc.LatinHypercube(d=n_dim, seed=seed)
        return sampler.random(n_samples)
    if method == "Sobol":
        sampler = qmc.Sobol(d=n_dim, scramble=True, seed=seed)
        return sampler.random(n_samples)
    if method == "Halton":
        sampler = qmc.Halton(d=n_dim, scramble=True, seed=seed)
        return sampler.random(n_samples)
    rng = np.random.default_rng(seed)
    return rng.random((n_samples, n_dim))


def _full_factorial(variables: pd.DataFrame, levels: int) -> np.ndarray:
    axes = [
        np.linspace(float(row["min"]), float(row["max"]), levels)
        for _, row in variables.iterrows()
    ]
    return np.array(list(product(*axes)), dtype=float)


def generate_samples(
    variables: pd.DataFrame,
    method: str,
    n_samples: int,
    seed: int = 42,
    levels: int = 3,
) -> pd.DataFrame:
    cleaned = variables.dropna(subset=["name"]).copy()
    cleaned.loc[:, "name"] = cleaned["name"].astype(str).str.strip()
    cleaned = cleaned.loc[cleaned["name"] != ""].copy()
    if cleaned.empty:
        raise ValueError("Add at least one named input variable.")
    if cleaned["name"].duplicated().any():
        raise ValueError("Variable names must be unique.")
    if (cleaned["max"] <= cleaned["min"]).any():
        raise ValueError("Each variable max must be greater than min.")

    names = cleaned["name"].tolist()
    lows = cleaned["min"].to_numpy(dtype=float)
    highs = cleaned["max"].to_numpy(dtype=float)

    if method == "Full Factorial":
        raw = _full_factorial(cleaned, levels)
    else:
        unit = _unit_sample(method, len(names), n_samples, seed)
        raw = qmc.scale(unit, lows, highs)

    frame = pd.DataFrame(raw, columns=names)
    for idx, var_type in enumerate(cleaned["type"].tolist()):
        if str(var_type).lower().startswith("int"):
            col = names[idx]
            frame[col] = np.rint(frame[col]).astype(int)
    frame.insert(0, "sample_id", np.arange(1, len(frame) + 1))
    return frame


AUGMENT_METHODS = [
    "Augmented Latin Hypercube Sampling (LHS)",
    "Greedy Maximin Distance",
    "Greedy Maximin Distance (boundary-weighted)",
]

AUGMENT_METHOD_NOTES: dict[str, str] = {
    "Augmented Latin Hypercube Sampling (LHS)": (
        "Constructs an augmented design of size N₀ + N_new within the specified "
        "exploration bounds. Existing samples are mapped to the unit hypercube and "
        "used to identify unoccupied one-dimensional LHS strata at the target size. "
        "New locations are allocated to those strata and refined to improve spatial "
        "dispersion while preserving approximate Latin hypercube balance. "
        "This method is generally preferred when uniform coverage and reduced "
        "boundary concentration are desired."
    ),
    "Greedy Maximin Distance": (
        "Sequentially augments the design by maximizing the minimum Euclidean "
        "distance to all currently accepted points (the original design plus any "
        "points already added). At each iteration, a large pool of random candidates "
        "is evaluated in the normalized design space, and the candidate with the "
        "largest nearest-neighbor distance is retained. The approach yields strong "
        "space-filling behavior, but solutions often concentrate near domain edges "
        "and corners of a bounded box."
    ),
    "Greedy Maximin Distance (boundary-weighted)": (
        "Extends greedy maximin selection with a boundary-aware objective: "
        "score = d_min × (d_boundary)^α, where d_min is the distance to the nearest "
        "accepted point and d_boundary is the distance to the closest face of the "
        "unit hypercube. The exponent α controls interior preference "
        "(α = 0 recovers classical maximin; larger α penalizes edge/corner placement). "
        "This retains iterative space-filling updates while mitigating boundary bias."
    ),
}


def design_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric design columns excluding sample_id."""
    cols = frame.drop(columns=["sample_id"], errors="ignore").select_dtypes(include="number").columns.tolist()
    return [str(c) for c in cols]


def design_bounds_from_frame(frame: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Default min/max table from an existing design matrix."""
    feats = features or design_feature_columns(frame)
    rows = []
    for name in feats:
        series = pd.to_numeric(frame[name], errors="coerce")
        rows.append({"name": name, "min": float(series.min()), "max": float(series.max())})
    return pd.DataFrame(rows)


def parse_bounds_csv(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Parse an uploaded bounds table into name / min / max rows for ``features``.

    Accepts flexible headers such as name/feature, min/lower, max/upper.
    """
    if frame is None or frame.empty:
        raise ValueError("Bounds CSV is empty.")

    rename: dict[str, str] = {}
    for col in frame.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in {"name", "feature", "variable", "param", "parameter"}:
            rename[col] = "name"
        elif key in {"min", "minimum", "lower", "low", "lb"}:
            rename[col] = "min"
        elif key in {"max", "maximum", "upper", "high", "ub"}:
            rename[col] = "max"
    parsed = frame.rename(columns=rename)
    required = {"name", "min", "max"}
    if not required.issubset(parsed.columns):
        raise ValueError(
            "Bounds CSV must include columns for feature name, min, and max "
            "(e.g. name,min,max)."
        )

    parsed = parsed[["name", "min", "max"]].copy()
    parsed.loc[:, "name"] = parsed["name"].astype(str).str.strip()
    parsed.loc[:, "min"] = pd.to_numeric(parsed["min"], errors="coerce")
    parsed.loc[:, "max"] = pd.to_numeric(parsed["max"], errors="coerce")
    if parsed["min"].isna().any() or parsed["max"].isna().any():
        raise ValueError("Bounds CSV contains non-numeric min/max values.")
    if (parsed["max"] <= parsed["min"]).any():
        raise ValueError("Each feature max must be greater than min in the bounds CSV.")

    by_name = {str(r["name"]): (float(r["min"]), float(r["max"])) for _, r in parsed.iterrows()}
    missing = [f for f in features if f not in by_name]
    if missing:
        raise ValueError(f"Bounds CSV is missing features: {', '.join(missing)}")

    rows = [{"name": f, "min": by_name[f][0], "max": by_name[f][1]} for f in features]
    return pd.DataFrame(rows)


def _to_unit(x: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    span = np.maximum(highs - lows, 1e-15)
    return (x - lows) / span


def _from_unit(u: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    return lows + u * (highs - lows)


def _pairwise_min_dist(candidates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """For each candidate row, minimum Euclidean distance to any reference row."""
    if reference.size == 0:
        return np.full(candidates.shape[0], np.inf)
    # (n_cand, n_ref)
    diff = candidates[:, None, :] - reference[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return dist.min(axis=1)


def greedy_maximin_points(
    existing: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    n_new: int,
    seed: int = 42,
    n_candidates: int = 4000,
) -> np.ndarray:
    """Greedily add points that maximize the minimum distance to the current set."""
    if n_new <= 0:
        return np.empty((0, existing.shape[1]), dtype=float)
    rng = np.random.default_rng(seed)
    current = _to_unit(np.asarray(existing, dtype=float), lows, highs)
    added: list[np.ndarray] = []
    for i in range(n_new):
        candidates = rng.random((n_candidates, current.shape[1]))
        scores = _pairwise_min_dist(candidates, current)
        best = candidates[int(np.argmax(scores))]
        current = np.vstack([current, best[None, :]])
        added.append(best)
    return _from_unit(np.asarray(added, dtype=float), lows, highs)


def _boundary_distance_unit(candidates: np.ndarray) -> np.ndarray:
    """Min distance to any face of the unit hypercube (0 = on boundary)."""
    return np.min(np.minimum(candidates, 1.0 - candidates), axis=1)


def greedy_maximin_boundary_weighted_points(
    existing: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    n_new: int,
    seed: int = 42,
    n_candidates: int = 4000,
    boundary_weight_alpha: float = 0.1,
    boundary_eps: float = 1e-6,
) -> np.ndarray:
    """Greedy maximin with score = min_dist * (boundary_dist ** alpha).

    ``boundary_weight_alpha`` softens corner/edge preference:
    - 0 → same as plain maximin
    - larger → stronger preference for interior points
    """
    if n_new <= 0:
        return np.empty((0, existing.shape[1]), dtype=float)
    alpha = float(max(boundary_weight_alpha, 0.0))
    eps = float(max(boundary_eps, 0.0))
    rng = np.random.default_rng(seed)
    current = _to_unit(np.asarray(existing, dtype=float), lows, highs)
    added: list[np.ndarray] = []
    for _ in range(n_new):
        candidates = rng.random((n_candidates, current.shape[1]))
        min_dist = _pairwise_min_dist(candidates, current)
        boundary = _boundary_distance_unit(candidates)
        scores = min_dist * np.power(boundary + eps, alpha)
        best = candidates[int(np.argmax(scores))]
        current = np.vstack([current, best[None, :]])
        added.append(best)
    return _from_unit(np.asarray(added, dtype=float), lows, highs)


def augmented_lhs_points(
    existing: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    n_new: int,
    seed: int = 42,
) -> np.ndarray:
    """Augment an existing design toward an LHS of size n_existing + n_new.

    Existing points are kept. New points are placed in currently empty
    one-dimensional strata of the target LHS (size = n_existing + n_new),
    with random within-stratum positions, then lightly refined by maximin.
    """
    if n_new <= 0:
        return np.empty((0, existing.shape[1]), dtype=float)

    rng = np.random.default_rng(seed)
    existing_u = np.clip(_to_unit(np.asarray(existing, dtype=float), lows, highs), 0.0, 1.0 - 1e-12)
    n0, n_dim = existing_u.shape
    n_total = n0 + n_new

    # Empty strata per dimension for an n_total-level LHS
    empty: list[list[int]] = []
    for j in range(n_dim):
        occupied = set(np.floor(existing_u[:, j] * n_total).astype(int).clip(0, n_total - 1).tolist())
        empty.append([k for k in range(n_total) if k not in occupied])

    # Build n_new points by assigning unused strata (with wrap if needed)
    new_u = np.zeros((n_new, n_dim), dtype=float)
    for j in range(n_dim):
        pool = list(empty[j])
        rng.shuffle(pool)
        if len(pool) < n_new:
            # Not enough empty bins (duplicate bins from clustering); recycle randomly
            extra = rng.integers(0, n_total, size=n_new - len(pool)).tolist()
            pool = pool + extra
        chosen = pool[:n_new]
        rng.shuffle(chosen)
        for i, bin_id in enumerate(chosen):
            new_u[i, j] = (bin_id + rng.random()) / n_total

    # Greedy maximin polish among LHS-style candidates for better spacing
    current = existing_u.copy()
    polished: list[np.ndarray] = []
    for i in range(n_new):
        # Local candidates: keep stratum membership roughly, jitter within bins
        base = new_u[i]
        jitter = np.clip(base + rng.normal(0.0, 0.5 / n_total, size=(800, n_dim)), 0.0, 1.0 - 1e-12)
        candidates = np.vstack([base[None, :], jitter])
        scores = _pairwise_min_dist(candidates, current)
        best = candidates[int(np.argmax(scores))]
        current = np.vstack([current, best[None, :]])
        polished.append(best)

    return _from_unit(np.asarray(polished, dtype=float), lows, highs)


def generate_next_doe_locations(
    existing: pd.DataFrame,
    bounds: pd.DataFrame,
    method: str,
    n_new: int,
    seed: int = 42,
    *,
    n_candidates: int = 4000,
    boundary_weight_alpha: float = 0.1,
    boundary_eps: float = 1e-6,
) -> pd.DataFrame:
    """Generate additional DOE points in an (optionally enlarged) exploration box."""
    if n_new <= 0:
        raise ValueError("Number of additional points must be at least 1.")

    bound_frame = bounds.dropna(subset=["name"]).copy()
    bound_frame.loc[:, "name"] = bound_frame["name"].astype(str).str.strip()
    bound_frame = bound_frame.loc[bound_frame["name"] != ""].copy()
    if bound_frame.empty:
        raise ValueError("Provide min/max bounds for at least one feature.")
    if (bound_frame["max"] <= bound_frame["min"]).any():
        raise ValueError("Each feature max must be greater than min.")

    features = bound_frame["name"].tolist()
    missing = [f for f in features if f not in existing.columns]
    if missing:
        raise ValueError(f"Existing design is missing columns: {', '.join(missing)}")

    x_exist = existing[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(x_exist).any():
        raise ValueError("Existing design contains non-numeric values in selected features.")

    lows = bound_frame["min"].to_numpy(dtype=float)
    highs = bound_frame["max"].to_numpy(dtype=float)

    if method == "Greedy Maximin Distance":
        raw = greedy_maximin_points(x_exist, lows, highs, n_new, seed=seed, n_candidates=n_candidates)
    elif method == "Greedy Maximin Distance (boundary-weighted)":
        raw = greedy_maximin_boundary_weighted_points(
            x_exist,
            lows,
            highs,
            n_new,
            seed=seed,
            n_candidates=n_candidates,
            boundary_weight_alpha=boundary_weight_alpha,
            boundary_eps=boundary_eps,
        )
    elif method == "Augmented Latin Hypercube Sampling (LHS)":
        raw = augmented_lhs_points(x_exist, lows, highs, n_new, seed=seed)
    else:
        raise ValueError(f"Unknown next-location method: {method}")

    start_id = 1
    if "sample_id" in existing.columns:
        try:
            start_id = int(pd.to_numeric(existing["sample_id"], errors="coerce").max()) + 1
        except Exception:
            start_id = len(existing) + 1
    else:
        start_id = len(existing) + 1

    frame = pd.DataFrame(raw, columns=features)
    frame.insert(0, "sample_id", np.arange(start_id, start_id + len(frame)))
    return frame
