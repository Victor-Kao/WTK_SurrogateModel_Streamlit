"""Model registries, training, and scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    auc as sk_auc,
)
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC, SVR

REGRESSORS: dict[str, Any] = {
    "Linear Regression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "SVR": SVR,
    "KNN": KNeighborsRegressor,
    "MLP": MLPRegressor,
    "GPR": GaussianProcessRegressor,
}

CLASSIFIERS: dict[str, Any] = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "SVC": SVC,
    "KNN": KNeighborsClassifier,
    "MLP": MLPClassifier,
    "GPC": GaussianProcessClassifier,
}

# Full display names for UI labels (abbreviations / class names → expanded names).
ESTIMATOR_FULL_NAMES: dict[str, str] = {
    "Linear Regression": "Linear Regression",
    "LinearRegression": "Linear Regression",
    "Ridge": "Ridge Regression",
    "Lasso": "Lasso Regression",
    "Random Forest": "Random Forest",
    "RandomForestRegressor": "Random Forest Regressor",
    "RandomForestClassifier": "Random Forest Classifier",
    "Gradient Boosting": "Gradient Boosting",
    "GradientBoostingRegressor": "Gradient Boosting Regressor",
    "GradientBoostingClassifier": "Gradient Boosting Classifier",
    "SVR": "Support Vector Regression (SVR)",
    "SVC": "Support Vector Classification (SVC)",
    "KNN": "K-Nearest Neighbors (KNN)",
    "KNeighborsRegressor": "K-Nearest Neighbors Regressor",
    "KNeighborsClassifier": "K-Nearest Neighbors Classifier",
    "MLP": "Multi-Layer Perceptron (MLP)",
    "MLPRegressor": "Multi-Layer Perceptron Regressor",
    "MLPClassifier": "Multi-Layer Perceptron Classifier",
    "GPR": "Gaussian Process Regression (GPR)",
    "GPC": "Gaussian Process Classification (GPC)",
    "GaussianProcessRegressor": "Gaussian Process Regression (GPR)",
    "GaussianProcessClassifier": "Gaussian Process Classification (GPC)",
    "Logistic Regression": "Logistic Regression",
    "LogisticRegression": "Logistic Regression",
}


def estimator_display_name(name: str | None) -> str:
    """Return a full estimator label for UI display."""
    if not name:
        return "—"
    return ESTIMATOR_FULL_NAMES.get(str(name), str(name))


# Editable hyperparameters shown in Algorithm Setup (type, default).
# Types: int | float | bool | str | optional_int | tuple_int | choice
_REG_HYPERPARAMS: dict[str, dict[str, tuple[str, Any]]] = {
    "Linear Regression": {
        "fit_intercept": ("bool", True),
    },
    "Ridge": {
        "alpha": ("float", 1.0),
        "fit_intercept": ("bool", True),
        "solver": ("choice", "auto"),
    },
    "Lasso": {
        "alpha": ("float", 0.01),
        "max_iter": ("int", 5000),
        "fit_intercept": ("bool", True),
    },
    "Random Forest": {
        "n_estimators": ("int", 200),
        "max_depth": ("optional_int", None),
        "min_samples_split": ("int", 2),
        "min_samples_leaf": ("int", 1),
        "max_features": ("choice", "sqrt"),
        "random_state": ("int", 42),
    },
    "Gradient Boosting": {
        "n_estimators": ("int", 100),
        "learning_rate": ("float", 0.1),
        "max_depth": ("int", 3),
        "min_samples_split": ("int", 2),
        "min_samples_leaf": ("int", 1),
        "subsample": ("float", 1.0),
        "random_state": ("int", 42),
    },
    "SVR": {
        "C": ("float", 1.0),
        "kernel": ("choice", "rbf"),
        "gamma": ("str", "scale"),
        "epsilon": ("float", 0.1),
    },
    "KNN": {
        "n_neighbors": ("int", 5),
        "weights": ("choice", "uniform"),
        "p": ("int", 2),
    },
    "MLP": {
        "hidden_layer_sizes": ("tuple_int", (64, 32)),
        "activation": ("choice", "relu"),
        "alpha": ("float", 0.0001),
        "learning_rate_init": ("float", 0.001),
        "max_iter": ("int", 500),
        "random_state": ("int", 42),
    },
    "GPR": {
        "kernel": ("choice", "RBF"),
        "alpha": ("float", 1e-10),
        "n_restarts_optimizer": ("int", 2),
        "normalize_y": ("bool", True),
        "random_state": ("int", 42),
    },
}

_CLF_HYPERPARAMS: dict[str, dict[str, tuple[str, Any]]] = {
    "Logistic Regression": {
        "C": ("float", 1.0),
        "max_iter": ("int", 500),
        "penalty": ("choice", "l2"),
        "solver": ("choice", "lbfgs"),
    },
    "Random Forest": {
        "n_estimators": ("int", 200),
        "max_depth": ("optional_int", None),
        "min_samples_split": ("int", 2),
        "min_samples_leaf": ("int", 1),
        "max_features": ("choice", "sqrt"),
        "random_state": ("int", 42),
    },
    "Gradient Boosting": {
        "n_estimators": ("int", 100),
        "learning_rate": ("float", 0.1),
        "max_depth": ("int", 3),
        "min_samples_split": ("int", 2),
        "min_samples_leaf": ("int", 1),
        "subsample": ("float", 1.0),
        "random_state": ("int", 42),
    },
    "SVC": {
        "C": ("float", 1.0),
        "kernel": ("choice", "rbf"),
        "gamma": ("str", "scale"),
        "probability": ("bool", True),
    },
    "KNN": {
        "n_neighbors": ("int", 5),
        "weights": ("choice", "uniform"),
        "p": ("int", 2),
    },
    "MLP": {
        "hidden_layer_sizes": ("tuple_int", (64, 32)),
        "activation": ("choice", "relu"),
        "alpha": ("float", 0.0001),
        "learning_rate_init": ("float", 0.001),
        "max_iter": ("int", 500),
        "random_state": ("int", 42),
    },
    "GPC": {
        "kernel": ("choice", "RBF"),
        "n_restarts_optimizer": ("int", 2),
        "max_iter_predict": ("int", 100),
        "multi_class": ("choice", "one_vs_rest"),
        "random_state": ("int", 42),
    },
}

CHOICES: dict[str, list[Any]] = {
    "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs", "newton-cg", "liblinear"],
    "kernel": ["rbf", "linear", "poly", "sigmoid"],
    "gp_kernel": ["RBF", "Matern", "RBF+White", "DotProduct", "DotProduct+White"],
    "weights": ["uniform", "distance"],
    "activation": ["relu", "tanh", "logistic", "identity"],
    "penalty": ["l2", "l1", "elasticnet", "None"],
    "max_features": ["sqrt", "log2", "None", "1.0"],
    "multi_class": ["one_vs_rest", "one_vs_one"],
}


def hyperparam_schema(task: str, name: str) -> dict[str, tuple[str, Any]]:
    table = _REG_HYPERPARAMS if task == "regression" else _CLF_HYPERPARAMS
    return dict(table.get(name, {}))


def default_hyperparams(task: str, name: str) -> dict[str, Any]:
    return {key: default for key, (_kind, default) in hyperparam_schema(task, name).items()}

SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
    "Random Forest": {
        "model__n_estimators": [80, 150, 250],
        "model__max_depth": [4, 8, 12, None],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "Gradient Boosting": {
        "model__n_estimators": [80, 150],
        "model__learning_rate": [0.03, 0.1, 0.2],
        "model__max_depth": [2, 3, 4],
    },
    "SVR": {"model__C": [0.5, 1.0, 5.0], "model__gamma": ["scale", 0.1]},
    "SVC": {"model__C": [0.5, 1.0, 5.0], "model__gamma": ["scale", 0.1]},
    "KNN": {"model__n_neighbors": [3, 5, 9, 15]},
    "Logistic Regression": {"model__C": [0.1, 1.0, 5.0]},
    "GPR": {
        "model__alpha": [1e-10, 1e-5, 1e-2],
        "model__n_restarts_optimizer": [0, 2, 5],
    },
    "GPC": {
        "model__n_restarts_optimizer": [0, 2, 5],
        "model__max_iter_predict": [50, 100, 200],
    },
}


def registry(task: str) -> dict[str, Any]:
    return REGRESSORS if task == "regression" else CLASSIFIERS


def _build_gp_kernel(name: str, n_features: int | None = None) -> Any:
    """Map a kernel label to a scikit-learn Gaussian-process kernel.

    When ``n_features`` is set, RBF/Matern use anisotropic (ARD) length scales
    so per-feature relevance can be read after fitting.
    """
    key = str(name).strip()
    length_scale: Any = 1.0
    if n_features is not None and int(n_features) > 0:
        length_scale = np.ones(int(n_features), dtype=float)
    mapping = {
        "RBF": 1.0 * RBF(length_scale=length_scale),
        "Matern": 1.0 * Matern(length_scale=length_scale, nu=2.5),
        "RBF+White": 1.0 * RBF(length_scale=length_scale) + WhiteKernel(noise_level=1e-5),
        "DotProduct": DotProduct(sigma_0=1.0),
        "DotProduct+White": DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=1e-5),
        # SVM-style labels fall back to RBF for GP models
        "rbf": 1.0 * RBF(length_scale=length_scale),
        "linear": DotProduct(sigma_0=1.0),
    }
    if key not in mapping:
        raise ValueError(f"Unknown GP kernel: {name}")
    return mapping[key]


def make_estimator(
    task: str,
    name: str,
    params: dict[str, Any] | None = None,
    *,
    n_features: int | None = None,
) -> Any:
    merged = default_hyperparams(task, name)
    if params:
        merged.update(params)
    # max_features: allow numeric string "1.0" from choice widgets
    if "max_features" in merged and isinstance(merged["max_features"], str):
        raw = merged["max_features"]
        if raw in {"None", "none"}:
            merged["max_features"] = None
        else:
            try:
                merged["max_features"] = float(raw)
            except ValueError:
                pass
    if "penalty" in merged and merged["penalty"] in {"None", "none"}:
        merged["penalty"] = None
    if name in {"GPR", "GPC"} and "kernel" in merged and isinstance(merged["kernel"], str):
        merged["kernel"] = _build_gp_kernel(merged["kernel"], n_features=n_features)
    return registry(task)[name](**merged)


def make_pipeline(estimator: Any, scale: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def split_xy(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = frame[features]
    y = frame[target]
    strat = y if stratify else None
    positions = np.arange(len(frame))
    train_pos, test_pos = train_test_split(
        positions, test_size=test_size, random_state=random_state, stratify=strat
    )
    return x.iloc[train_pos], x.iloc[test_pos], y.iloc[train_pos], y.iloc[test_pos]


def make_cv_splitter(scheme: str, n_folds: int, seed: int, stratify: bool) -> Any:
    if scheme == "LOOCV":
        return LeaveOneOut()
    if stratify:
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return KFold(n_splits=n_folds, shuffle=True, random_state=seed)


def row_assignments(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    scheme: str,
    val_fraction: float,
    n_folds: int,
    seed: int,
    stratify: bool,
) -> pd.Series:
    """Label each row with fold ID or Training/Testing, matching the training split."""
    n = len(frame)
    x = frame[features]
    y = frame[target]
    if scheme == "Split Dataset":
        positions = np.arange(n)
        _, test_pos = train_test_split(
            positions,
            test_size=val_fraction,
            random_state=seed,
            stratify=y if stratify else None,
        )
        labels = np.full(n, "Training", dtype=object)
        labels[np.asarray(test_pos)] = "Testing"
        return pd.Series(labels, name="Data type")

    splitter = make_cv_splitter(scheme, n_folds, seed, stratify=stratify)
    fold_id = np.zeros(n, dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(x, y), start=1):
        fold_id[np.asarray(test_idx)] = fold
    return pd.Series(fold_id, name="CV fold ID")


def assignment_export_frame(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    assignment: pd.Series,
) -> pd.DataFrame:
    """Combine id, X, Y, and fold/split labels into one table for CSV download."""
    export = pd.DataFrame(index=frame.index)
    if "id" in frame.columns:
        export["id"] = frame["id"].to_numpy()
    elif "sample_id" in frame.columns:
        export["id"] = frame["sample_id"].to_numpy()
    else:
        export["id"] = np.arange(1, len(frame) + 1)
    for col in features:
        export[col] = frame[col].to_numpy()
    export[target] = frame[target].to_numpy()
    export[assignment.name] = assignment.to_numpy()
    return export.reset_index(drop=True)


def split_type_label(scheme: str) -> str:
    if scheme == "Split Dataset":
        return "DATA_SPLIT"
    if scheme in {"CV folds", "LOOCV"}:
        return "CV_FOLD"
    raise ValueError(f"Unknown scheme: {scheme}")


def adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """Adjusted coefficient of determination."""
    denom = n_samples - n_features - 1
    if denom <= 0 or n_samples <= 1:
        return float("nan")
    return float(1.0 - (1.0 - r2) * (n_samples - 1) / denom)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int | None = None,
) -> dict[str, float]:
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    p = int(n_features) if n_features is not None else 0
    return {
        "R²": r2,
        "R² adj": adjusted_r2(r2, len(y_true), p),
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    # Binary: class-specific F-beta; multiclass: macro F-beta (equal class weight).
    f_avg = "binary" if n_classes <= 2 else "macro"
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "F0.5 (precision-focused)": float(
            fbeta_score(y_true, y_pred, beta=0.5, average=f_avg, zero_division=0)
        ),
        "F2 (recall-focused)": float(
            fbeta_score(y_true, y_pred, beta=2.0, average=f_avg, zero_division=0)
        ),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Micro-F1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def predict_class_scores(model: Any, x_data: pd.DataFrame) -> tuple[np.ndarray | None, list[Any] | None]:
    """Return class scores for ROC: (n_samples, n_classes) and class label list."""
    pipe = get_pipeline(model)
    classes = getattr(pipe, "classes_", None)
    if classes is None and isinstance(pipe, Pipeline):
        est = pipe.named_steps.get("model", pipe.steps[-1][1])
        classes = getattr(est, "classes_", None)

    if hasattr(pipe, "predict_proba"):
        try:
            scores = np.asarray(pipe.predict_proba(x_data), dtype=float)
            if scores.ndim == 1:
                scores = np.column_stack([1.0 - scores, scores])
            labels = list(classes) if classes is not None else list(range(scores.shape[1]))
            return scores, labels
        except Exception:
            pass

    if hasattr(pipe, "decision_function"):
        try:
            raw = np.asarray(pipe.decision_function(x_data), dtype=float)
            if raw.ndim == 1:
                # Binary decision scores → pseudo two-column layout
                scores = np.column_stack([-raw, raw])
                labels = list(classes) if classes is not None else [0, 1]
                return scores, labels
            labels = list(classes) if classes is not None else list(range(raw.shape[1]))
            return raw, labels
        except Exception:
            pass
    return None, None


def _binary_class_order(class_labels: list[Any]) -> tuple[Any, Any, int]:
    """Return (negative_label, positive_label, positive_column_index).

    Prefer label ``1`` as positive when both 0 and 1 are present; otherwise use
    sklearn's ``classes_[1]`` as the positive class.
    """
    labels = list(class_labels)
    if len(labels) != 2:
        raise ValueError("Binary scoring requires exactly two classes.")
    as_set = set(labels)
    if as_set == {0, 1} or as_set == {0.0, 1.0}:
        positive = 1 if 1 in labels else 1.0
        negative = 0 if 0 in labels else 0.0
        pos_idx = labels.index(positive)
        return negative, positive, pos_idx
    # Also handle string "0"/"1"
    if as_set == {"0", "1"}:
        pos_idx = labels.index("1")
        return "0", "1", pos_idx
    return labels[0], labels[1], 1


def binary_positive_probability(
    model: Any, x_data: pd.DataFrame
) -> tuple[np.ndarray, Any, Any, str] | None:
    """Positive-class score in ~[0, 1] for thresholding.

    Returns ``(scores, negative_label, positive_label, kind)`` where ``kind`` is
    ``\"probability\"`` (from ``predict_proba``) or ``\"sigmoid_decision\"``
    (logistic of ``decision_function``). Same fitted pickle — no second model.
    """
    pipe = get_pipeline(model)
    classes = getattr(pipe, "classes_", None)
    if classes is None and isinstance(pipe, Pipeline):
        est = pipe.named_steps.get("model", pipe.steps[-1][1])
        classes = getattr(est, "classes_", None)

    if hasattr(pipe, "predict_proba"):
        try:
            proba = np.asarray(pipe.predict_proba(x_data), dtype=float)
            if proba.ndim == 1:
                proba = np.column_stack([1.0 - proba, proba])
            if proba.shape[1] != 2:
                return None
            labels = list(classes) if classes is not None else [0, 1]
            negative, positive, pos_idx = _binary_class_order(labels)
            return proba[:, pos_idx], negative, positive, "probability"
        except Exception:
            pass

    if hasattr(pipe, "decision_function"):
        try:
            raw = np.asarray(pipe.decision_function(x_data), dtype=float).reshape(-1)
            labels = list(classes) if classes is not None else [0, 1]
            if len(labels) != 2:
                return None
            negative, positive, _ = _binary_class_order(labels)
            # Map decision margin → (0, 1) so the same threshold slider applies
            scores = 1.0 / (1.0 + np.exp(-raw))
            return scores, negative, positive, "sigmoid_decision"
        except Exception:
            pass
    return None


def labels_from_binary_scores(
    scores: np.ndarray,
    threshold: float,
    negative_label: Any,
    positive_label: Any,
) -> np.ndarray:
    """``predicted = positive if score >= threshold else negative``."""
    scores = np.asarray(scores, dtype=float)
    return np.where(scores >= float(threshold), positive_label, negative_label)


def roc_curve_table(
    y_true: np.ndarray,
    scores: np.ndarray,
    class_labels: list[Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build ROC curve rows (for CSV/plot) and AUC summary metrics.

    Binary: one positive-class curve.
    Multiclass: one-vs-rest curves per class plus a micro-average curve.
    """
    y_true = np.asarray(y_true)
    labels = list(class_labels)
    rows: list[dict[str, Any]] = []
    auc_summary: dict[str, float] = {}

    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("ROC requires at least two class score columns.")

    if len(labels) == 2:
        pos = labels[1]
        y_bin = (y_true == pos).astype(int)
        fpr, tpr, thr = roc_curve(y_bin, scores[:, 1])
        roc_auc = float(sk_auc(fpr, tpr))
        auc_summary["AUC"] = roc_auc
        auc_summary[f"AUC ({pos})"] = roc_auc
        for i in range(len(fpr)):
            rows.append(
                {
                    "curve": f"positive={pos}",
                    "fpr": float(fpr[i]),
                    "tpr": float(tpr[i]),
                    "threshold": float(thr[i]) if i < len(thr) else np.nan,
                    "auc": roc_auc,
                }
            )
    else:
        y_bin = label_binarize(y_true, classes=labels)
        if y_bin.shape[1] == 1:
            # label_binarize edge case with 2 classes represented oddly
            y_bin = np.hstack([1 - y_bin, y_bin])
        per_class_auc: list[float] = []
        for idx, label in enumerate(labels):
            if idx >= scores.shape[1]:
                break
            fpr, tpr, thr = roc_curve(y_bin[:, idx], scores[:, idx])
            roc_auc = float(sk_auc(fpr, tpr))
            per_class_auc.append(roc_auc)
            auc_summary[f"AUC ({label})"] = roc_auc
            for i in range(len(fpr)):
                rows.append(
                    {
                        "curve": f"ovr={label}",
                        "fpr": float(fpr[i]),
                        "tpr": float(tpr[i]),
                        "threshold": float(thr[i]) if i < len(thr) else np.nan,
                        "auc": roc_auc,
                    }
                )
        # Micro-average
        fpr_m, tpr_m, thr_m = roc_curve(y_bin.ravel(), scores[:, : y_bin.shape[1]].ravel())
        micro_auc = float(sk_auc(fpr_m, tpr_m))
        auc_summary["AUC micro"] = micro_auc
        if per_class_auc:
            auc_summary["AUC macro"] = float(np.mean(per_class_auc))
        for i in range(len(fpr_m)):
            rows.append(
                {
                    "curve": "micro-average",
                    "fpr": float(fpr_m[i]),
                    "tpr": float(tpr_m[i]),
                    "threshold": float(thr_m[i]) if i < len(thr_m) else np.nan,
                    "auc": micro_auc,
                }
            )
        try:
            auc_summary["AUC weighted OvR"] = float(
                roc_auc_score(y_true, scores, multi_class="ovr", average="weighted", labels=labels)
            )
        except Exception:
            pass

    return pd.DataFrame(rows), auc_summary


METRIC_GUIDE: dict[str, dict[str, str]] = {
    "R²": {
        "full_name": "Coefficient of determination (R-squared)",
        "good": "Closer to 1 is better. Often ≥ 0.8 is strong; ~0.5–0.8 moderate; near 0 weak. Negative means worse than predicting the mean.",
        "description": "Fraction of target variance explained by the model. 1 = perfect fit on this set; 0 = no better than the mean.",
    },
    "R² adj": {
        "full_name": "Adjusted coefficient of determination",
        "good": "Closer to 1 is better. Prefer models where R² adj stays close to R²; a large drop suggests extra features are not helping.",
        "description": "R² penalized for the number of inputs, so adding useless features tends to lower the score.",
    },
    "RMSE": {
        "full_name": "Root mean squared error",
        "good": "Lower is better. Interpret in the same units as Y; good values depend on your target scale (compare to Y’s range/std).",
        "description": "Square root of average squared prediction error. Emphasizes larger mistakes more than MAE.",
    },
    "MAE": {
        "full_name": "Mean absolute error",
        "good": "Lower is better. Same units as Y; easier to read as a typical absolute miss.",
        "description": "Average absolute difference between observed and predicted values.",
    },
    "MAPE (%)": {
        "full_name": "Mean absolute percentage error",
        "good": "Lower is better. Often < 10% is strong, 10–20% acceptable, > 20% weak — but unstable if Y is near zero.",
        "description": "Average absolute error as a percent of the true value. Useful for relative accuracy across scales.",
    },
    "Accuracy": {
        "full_name": "Classification accuracy",
        "good": "Closer to 1 is better. Often ≥ 0.9 is strong on balanced problems; with imbalance, compare against Balanced accuracy / Macro-F1.",
        "description": "Overall share of labels predicted correctly. Can look high when a majority class dominates.",
    },
    "Balanced accuracy": {
        "full_name": "Balanced accuracy (macro-averaged recall)",
        "good": "Closer to 1 is better. Prefer this over Accuracy when class sizes differ; ≥ 0.8–0.9 is often a strong target depending on difficulty.",
        "description": "Unweighted mean of per-class recall. Each class contributes equally, so poor minority-class performance lowers the score.",
    },
    "F0.5 (precision-focused)": {
        "full_name": "F₀.₅ score (precision-focused F-beta)",
        "good": "Closer to 1 is better. Prioritize when false alarms are costly (e.g. spam filtering, automated blocking). Strong models typically keep F₀.₅ high while not collapsing recall entirely.",
        "description": "F-beta with β = 0.5 weights precision more than recall. Binary: positive-class F₀.₅. Multiclass: macro-averaged F₀.₅.",
    },
    "F2 (recall-focused)": {
        "full_name": "F₂ score (recall-focused F-beta)",
        "good": "Closer to 1 is better. Prioritize when missed detections are costly (e.g. disease screening, machine fault detection). Strong models raise F₂ without extreme false-positive rates.",
        "description": "F-beta with β = 2 weights recall more than precision. Binary: positive-class F₂. Multiclass: macro-averaged F₂.",
    },
    "Macro-F1": {
        "full_name": "Macro-averaged F1",
        "good": "Closer to 1 is better. Use when every class is equally important regardless of support. Failure on a rare class lowers Macro-F1 sharply — that is intended.",
        "description": "Computes F1 per class, then takes the unweighted mean: Macro-F1 = (1/K) Σ F1_k. Preferred for multiclass settings with equal class importance.",
    },
    "Micro-F1": {
        "full_name": "Micro-averaged F1",
        "good": "Closer to 1 is better. Reflects global TP/FP/FN counts. In single-label multiclass problems, Micro-F1 equals Accuracy.",
        "description": "Aggregates true positives, false positives, and false negatives globally before computing F1. Emphasizes overall correct decisions more than rare-class failures.",
    },
}


def metric_help_text(name: str) -> str:
    info = METRIC_GUIDE.get(name)
    if not info:
        return name
    return (
        f"{info['full_name']}\n\n"
        f"Good model: {info['good']}\n\n"
        f"{info['description']}"
    )


def score_predictions(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int | None = None,
) -> dict[str, float]:
    if task == "regression":
        return regression_metrics(y_true, y_pred, n_features=n_features)
    return classification_metrics(y_true, y_pred)


def train_model(
    task: str,
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    scale: bool,
    params: dict[str, Any] | None = None,
) -> tuple[Pipeline, dict[str, dict[str, float]], np.ndarray, np.ndarray]:
    n_features = int(x_train.shape[1])
    pipe = make_pipeline(make_estimator(task, name, params, n_features=n_features), scale)
    pipe.fit(x_train, y_train)
    y_pred_train = pipe.predict(x_train)
    y_pred_test = pipe.predict(x_test)
    metrics = {
        "train": score_predictions(task, y_train.to_numpy(), y_pred_train, n_features=n_features),
        "test": score_predictions(task, y_test.to_numpy(), y_pred_test, n_features=n_features),
    }
    return pipe, metrics, y_pred_train, y_pred_test


def fit_and_evaluate(
    task: str,
    name: str,
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    scale: bool,
    scheme: str,
    val_fraction: float,
    n_folds: int,
    seed: int,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one model under hold-out split, k-fold CV, or LOOCV."""
    stratify = task == "classification" and scheme != "LOOCV"
    x = frame[features]
    y = frame[target]

    if scheme == "Split Dataset":
        x_train, x_test, y_train, y_test = split_xy(
            frame, features, target, val_fraction, seed, stratify=stratify
        )
        pipe, metrics, y_pred_train, y_pred_test = train_model(
            task, name, x_train, y_train, x_test, y_test, scale, params=params
        )
        return {
            "pipeline": pipe,
            "metrics": metrics,
            "X_train": x_train,
            "X_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "metric_label": "Hold-out validation",
        }

    splitter = make_cv_splitter(scheme, n_folds, seed, stratify=stratify)
    pipe = make_pipeline(make_estimator(task, name, params, n_features=int(x.shape[1])), scale)
    y_oof = cross_val_predict(pipe, x, y, cv=splitter)
    pipe.fit(x, y)
    y_hat = pipe.predict(x)
    metrics = {
        "train": score_predictions(task, y.to_numpy(), y_hat, n_features=x.shape[1]),
        "test": score_predictions(task, y.to_numpy(), y_oof, n_features=x.shape[1]),
    }
    label = "LOOCV" if scheme == "LOOCV" else f"{n_folds}-fold CV"
    return {
        "pipeline": pipe,
        "metrics": metrics,
        "X_train": x,
        "X_test": x,
        "y_train": y,
        "y_test": y,
        "y_pred_train": y_hat,
        "y_pred_test": y_oof,
        "metric_label": label,
    }


def compare_models(
    task: str,
    names: list[str],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    scale: bool,
    n_iter: int,
    seed: int,
    cv: Any,
) -> tuple[pd.DataFrame, str, Pipeline]:
    scoring = "r2" if task == "regression" else "f1_weighted"
    splitter = cv

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    n_features = int(x_train.shape[1])
    for name in names:
        pipe = make_pipeline(make_estimator(task, name, n_features=n_features), scale)
        space = SEARCH_SPACES.get(name)
        if space:
            search = RandomizedSearchCV(
                pipe,
                space,
                n_iter=min(n_iter, int(np.prod([len(v) for v in space.values()]))),
                cv=splitter,
                scoring=scoring,
                random_state=seed,
                n_jobs=1,
            )
            search.fit(x_train, y_train)
            best_pipe = search.best_estimator_
            cv_mean = float(search.best_score_)
            params = search.best_params_
        else:
            scores = cross_val_score(pipe, x_train, y_train, cv=splitter, scoring=scoring, n_jobs=1)
            pipe.fit(x_train, y_train)
            best_pipe = pipe
            cv_mean = float(scores.mean())
            params = {}
        fitted[name] = best_pipe
        rows.append(
            {
                "Model": name,
                "CV score": cv_mean,
                "Best params": ", ".join(f"{k.replace('model__', '')}={v}" for k, v in params.items()) or "defaults",
            }
        )

    table = pd.DataFrame(rows).sort_values("CV score", ascending=False).reset_index(drop=True)
    winner = str(table.iloc[0]["Model"])
    return table, winner, fitted[winner]


def _kernel_length_scales(kernel: Any) -> np.ndarray | None:
    """Extract length_scale vector from a (possibly composite) GP kernel."""
    if kernel is None:
        return None
    if hasattr(kernel, "k1") or hasattr(kernel, "k2"):
        found: list[np.ndarray] = []
        for attr in ("k1", "k2"):
            if hasattr(kernel, attr):
                part = _kernel_length_scales(getattr(kernel, attr))
                if part is not None:
                    found.append(part)
        anisotropic = [arr for arr in found if arr.size > 1]
        if anisotropic:
            return anisotropic[0]
        if found:
            return found[0]
    if hasattr(kernel, "length_scale"):
        return np.atleast_1d(np.asarray(kernel.length_scale, dtype=float).ravel())
    return None


def _normalize_importance(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.maximum(vals, 0.0)
    total = float(vals.sum())
    if total > 0:
        vals = vals / total
    return vals


def _gpr_ard_importance(estimator: Any, n_features: int) -> np.ndarray | None:
    """Feature relevance from ARD length scales: importance ∝ 1 / length_scale."""
    kernel = getattr(estimator, "kernel_", None)
    if kernel is None:
        kernel = getattr(estimator, "kernel", None)
    length_scales = _kernel_length_scales(kernel)
    if length_scales is None or length_scales.size != n_features:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        relevance = 1.0 / np.maximum(length_scales, 1e-12)
    return _normalize_importance(relevance)


def _permutation_importance_values(
    model: Any,
    x_data: pd.DataFrame,
    y_data: pd.Series | np.ndarray,
    *,
    seed: int = 42,
) -> np.ndarray | None:
    try:
        result = permutation_importance(
            model,
            x_data,
            y_data,
            n_repeats=8,
            random_state=seed,
            scoring="r2",
            n_jobs=1,
        )
        return _normalize_importance(np.asarray(result.importances_mean, dtype=float))
    except Exception:
        return None


def _native_importance_values(estimator: Any, n_features: int) -> np.ndarray | None:
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coef).ravel() if coef.ndim == 1 else np.mean(np.abs(coef), axis=0)
    else:
        return None
    if values.size != n_features:
        return None
    return _normalize_importance(values)


def compute_feature_importance_map(
    model: Any,
    feature_names: list[str],
    x_data: pd.DataFrame | None = None,
    y_data: pd.Series | np.ndarray | None = None,
) -> tuple[dict[str, float], str | None]:
    """Compute feature → importance map and a short method label."""
    pipeline = get_pipeline(model)
    estimator = pipeline.named_steps.get("model", pipeline) if isinstance(pipeline, Pipeline) else pipeline
    n_features = len(feature_names)

    values = _native_importance_values(estimator, n_features)
    method: str | None = None
    if values is not None:
        if hasattr(estimator, "feature_importances_"):
            method = "model feature_importances_"
        else:
            method = "|coef| (normalized)"
    elif isinstance(estimator, GaussianProcessRegressor):
        values = _gpr_ard_importance(estimator, n_features)
        if values is not None:
            method = "GPR ARD (1 / length_scale)"
        elif x_data is not None and y_data is not None:
            values = _permutation_importance_values(pipeline, x_data[feature_names], y_data)
            if values is not None:
                method = "permutation importance (R²)"

    if values is None or len(values) != n_features:
        return {}, None
    return {str(name): float(val) for name, val in zip(feature_names, values)}, method


def feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    model = ensure_surrogate(model) if is_surrogate(model) else model
    if is_surrogate(model) and getattr(model, "feature_importance", None):
        names = list(feature_names) if feature_names else list(model.feature_names)
        if not names:
            names = list(model.feature_importance.keys())
        values = [float(model.feature_importance.get(name, 0.0)) for name in names]
        if names and (any(abs(v) > 0 for v in values) or set(model.feature_importance.keys()) & set(names)):
            frame = pd.DataFrame({"feature": names, "importance": values})
            return frame.sort_values("importance", ascending=False).reset_index(drop=True)

    importance_map, _method = compute_feature_importance_map(model, feature_names)
    if not importance_map:
        return None
    frame = pd.DataFrame(
        {"feature": list(importance_map.keys()), "importance": list(importance_map.values())}
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


@dataclass
class SurrogateModel:
    """Fitted surrogate plus metadata (including supported X ranges)."""

    pipeline: Any
    feature_names: list[str]
    feature_bounds: dict[str, dict[str, float]]
    model_name: str | None = None
    task: str | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    importance_method: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def decision_function(self, X):
        return self.pipeline.decision_function(X)

    def get_params(self, deep: bool = True):
        if hasattr(self.pipeline, "get_params"):
            return self.pipeline.get_params(deep=deep)
        return {}

    def __reduce__(self):
        # Stable across Streamlit module reloads (avoids "not the same object as SurrogateModel")
        return (_rebuild_surrogate_from_payload, (surrogate_to_payload(self),))


def is_surrogate(model: Any) -> bool:
    """True for SurrogateModel, reload-stale class instances, or serialized payloads."""
    if isinstance(model, SurrogateModel):
        return True
    if isinstance(model, dict) and (
        model.get("__wtk_surrogate__")
        or ("pipeline" in model and "feature_bounds" in model)
    ):
        return True
    return (
        type(model).__name__ == "SurrogateModel"
        and hasattr(model, "pipeline")
        and hasattr(model, "feature_bounds")
        and hasattr(model, "feature_names")
    )


def surrogate_to_payload(model: Any) -> dict[str, Any]:
    """Convert a surrogate to a plain dict for pickle/joblib (class-identity safe)."""
    if isinstance(model, dict) and "pipeline" in model:
        payload = dict(model)
        payload["__wtk_surrogate__"] = True
        return payload
    return {
        "__wtk_surrogate__": True,
        "pipeline": getattr(model, "pipeline"),
        "feature_names": [str(n) for n in getattr(model, "feature_names", [])],
        "feature_bounds": {
            str(k): {"min": float(v["min"]), "max": float(v["max"])}
            for k, v in dict(getattr(model, "feature_bounds", {}) or {}).items()
        },
        "model_name": getattr(model, "model_name", None),
        "task": getattr(model, "task", None),
        "feature_importance": {
            str(k): float(v) for k, v in dict(getattr(model, "feature_importance", {}) or {}).items()
        },
        "importance_method": getattr(model, "importance_method", None),
        "extra": dict(getattr(model, "extra", {}) or {}),
    }


def _rebuild_surrogate_from_payload(payload: dict[str, Any]) -> SurrogateModel:
    bounds = payload.get("feature_bounds") or {}
    names = payload.get("feature_names") or list(bounds.keys())
    importance = payload.get("feature_importance") or {}
    known = {
        "__wtk_surrogate__",
        "pipeline",
        "feature_bounds",
        "feature_names",
        "model_name",
        "task",
        "feature_importance",
        "importance_method",
        "extra",
    }
    extra = dict(payload.get("extra") or {})
    extra.update({k: v for k, v in payload.items() if k not in known})
    return SurrogateModel(
        pipeline=payload["pipeline"],
        feature_names=[str(n) for n in names],
        feature_bounds={str(k): dict(v) for k, v in bounds.items()},
        model_name=payload.get("model_name"),
        task=payload.get("task"),
        feature_importance={str(k): float(v) for k, v in importance.items()},
        importance_method=payload.get("importance_method"),
        extra=extra,
    )


def ensure_surrogate(model: Any) -> Any:
    """Rebuild onto the current SurrogateModel class (fixes Streamlit hot-reload stale instances)."""
    if isinstance(model, SurrogateModel):
        return model
    if is_surrogate(model):
        return _rebuild_surrogate_from_payload(surrogate_to_payload(model))
    return model


def feature_bounds_from_frame(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Min/max of each numeric input column (model supported range)."""
    bounds: dict[str, dict[str, float]] = {}
    for col in frame.columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        bounds[str(col)] = {
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return bounds


def wrap_surrogate(
    pipeline: Any,
    x_data: pd.DataFrame,
    *,
    y_data: pd.Series | np.ndarray | None = None,
    model_name: str | None = None,
    task: str | None = None,
) -> SurrogateModel:
    """Attach training feature names, min/max ranges, and feature importance."""
    features = [str(c) for c in x_data.columns]
    importance_map, importance_method = compute_feature_importance_map(
        pipeline, features, x_data=x_data, y_data=y_data
    )
    return SurrogateModel(
        pipeline=pipeline,
        feature_names=features,
        feature_bounds=feature_bounds_from_frame(x_data[features]),
        model_name=model_name,
        task=task,
        feature_importance=importance_map,
        importance_method=importance_method,
    )


def get_pipeline(model: Any) -> Any:
    """Return the sklearn pipeline/estimator from a SurrogateModel or raw object."""
    if is_surrogate(model):
        if isinstance(model, dict):
            return model["pipeline"]
        return model.pipeline
    return model


def get_feature_bounds(model: Any) -> dict[str, dict[str, float]]:
    if is_surrogate(model):
        if isinstance(model, dict):
            return dict(model.get("feature_bounds") or {})
        return dict(model.feature_bounds)
    return dict(getattr(model, "feature_bounds_", {}) or {})


def load_pipeline(data: bytes, filename: str = "") -> Any:
    """Load a fitted model from pickle or joblib bytes."""
    name = filename.lower()
    buffer = BytesIO(data)
    errors: list[str] = []

    def _normalize(obj: Any) -> Any:
        # Legacy WTK dict export
        if is_surrogate(obj):
            return ensure_surrogate(obj)
        # Current export: sklearn pipeline/estimator with optional wtk_meta_
        meta = getattr(obj, "wtk_meta_", None)
        if isinstance(meta, dict):
            return SurrogateModel(
                pipeline=obj,
                feature_names=[str(n) for n in (meta.get("feature_names") or [])],
                feature_bounds={
                    str(k): dict(v) for k, v in dict(meta.get("feature_bounds") or {}).items()
                },
                model_name=meta.get("model_name"),
                task=meta.get("task"),
                feature_importance={
                    str(k): float(v) for k, v in dict(meta.get("feature_importance") or {}).items()
                },
                importance_method=meta.get("importance_method"),
                extra=dict(meta.get("extra") or {}),
            )
        return obj

    if name.endswith(".joblib") or name.endswith(".jl"):
        try:
            return _normalize(joblib.load(buffer))
        except Exception as exc:
            errors.append(f"joblib: {exc}")
            buffer.seek(0)
    if name.endswith(".pkl") or name.endswith(".pickle"):
        try:
            return _normalize(pickle.load(buffer))
        except Exception as exc:
            errors.append(f"pickle: {exc}")
            buffer.seek(0)
    buffer.seek(0)
    try:
        return _normalize(joblib.load(buffer))
    except Exception as exc:
        errors.append(f"joblib: {exc}")
    buffer.seek(0)
    try:
        return _normalize(pickle.load(buffer))
    except Exception as exc:
        errors.append(f"pickle: {exc}")
    raise ValueError("Could not load model as joblib or pickle. " + " | ".join(errors))


def _estimator_task(estimator: Any) -> str | None:
    from sklearn.base import ClassifierMixin, RegressorMixin

    if isinstance(estimator, ClassifierMixin):
        return "classification"
    if isinstance(estimator, RegressorMixin):
        return "regression"
    name = type(estimator).__name__.lower()
    if "classif" in name:
        return "classification"
    if "regress" in name or name.endswith("svr") or "gpr" in name:
        return "regression"
    return None


def describe_pipeline(model: Any) -> dict[str, Any]:
    """Summarize a loaded pipeline / SurrogateModel for Validation / Model Info pages."""
    model = ensure_surrogate(model) if is_surrogate(model) else model
    bounds = get_feature_bounds(model)
    pipe = get_pipeline(model)
    steps: list[str] = []
    estimator = pipe
    has_scaler = False
    if isinstance(pipe, Pipeline):
        steps = [f"{name}: {type(step).__name__}" for name, step in pipe.steps]
        has_scaler = any(name == "scaler" or "scaler" in type(step).__name__.lower() for name, step in pipe.steps)
        estimator = pipe.named_steps.get("model", pipe.steps[-1][1])

    feature_names: list[str] = []
    if is_surrogate(model) and getattr(model, "feature_names", None):
        feature_names = list(model.feature_names)
    else:
        for obj in (pipe, estimator):
            names = getattr(obj, "feature_names_in_", None)
            if names is not None:
                feature_names = [str(x) for x in names]
                break
    if not feature_names and bounds:
        feature_names = list(bounds.keys())

    n_features = getattr(pipe, "n_features_in_", None)
    if n_features is None:
        n_features = getattr(estimator, "n_features_in_", None)
    if n_features is None and feature_names:
        n_features = len(feature_names)

    params = {}
    try:
        raw = estimator.get_params(deep=False)
        for key, value in raw.items():
            text = repr(value)
            if len(text) > 120:
                text = text[:117] + "..."
            params[key] = text
    except Exception:
        params = {}

    task = None
    model_name = None
    stored_importance: dict[str, float] = {}
    importance_method = None
    if is_surrogate(model):
        task = getattr(model, "task", None)
        model_name = getattr(model, "model_name", None)
        stored_importance = dict(getattr(model, "feature_importance", None) or {})
        importance_method = getattr(model, "importance_method", None)
    task = task or _estimator_task(estimator)
    estimator_label = model_name or type(estimator).__name__

    return {
        "type": "SurrogateModel" if is_surrogate(model) else type(pipe).__name__,
        "estimator": estimator_label,
        "task": task,
        "steps": steps,
        "has_scaler": has_scaler,
        "n_features": int(n_features) if n_features is not None else None,
        "feature_names": feature_names,
        "feature_bounds": bounds,
        "feature_importance": stored_importance,
        "importance_method": importance_method,
        "params": params,
    }


def dump_pipeline(model: Any, fmt: str = "pickle") -> bytes:
    """Serialize a SurrogateModel or raw pipeline (pickle / joblib).

    Surrogates are written as the fitted sklearn estimator/pipeline (with a small
    ``wtk_meta_`` attribute for this app). External code can simply call
    ``model.predict(X)`` after ``pickle.load``.
    Legacy dict exports remain loadable via :func:`load_pipeline`.
    """
    if is_surrogate(model):
        model = ensure_surrogate(model)
        to_dump = get_pipeline(model)
        to_dump.wtk_meta_ = {
            "feature_names": [str(n) for n in getattr(model, "feature_names", [])],
            "feature_bounds": {
                str(k): {"min": float(v["min"]), "max": float(v["max"])}
                for k, v in dict(getattr(model, "feature_bounds", {}) or {}).items()
            },
            "model_name": getattr(model, "model_name", None),
            "task": getattr(model, "task", None),
            "feature_importance": {
                str(k): float(v)
                for k, v in dict(getattr(model, "feature_importance", {}) or {}).items()
            },
            "importance_method": getattr(model, "importance_method", None),
            "extra": dict(getattr(model, "extra", {}) or {}),
        }
    else:
        to_dump = model
    fmt = fmt.lower()
    if fmt == "joblib":
        buffer = BytesIO()
        joblib.dump(to_dump, buffer)
        return buffer.getvalue()
    if fmt == "pickle":
        return pickle.dumps(to_dump, protocol=pickle.HIGHEST_PROTOCOL)
    raise ValueError(f"Unknown model format: {fmt}")


def model_filename(stem: str, fmt: str) -> str:
    ext = ".joblib" if fmt.lower() == "joblib" else ".pkl"
    return f"{stem}{ext}"


def save_pipeline(model: Any, stem: str, fmt: str = "pickle") -> Path:
    """Write the fitted surrogate (with metadata) to models/ and return that path."""
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / model_filename(stem, fmt)
    path.write_bytes(dump_pipeline(model, fmt))
    return path


