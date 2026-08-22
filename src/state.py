"""Session-state keys shared across pages."""

from __future__ import annotations

import streamlit as st

# ML family task ids used across Import / Training / Validation / Model Info
TASK_REGRESSION = "regression"
TASK_BINARY = "binary_classification"
TASK_MULTICLASS = "multiclass_classification"

FAMILY_LABELS = {
    TASK_REGRESSION: "Regression",
    TASK_BINARY: "Binary Classification",
    TASK_MULTICLASS: "Multiclass Classification",
}

_PREFIX = {
    TASK_REGRESSION: "reg",
    TASK_BINARY: "bin",
    TASK_MULTICLASS: "multi",
    # legacy single classification family (pre-split)
    "classification": "clf",
}


def _empty_ml_defaults(pfx: str) -> dict:
    return {
        f"{pfx}_df": None,
        f"{pfx}_features": [],
        f"{pfx}_target": None,
        f"{pfx}_X_train": None,
        f"{pfx}_X_test": None,
        f"{pfx}_y_train": None,
        f"{pfx}_y_test": None,
        f"{pfx}_y_pred_train": None,
        f"{pfx}_y_pred_test": None,
        f"{pfx}_pipeline": None,
        f"{pfx}_metrics": None,
        f"{pfx}_model_name": None,
        f"{pfx}_comparison": None,
    }


_DEFAULTS = {
    "nav_main": "introduction",
    "ml_family": "Regression",
    "ml_reg_step": "Import Data",
    "ml_bin_step": "Import Data",
    "ml_multi_step": "Import Data",
    "ml_clf_step": "Import Data",  # legacy
    "doe_variables": None,
    "doe_samples": None,
    "doe_method": "Latin Hypercube",
    "doe_step": "DOE Sample Generate",
    "doe_analysis_samples": None,
    "doe_next_samples": None,
    **_empty_ml_defaults("reg"),
    **_empty_ml_defaults("bin"),
    **_empty_ml_defaults("multi"),
    **_empty_ml_defaults("clf"),
    "val_df": None,
    "val_metrics": None,
    "val_predictions": None,
    "val_task": None,
}


def init_state() -> None:
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def prefix(task: str) -> str:
    return _PREFIX.get(task, "clf")


def ml_kind(task: str) -> str:
    """Map family task id → algorithm family used by ``src.ml`` ('regression' | 'classification')."""
    if task == TASK_REGRESSION or task == "regression":
        return "regression"
    return "classification"


def family_label(task: str) -> str:
    return FAMILY_LABELS.get(task, str(task).replace("_", " ").title())


def task_from_family_label(label: str) -> str:
    for task, name in FAMILY_LABELS.items():
        if name == label:
            return task
    if label == "Classification":
        return TASK_BINARY
    return TASK_REGRESSION


def is_classification_task(task: str) -> bool:
    return ml_kind(task) == "classification"


def has_trained_model(task: str) -> bool:
    return st.session_state.get(f"{prefix(task)}_pipeline") is not None


_ML_SUFFIXES = (
    "df",
    "features",
    "target",
    "val_scheme",
    "n_folds",
    "val_fraction",
    "split_seed",
    "metric_label",
    "assign_snapshot",
    "assign_table",
    "assign_stamp",
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "y_pred_train",
    "y_pred_test",
    "pipeline",
    "metrics",
    "model_name",
    "model_path",
    "feature_bounds",
    "feature_importance",
    "importance_method",
    "comparison",
)


def clear_ml_data(task: str) -> None:
    """Remove stored dataset, split, and fitted model for one ML family."""
    pfx = prefix(task)
    for suffix in _ML_SUFFIXES:
        key = f"{pfx}_{suffix}"
        if key in _DEFAULTS:
            st.session_state[key] = _DEFAULTS[key]
        elif key in st.session_state:
            del st.session_state[key]
    if st.session_state.get("val_task") == task:
        st.session_state.val_df = None
        st.session_state.val_metrics = None
        st.session_state.val_predictions = None
        st.session_state.val_task = None


def clear_doe_data() -> None:
    """Remove generated DOE samples and reset variable / method defaults."""
    st.session_state.doe_samples = None
    st.session_state.doe_variables = None
    st.session_state.doe_method = "Latin Hypercube"
    # Drop data_editor widget state so the table resets cleanly
    for key in ("doe_editor",):
        if key in st.session_state:
            del st.session_state[key]
    clear_doe_analysis_data()


def clear_doe_analysis_data() -> None:
    """Remove DOE Sample Analysis upload / working cache."""
    st.session_state.doe_analysis_samples = None
    st.session_state.doe_next_samples = None
    for key in (
        "doe_analysis_upload",
        "doe_analysis_source",
        "doe_next_bounds_editor",
        "doe_next_bounds_upload",
        "doe_next_bounds_mode",
        "doe_next_bounds_sig",
    ):
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.doe_next_bounds = None
