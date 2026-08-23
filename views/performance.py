"""Page 5 — Model information and surrogate visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.ml import describe_pipeline, get_feature_bounds, load_pipeline
from src.plots import response_2d, response_3d, response_surface_values
from src.state import FAMILY_LABELS, has_trained_model, ml_kind, prefix, task_from_family_label
from src.ui import download_fitted_model, page_header, show_model_info

# Grid density for surrogate slice axes (min → max)
_SLICE_NODES = 40


def _resolve_feature_bounds(
    pipeline,
    features: list[str],
    x_ref_data: pd.DataFrame | None = None,
) -> dict[str, dict[str, float]]:
    """Prefer model-stored ranges; fall back to reference CSV min/max."""
    stored = get_feature_bounds(pipeline) or {}
    resolved: dict[str, dict[str, float]] = {}
    for name in features:
        if name in stored and stored[name].get("min") is not None and stored[name].get("max") is not None:
            resolved[name] = {
                "min": float(stored[name]["min"]),
                "max": float(stored[name]["max"]),
            }
        elif x_ref_data is not None and name in x_ref_data.columns:
            series = pd.to_numeric(x_ref_data[name], errors="coerce")
            resolved[name] = {"min": float(series.min()), "max": float(series.max())}
    return resolved


def _midpoint_probe(features: list[str], bounds: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Hold every input at 0.5 * (min + max) of its supported range."""
    row = {name: 0.5 * (bounds[name]["min"] + bounds[name]["max"]) for name in features}
    return pd.DataFrame([row])


def _render_visuals(
    task: str,
    pipeline,
    features: list[str],
    *,
    x_ref_data: pd.DataFrame | None = None,
    comparison=None,
    download_key: str = "perf_dl",
    model_name: str = "model",
) -> None:
    if comparison is not None:
        st.subheader("Hyperparameter optimization results")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    if ml_kind(task) == "classification":
        st.info("Classification-specific visualization will be added later.")
        st.session_state["perf_model_name"] = model_name
        download_fitted_model("perf", pipeline, key=download_key)
        return

    if len(features) < 2:
        st.info("A 2D/3D response surface needs at least two input features.")
        st.session_state["perf_model_name"] = model_name
        download_fitted_model("perf", pipeline, key=download_key)
        return

    bounds = _resolve_feature_bounds(pipeline, features, x_ref_data)
    missing = [f for f in features if f not in bounds]
    if missing:
        st.warning(
            "Surrogate slices need min/max for each input. "
            "Retrain/download a model that stores feature ranges, or upload a reference CSV. "
            f"Missing: {', '.join(missing)}"
        )
        st.session_state["perf_model_name"] = model_name
        download_fitted_model("perf", pipeline, key=download_key)
        return

    st.subheader("Surrogate slices")
    st.caption(
        f"Selected inputs are gridded from min to max with {_SLICE_NODES} nodes. "
        "Other inputs are held at mid-range: 0.5 × (min + max)."
    )
    x_ref = _midpoint_probe(features, bounds)
    c1, c2 = st.columns(2)
    fx = c1.selectbox("X input", features, index=0, key=f"{download_key}_fx")
    fy = c2.selectbox("Y input", features, index=min(1, len(features) - 1), key=f"{download_key}_fy")
    if fx == fy:
        st.warning("Choose two different inputs.")
    else:
        gx = np.linspace(bounds[fx]["min"], bounds[fx]["max"], _SLICE_NODES)
        gy = np.linspace(bounds[fy]["min"], bounds[fy]["max"], _SLICE_NODES)
        _xx, _yy, zz = response_surface_values(pipeline, x_ref, fx, fy, gx, gy)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                response_2d(pipeline, x_ref, fx, fy, gx, gy, zz=zz),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                response_3d(pipeline, x_ref, fx, fy, gx, gy, zz=zz),
                use_container_width=True,
            )

    st.session_state["perf_model_name"] = model_name
    download_fitted_model("perf", pipeline, key=download_key)


def _stored_model_flow() -> None:
    available = []
    for task, label in FAMILY_LABELS.items():
        if has_trained_model(task):
            available.append(label)
    if not available:
        st.warning("No stored model in this session. Train one first, or choose **Upload ML model**.")
        st.stop()

    family = st.radio("Stored model family", available, horizontal=True, key="perf_stored_family")
    task = task_from_family_label(family)
    pfx = prefix(task)
    pipeline = st.session_state[f"{pfx}_pipeline"]
    features = list(st.session_state.get(f"{pfx}_features") or [])
    model_name = st.session_state.get(f"{pfx}_model_name", "model")

    info = describe_pipeline(pipeline)
    info["estimator"] = model_name if model_name else info.get("estimator")
    info["task"] = ml_kind(task)
    if not info.get("feature_names"):
        info["feature_names"] = features
    if not features:
        features = list(info.get("feature_names") or [])
    if info.get("n_features") is None:
        info["n_features"] = len(features)
    show_model_info(info, "Current Training Result")

    _render_visuals(
        task,
        pipeline,
        features,
        x_ref_data=st.session_state.get(f"{pfx}_X_train"),
        comparison=st.session_state.get(f"{pfx}_comparison"),
        download_key=f"{pfx}_dl_fmt_perf",
        model_name=str(model_name),
    )


def _uploaded_model_flow() -> None:
    uploaded_model = st.file_uploader(
        "Upload fitted model (.pkl or .joblib)",
        type=["pkl", "pickle", "joblib", "jl"],
        key="perf_upload_model",
    )
    if uploaded_model is None:
        st.info("Upload a pickle or joblib model to inspect information and visualizations.")
        st.stop()

    try:
        pipeline = load_pipeline(uploaded_model.getvalue(), uploaded_model.name)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

    info = describe_pipeline(pipeline)
    show_model_info(info, uploaded_model.name)

    expected = list(info.get("feature_names") or [])
    n_features = info.get("n_features")
    inferred_task = info.get("task")
    task_options = ["regression", "classification"]
    if inferred_task in task_options:
        task = st.radio(
            "Task type",
            task_options,
            index=task_options.index(inferred_task),
            horizontal=True,
            key="perf_upload_task",
        )
    else:
        task = st.radio("Task type", task_options, horizontal=True, key="perf_upload_task")

    features = list(expected)
    x_ref = None

    # Optional CSV only if the model lacks feature names / ranges
    needs_csv = task == "regression" and (
        not features or len(get_feature_bounds(pipeline) or {}) < 2
    )
    st.subheader("Reference CSV" + (" (required for ranges)" if needs_csv else " (optional)"))
    if needs_csv:
        st.caption("This model has incomplete feature names or ranges. Upload a CSV to enable surrogate slices.")
    else:
        st.caption("Optional. Used only if you want to override feature names or fill missing ranges.")
    uploaded_csv = st.file_uploader("CSV file", type=["csv"], key="perf_upload_csv")

    if uploaded_csv is not None:
        frame = pd.read_csv(uploaded_csv)
        all_cols = frame.columns.tolist()
        numeric_cols = frame.select_dtypes(include="number").columns.tolist()
        default_features = [c for c in expected if c in all_cols]
        if not default_features and n_features and len(numeric_cols) >= int(n_features):
            default_features = numeric_cols[: int(n_features)]
        if not default_features:
            default_features = numeric_cols

        features = st.multiselect(
            "Feature columns (X)",
            options=all_cols,
            default=default_features,
            key="perf_upload_features",
        )
        if not features:
            st.warning("Select one or more feature columns.")
            st.stop()
        if n_features is not None and len(features) != int(n_features):
            st.warning(f"Model expects {n_features} features; you selected {len(features)}.")
        x_ref = frame.dropna(subset=features)[features]
    elif not features:
        st.info("Upload a CSV so feature names can be set.")
        st.stop()

    _render_visuals(
        task,
        pipeline,
        features,
        x_ref_data=x_ref,
        download_key="perf_upload_dl",
        model_name=info.get("estimator") or "uploaded_model",
    )


def render() -> None:
    page_header(
        "Page 05",
        "Model Information & Visualization",
        "Inspect a stored or uploaded surrogate: parameters, ranges, and response surfaces.",
        active_step=5,
    )

    source = st.radio(
        "Model source",
        ["Stored ML model", "Upload ML model"],
        horizontal=True,
        key="perf_model_source",
    )
    if source == "Stored ML model":
        _stored_model_flow()
    else:
        _uploaded_model_flow()
