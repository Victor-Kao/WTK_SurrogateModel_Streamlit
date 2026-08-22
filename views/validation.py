"""Page 4 — Validation against a hold-out or independent set."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.ml import (
    binary_positive_probability,
    describe_pipeline,
    labels_from_binary_scores,
    load_pipeline,
    predict_class_scores,
    roc_curve_table,
    score_predictions,
)
from src.plots import confusion_heatmap, error_histogram, parity_plot, residual_plot, roc_auc_plot
from src.state import (
    FAMILY_LABELS,
    TASK_BINARY,
    TASK_MULTICLASS,
    TASK_REGRESSION,
    family_label,
    has_trained_model,
    ml_kind,
    prefix,
    task_from_family_label,
)
from src.ui import page_header, show_metric_row, show_model_info


def _run_validation(
    task: str,
    pipeline,
    features: list[str],
    target: str,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    source_df: pd.DataFrame | None = None,
) -> None:
    kind = ml_kind(task)
    y_obs = pd.Series(y_val).to_numpy()

    binary_pack = None
    threshold = 0.5
    if task == TASK_BINARY:
        binary_pack = binary_positive_probability(pipeline, x_val)
        if binary_pack is not None:
            st.subheader("Decision threshold")
            score_kind = binary_pack[3]
            if score_kind == "probability":
                st.caption(
                    "Same fitted model — scores come from `predict_proba` (P(positive class)). "
                    "Labels are cut at your threshold (not sklearn’s default `.predict()`)."
                )
            else:
                st.caption(
                    "This estimator has no `predict_proba`; scores are a sigmoid of "
                    "`decision_function` (approximate probability scale)."
                )
            threshold = st.slider(
                "Positive-class threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                key="val_binary_threshold",
                help="Label = positive if score ≥ threshold, else negative. "
                "Example: score 0.75 with threshold 0.80 → negative (0).",
            )
            probs, neg_label, pos_label, _ = binary_pack
            st.caption(f"Negative label: `{neg_label}` · Positive label: `{pos_label}`")
            y_pred = labels_from_binary_scores(probs, threshold, neg_label, pos_label)
        else:
            st.warning(
                "Binary probability scores unavailable for this model; "
                "falling back to hard `.predict()` labels."
            )
            y_pred = pipeline.predict(x_val)
    else:
        y_pred = pipeline.predict(x_val)

    metrics = score_predictions(kind, y_obs, y_pred, n_features=len(features))

    preview = x_val.reset_index(drop=True).copy()
    if (
        isinstance(source_df, pd.DataFrame)
        and "id" in source_df.columns
        and x_val.index.isin(source_df.index).all()
    ):
        preview.insert(0, "id", source_df.loc[x_val.index, "id"].to_numpy())
    preview["observed"] = y_obs
    if binary_pack is not None:
        probs, _neg, _pos, score_kind = binary_pack
        col_name = "predicted_probability" if score_kind == "probability" else "predicted_score"
        preview[col_name] = probs
        preview["predicted"] = y_pred
        preview["threshold"] = threshold
    else:
        preview["predicted"] = y_pred

    try:
        y_obs_num = np.asarray(y_obs, dtype=float)
        y_pred_num = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError):
        y_obs_num = pd.Series(y_obs).astype("category").cat.codes.to_numpy(dtype=float)
        y_pred_num = pd.Series(y_pred).astype("category").cat.codes.to_numpy(dtype=float)
    signed_err = y_obs_num - y_pred_num
    if kind == "regression":
        preview["abs error"] = np.abs(signed_err)

    st.session_state.val_task = task
    st.session_state.val_metrics = metrics
    st.session_state.val_predictions = preview

    metric_title = "Validation scores"
    if task == TASK_BINARY and binary_pack is not None:
        metric_title = f"Validation scores (threshold = {threshold:.2f})"
    show_metric_row(metrics, title=metric_title, n_samples=len(preview))

    if kind == "regression":
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(parity_plot(y_obs, y_pred, "Validation parity"), use_container_width=True)
        with c2:
            st.plotly_chart(residual_plot(y_obs, y_pred), use_container_width=True)
    else:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(confusion_heatmap(y_obs, y_pred), use_container_width=False)
        with right:
            roc_frame = None
            auc_summary: dict = {}
            try:
                scores, class_labels = predict_class_scores(pipeline, x_val)
                if scores is not None and class_labels is not None:
                    roc_frame, auc_summary = roc_curve_table(y_obs, scores, class_labels)
            except Exception as exc:
                st.warning(f"ROC/AUC unavailable: {exc}")
            if roc_frame is not None and not roc_frame.empty:
                if "AUC" in auc_summary:
                    st.metric("AUC", f"{auc_summary['AUC']:.4f}")
                elif "AUC micro" in auc_summary:
                    st.metric("AUC (micro)", f"{auc_summary['AUC micro']:.4f}")
                st.plotly_chart(roc_auc_plot(roc_frame, auc_summary), use_container_width=True)
                st.download_button(
                    "Download ROC-AUC curve CSV",
                    data=roc_frame.to_csv(index=False).encode("utf-8"),
                    file_name="validation_roc_auc_curve.csv",
                    mime="text/csv",
                    key="val_roc_auc_download",
                )
            else:
                st.info(
                    "ROC/AUC requires class probabilities or decision scores "
                    "(e.g. models with `predict_proba`)."
                )

    st.subheader("Predictions")
    if binary_pack is not None:
        st.caption(
            f"Features ({', '.join(features)}) · observed `{target}` · "
            "predicted probability/score · predicted label at threshold"
        )
    else:
        st.caption(f"Features ({', '.join(features)}) · observed `{target}` · predicted · abs error")
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions CSV",
        data=preview.to_csv(index=False).encode("utf-8"),
        file_name="validation_predictions.csv",
        mime="text/csv",
    )

    if kind == "regression":
        err_mean = float(np.mean(signed_err))
        err_std = float(np.std(signed_err, ddof=1)) if len(signed_err) > 1 else 0.0
        st.subheader("Error distribution")
        m1, m2 = st.columns(2)
        m1.metric("Mean error", f"{err_mean:.6g}")
        m2.metric("Std error", f"{err_std:.6g}")
        st.plotly_chart(error_histogram(signed_err, err_mean, err_std), use_container_width=True)


def _stored_model_flow() -> None:
    available = []
    for task_id, label in FAMILY_LABELS.items():
        if has_trained_model(task_id):
            available.append(label)
    if not available:
        st.warning("No stored model in this session. Train one first, or choose **Upload ML model**.")
        st.stop()

    family = st.radio("Stored model family", available, horizontal=True, key="val_stored_family")
    task = task_from_family_label(family)
    pfx = prefix(task)
    pipeline = st.session_state[f"{pfx}_pipeline"]
    features = list(st.session_state[f"{pfx}_features"])
    target = st.session_state[f"{pfx}_target"]
    model_name = st.session_state.get(f"{pfx}_model_name", "model")

    info = describe_pipeline(pipeline)
    info["estimator"] = model_name if model_name else info.get("estimator")
    info["task"] = ml_kind(task)
    if not info.get("feature_names"):
        info["feature_names"] = features
    if info.get("n_features") is None:
        info["n_features"] = len(features)
    show_model_info(info, "Current Training Result")

    mode = st.radio(
        "Validation data",
        ["Stored test split", "Upload independent CSV"],
        horizontal=True,
        key="val_stored_data_mode",
    )
    if mode == "Stored test split":
        x_val = st.session_state.get(f"{pfx}_X_test")
        y_val = st.session_state.get(f"{pfx}_y_test")
        if x_val is None or y_val is None:
            st.error("No stored test split. Re-train the model after importing data.")
            st.stop()
        _run_validation(task, pipeline, features, target, x_val, y_val, st.session_state.get(f"{pfx}_df"))
        return

    uploaded = st.file_uploader("Labeled independent CSV", type=["csv"], key="val_stored_csv")
    if uploaded is None:
        st.info("Upload a labeled CSV that contains the feature columns and the target.")
        st.stop()
    frame = pd.read_csv(uploaded)
    missing = [col for col in features + [target] if col not in frame.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()
    clean = frame.dropna(subset=features + [target])
    _run_validation(task, pipeline, features, target, clean[features], clean[target], clean)


def _uploaded_model_flow() -> None:
    uploaded_model = st.file_uploader(
        "Upload fitted model (.pkl or .joblib)",
        type=["pkl", "pickle", "joblib", "jl"],
        key="val_upload_model",
    )
    if uploaded_model is None:
        st.info("Upload a pickle or joblib model exported from Training / Model Information.")
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
    task_options = [TASK_REGRESSION, TASK_BINARY, TASK_MULTICLASS]
    task_labels = [family_label(t) for t in task_options]
    if inferred_task == "regression":
        default_idx = 0
    elif inferred_task == "classification":
        default_idx = 1
    else:
        default_idx = 0
    chosen = st.radio(
        "Task type",
        task_labels,
        index=default_idx,
        horizontal=True,
        key="val_upload_task",
        help="Inferred from the estimator when possible; pick Binary vs Multiclass to match your labels.",
    )
    task = task_from_family_label(chosen)

    st.subheader("Independent validation CSV")
    uploaded_csv = st.file_uploader("Upload labeled CSV", type=["csv"], key="val_upload_csv")
    if uploaded_csv is None:
        st.info("After the model loads, upload an independent labeled CSV for validation.")
        st.stop()

    frame = pd.read_csv(uploaded_csv)
    all_cols = frame.columns.tolist()
    numeric_cols = frame.select_dtypes(include="number").columns.tolist()

    default_features = [c for c in expected if c in all_cols]
    if not default_features and n_features and len(numeric_cols) >= int(n_features):
        default_features = numeric_cols[: int(n_features)]
    if not default_features:
        default_features = [c for c in numeric_cols if c != all_cols[-1]]

    c1, c2 = st.columns(2)
    with c1:
        features = st.multiselect(
            "Feature columns (X)",
            options=all_cols,
            default=default_features,
            key="val_upload_features",
        )
    with c2:
        remaining = [c for c in all_cols if c not in features] or all_cols
        default_target = remaining[-1]
        target = st.selectbox(
            "Target column (observed Y)",
            options=all_cols,
            index=all_cols.index(default_target) if default_target in all_cols else 0,
            key="val_upload_target",
        )

    if not features:
        st.warning("Select one or more feature columns.")
        st.stop()
    if target in features:
        st.error("Target cannot also be a feature.")
        st.stop()
    if n_features is not None and len(features) != int(n_features):
        st.warning(f"Model expects {n_features} features; you selected {len(features)}.")
    if expected and features != expected:
        missing = [c for c in expected if c not in features]
        extra = [c for c in features if c not in expected]
        if missing or extra:
            st.warning(
                "Selected features differ from the model’s stored feature names. "
                f"Missing expected: {missing or '—'} · Extra: {extra or '—'}."
            )

    clean = frame.dropna(subset=features + [target]).copy()
    if clean.empty:
        st.error("No rows left after dropping missing values in the selected columns.")
        st.stop()

    try:
        _run_validation(task, pipeline, features, target, clean[features], clean[target], clean)
    except Exception as exc:
        st.error(f"Prediction failed. Check feature order/names and task type. Details: {exc}")


def render() -> None:
    page_header(
        "Page 04",
        "Validation",
        "Validate a stored session model or an uploaded pickle/joblib surrogate on labeled data.",
        active_step=4,
    )

    source = st.radio(
        "Model source",
        ["Stored ML model", "Upload ML model"],
        horizontal=True,
        key="val_model_source",
    )
    if source == "Stored ML model":
        _stored_model_flow()
    else:
        _uploaded_model_flow()
