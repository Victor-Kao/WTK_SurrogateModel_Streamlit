"""Training and advance-training subpages."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import cross_val_predict

from src.ml import (
    CHOICES,
    CLASSIFIERS,
    REGRESSORS,
    assignment_export_frame,
    binary_positive_probability,
    compare_models,
    default_hyperparams,
    ensure_surrogate,
    feature_importance,
    fit_and_evaluate,
    hyperparam_schema,
    is_surrogate,
    labels_from_binary_scores,
    make_cv_splitter,
    predict_class_scores,
    roc_curve_table,
    row_assignments,
    score_predictions,
    split_type_label,
    split_xy,
    wrap_surrogate,
)
from src.plots import confusion_heatmap, importance_bar, parity_plot, residual_plot, roc_auc_plot
from src.state import TASK_BINARY, family_label, ml_kind, prefix
from src.ui import download_fitted_model, page_header, show_metric_row

SCHEMES = ("Split Dataset", "CV folds", "LOOCV")


def _require_data(task: str) -> tuple[str, pd.DataFrame, list[str], str]:
    pfx = prefix(task)
    frame = st.session_state.get(f"{pfx}_df")
    features = st.session_state.get(f"{pfx}_features") or []
    target = st.session_state.get(f"{pfx}_target")
    if frame is None or not features or not target:
        st.warning("Import a dataset on the Import Data subpage first.")
        st.stop()
    return pfx, frame, features, target


def _row_ids(frame: pd.DataFrame) -> pd.Index:
    if "id" in frame.columns:
        return pd.Index(frame["id"], name="id")
    if "sample_id" in frame.columns:
        return pd.Index(frame["sample_id"], name="id")
    return pd.Index(range(1, len(frame) + 1), name="id")


def _preview_xy(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    extra: pd.Series | None = None,
    heading: str | None = None,
) -> None:
    ids = _row_ids(frame)
    x_preview = frame[features].copy()
    y_preview = frame[[target]].copy()
    x_preview.index = ids
    y_preview.index = ids
    if extra is not None:
        x_preview[extra.name] = extra.to_numpy()
        y_preview[extra.name] = extra.to_numpy()
    if heading:
        st.subheader(heading)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**X · features**")
        st.caption(f"{len(x_preview)} rows × {x_preview.shape[1]} columns · id shown at left")
        st.dataframe(x_preview, use_container_width=True, hide_index=False, height=280)
    with right:
        st.markdown("**Y · target**")
        st.caption(f"{len(y_preview)} rows × {y_preview.shape[1]} columns (`{target}`) · id shown at left")
        st.dataframe(y_preview, use_container_width=True, hide_index=False, height=280)


def _validation_controls(pfx: str, n_rows: int) -> tuple[str, int, float, int]:
    st.subheader("Validation scheme")
    scheme = st.radio(
        "Method",
        SCHEMES,
        horizontal=True,
        key=f"{pfx}_val_scheme_radio",
        help="Split Dataset holds out a fraction once. CV folds rotate validation blocks. LOOCV leaves one row out each fold.",
    )
    n_folds = 5
    val_fraction = 0.2
    seed = 42
    if scheme == "CV folds":
        n_folds = st.slider("Number of folds", min_value=2, max_value=10, value=5, key=f"{pfx}_n_folds")
        seed = st.number_input("Shuffle seed", min_value=0, value=42, step=1, key=f"{pfx}_cv_seed")
    elif scheme == "Split Dataset":
        val_fraction = st.slider(
            "Validation fraction",
            min_value=0.05,
            max_value=0.50,
            value=0.20,
            step=0.05,
            format="%.2f",
            key=f"{pfx}_val_fraction",
            help="Fraction of rows held out for validation. Default 0.20 (20%).",
        )
        st.caption(f"{val_fraction:.0%} of the data is used for validation ({max(1, round(n_rows * val_fraction))} rows).")
        seed = st.number_input("Split seed", min_value=0, value=42, step=1, key=f"{pfx}_split_seed")
    else:
        st.caption(f"Leave-one-out uses {n_rows} folds (one row held out at a time).")
        if n_rows > 250:
            st.warning(f"LOOCV will fit {n_rows} models and can be slow on this dataset.")

    return scheme, int(n_folds), float(val_fraction), int(seed)


def _assign_snapshot(scheme: str, n_folds: int, val_fraction: float, seed: int, n_rows: int, features: list[str], target: str) -> tuple:
    return (scheme, n_folds, val_fraction, seed, n_rows, tuple(features), target)


def _assignment_preview(
    pfx: str,
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    scheme: str,
    n_folds: int,
    val_fraction: float,
    seed: int,
    stratify: bool,
) -> None:
    if scheme == "LOOCV":
        return

    if st.button("Refresh table", type="primary", key=f"{pfx}_refresh_assign"):
        try:
            assignment = row_assignments(
                frame, features, target, scheme, val_fraction, n_folds, seed, stratify
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state[f"{pfx}_assign_snapshot"] = _assign_snapshot(
            scheme, n_folds, val_fraction, seed, len(frame), features, target
        )
        st.session_state[f"{pfx}_assign_table"] = assignment
        st.session_state[f"{pfx}_assign_stamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    current = _assign_snapshot(scheme, n_folds, val_fraction, seed, len(frame), features, target)
    committed = st.session_state.get(f"{pfx}_assign_snapshot")
    assignment = st.session_state.get(f"{pfx}_assign_table")
    if committed != current or assignment is None:
        if assignment is None:
            st.warning("Click **Refresh table** to show the train/test assignment.")
        else:
            st.warning("Validation settings changed. Click **Refresh table** to update the train/test assignment.")
        return

    heading = (
        "X / Y with data type (Training or Testing)"
        if scheme == "Split Dataset"
        else "X / Y with CV fold ID"
    )
    _preview_xy(frame, features, target, extra=assignment, heading=heading)

    export = assignment_export_frame(frame, features, target, assignment)
    stamp = st.session_state.get(f"{pfx}_assign_stamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    split_tag = split_type_label(scheme)
    file_name = f"TRAIN_DATA_{split_tag}_{stamp}.csv"
    st.download_button(
        "Download X+Y assignment CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=f"{pfx}_download_assign_csv",
        help="Combined features, target, and Training/Testing or CV fold ID for each row.",
    )


def _assignment_is_fresh(
    pfx: str,
    scheme: str,
    n_folds: int,
    val_fraction: float,
    seed: int,
    n_rows: int,
    features: list[str],
    target: str,
) -> bool:
    if scheme == "LOOCV":
        return True
    current = _assign_snapshot(scheme, n_folds, val_fraction, seed, n_rows, features, target)
    committed = st.session_state.get(f"{pfx}_assign_snapshot")
    return committed == current and st.session_state.get(f"{pfx}_assign_table") is not None


def _persist_result(pfx: str, result: dict, model_name: str, task: str) -> None:
    pipeline = result["pipeline"]
    if not is_surrogate(pipeline):
        pipeline = wrap_surrogate(
            pipeline,
            result["X_train"],
            y_data=result["y_train"],
            model_name=model_name,
            task=task,
        )
    else:
        pipeline = ensure_surrogate(pipeline)
    st.session_state[f"{pfx}_pipeline"] = pipeline
    st.session_state[f"{pfx}_metrics"] = result["metrics"]
    st.session_state[f"{pfx}_X_train"] = result["X_train"]
    st.session_state[f"{pfx}_X_test"] = result["X_test"]
    st.session_state[f"{pfx}_y_train"] = result["y_train"]
    st.session_state[f"{pfx}_y_test"] = result["y_test"]
    st.session_state[f"{pfx}_y_pred_train"] = result["y_pred_train"]
    st.session_state[f"{pfx}_y_pred_test"] = result["y_pred_test"]
    st.session_state[f"{pfx}_metric_label"] = result["metric_label"]
    st.session_state[f"{pfx}_model_name"] = model_name
    st.session_state[f"{pfx}_model_path"] = None
    st.session_state[f"{pfx}_feature_bounds"] = pipeline.feature_bounds
    st.session_state[f"{pfx}_feature_importance"] = pipeline.feature_importance
    st.session_state[f"{pfx}_importance_method"] = pipeline.importance_method


def _show_metrics(metrics: dict, label: str, n_samples: int | None = None) -> None:
    show_metric_row(metrics["test"], title=label, n_samples=n_samples)


def _show_diagnostics(
    task: str,
    y_test,
    y_pred,
    pipeline,
    features,
    x_test=None,
    *,
    pfx: str = "clf",
) -> None:
    kind = ml_kind(task)
    if kind == "regression":
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(parity_plot(y_test.to_numpy(), y_pred), use_container_width=True)
        with c2:
            st.plotly_chart(residual_plot(y_test.to_numpy(), y_pred), use_container_width=True)
    else:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                confusion_heatmap(y_test.to_numpy(), y_pred),
                use_container_width=False,
            )
        with right:
            roc_frame = None
            auc_summary: dict = {}
            if x_test is not None:
                try:
                    scores, class_labels = predict_class_scores(pipeline, x_test)
                    if scores is not None and class_labels is not None:
                        roc_frame, auc_summary = roc_curve_table(
                            y_test.to_numpy(), scores, class_labels
                        )
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
                    file_name="roc_auc_curve.csv",
                    mime="text/csv",
                    key=f"{pfx}_roc_auc_download",
                )
            else:
                st.info(
                    "ROC/AUC requires class probabilities or decision scores "
                    "(e.g. models with `predict_proba`)."
                )

    importance = feature_importance(pipeline, features)
    if importance is not None:
        method = getattr(pipeline, "importance_method", None)
        if method:
            st.caption(f"Feature importance method: {method}")
        st.plotly_chart(importance_bar(importance), use_container_width=True)


def _show_binary_results(
    pfx: str,
    pipeline,
    y_test,
    x_test,
    features: list[str],
    metric_label: str,
) -> None:
    """Metrics / CM / table driven by predict_proba + user threshold (same pickle)."""
    y_obs = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)
    pack = binary_positive_probability(pipeline, x_test)
    if pack is None:
        y_pred = pipeline.predict(x_test)
        metrics = {"test": score_predictions("classification", y_obs, y_pred, n_features=len(features))}
        st.warning("Probability scores unavailable; showing hard `.predict()` labels.")
        _show_metrics(metrics, metric_label, n_samples=len(y_obs))
        _show_diagnostics(TASK_BINARY, y_test, y_pred, pipeline, features, x_test=x_test, pfx=pfx)
        return

    probs, neg_label, pos_label, score_kind = pack
    st.subheader("Decision threshold")
    if score_kind == "probability":
        st.caption(
            "Same fitted model — `predicted_probability` is P(positive) from `predict_proba`. "
            "No second pickle file. Labels use your threshold (not default `.predict()`)."
        )
    else:
        st.caption(
            "No `predict_proba` on this estimator; `predicted_score` is sigmoid(`decision_function`)."
        )
    threshold = st.slider(
        "Positive-class threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key=f"{pfx}_binary_threshold",
        help="Label = positive if score ≥ threshold, else negative. "
        "Example: 0.75 with threshold 0.80 → class 0.",
    )
    st.caption(f"Negative label: `{neg_label}` · Positive label: `{pos_label}`")
    y_pred = labels_from_binary_scores(probs, threshold, neg_label, pos_label)
    metrics = {
        "test": score_predictions("classification", y_obs, y_pred, n_features=len(features)),
    }
    _show_metrics(metrics, f"{metric_label} (threshold = {threshold:.2f})", n_samples=len(y_obs))
    _show_diagnostics(TASK_BINARY, y_test, y_pred, pipeline, features, x_test=x_test, pfx=pfx)

    preview = x_test.reset_index(drop=True).copy()
    col_name = "predicted_probability" if score_kind == "probability" else "predicted_score"
    preview["observed"] = y_obs
    preview[col_name] = probs
    preview["predicted"] = y_pred
    preview["threshold"] = threshold
    st.subheader("Predictions")
    st.caption("Score column + label at the selected threshold")
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions CSV",
        data=preview.to_csv(index=False).encode("utf-8"),
        file_name="binary_predictions.csv",
        mime="text/csv",
        key=f"{pfx}_binary_pred_download",
    )


def _parse_manual_value(kind: str, raw: Any, default: Any) -> Any:
    if kind == "bool":
        return bool(raw)
    if kind == "choice":
        if raw in {"None", "none", None}:
            return None
        return raw
    text = str(raw).strip()
    if kind == "optional_int":
        if text.lower() in {"", "none", "null"}:
            return None
        return int(float(text))
    if kind == "tuple_int":
        cleaned = text.strip("()[] ")
        if not cleaned:
            return default
        return tuple(int(float(part.strip())) for part in cleaned.split(",") if part.strip())
    if kind == "int":
        return int(float(text))
    if kind == "float":
        return float(text)
    if kind == "str":
        if text.lower() in {"none", "null"}:
            return None
        try:
            return float(text)
        except ValueError:
            return text
    return raw


def _bool_select(label: str, default: bool, key: str) -> bool:
    """Bool control that matches selectbox size/layout (label above, full width)."""
    options = ["True", "False"]
    chosen = st.selectbox(
        label,
        options,
        index=0 if default else 1,
        key=key,
    )
    return chosen == "True"


def _algorithm_setup(task: str, pfx: str, models: dict) -> tuple[str, bool, dict]:
    st.subheader("Algorithm Setup")
    c1, c2 = st.columns(2)
    with c1:
        model_name = st.selectbox("Algorithm", list(models), key=f"{pfx}_algo_select")
    with c2:
        scale = _bool_select("Standardize features", True, key=f"{pfx}_algo_scale")

    schema = hyperparam_schema(task, model_name)
    defaults = default_hyperparams(task, model_name)
    # GPR: normalize Y by default
    if model_name == "GPR":
        defaults["normalize_y"] = True
    params: dict = {}
    if not schema:
        st.caption("This algorithm has no editable hyperparameters in the current setup.")
        return model_name, scale, params

    st.markdown("**Hyperparameters**")
    st.caption("Defaults below are the current model settings. Edit freely — no range limits are enforced.")
    cols = st.columns(2)
    for idx, (name, (kind, default)) in enumerate(schema.items()):
        col = cols[idx % 2]
        key = f"{pfx}_hp_{model_name}_{name}".replace(" ", "_")
        value_default = defaults.get(name, default)
        with col:
            if kind == "bool":
                # Prefer defaults override (e.g. GPR normalize_y=True)
                params[name] = _bool_select(name, bool(value_default), key=key)
            elif kind == "choice":
                if name == "kernel" and model_name in {"GPR", "GPC"}:
                    options = CHOICES.get("gp_kernel", [str(default)])
                else:
                    options = CHOICES.get(name, [str(default)])
                display = [("None" if opt is None else str(opt)) for opt in options]
                current = "None" if value_default is None else str(value_default)
                if current not in display:
                    display = [current] + display
                chosen = st.selectbox(name, display, index=display.index(current), key=key)
                params[name] = _parse_manual_value(kind, chosen, value_default)
            else:
                shown = "None" if value_default is None else str(value_default)
                raw = st.text_input(name, value=shown, key=key)
                try:
                    params[name] = _parse_manual_value(kind, raw, value_default)
                except Exception:
                    st.error(f"Invalid value for `{name}`. Using default `{shown}`.")
                    params[name] = value_default

    with st.expander("Current hyperparameter values", expanded=False):
        st.json(params)
    return model_name, scale, params


def render(task: str, advanced: bool = False) -> None:
    kind = ml_kind(task)
    is_reg = kind == "regression"
    family = family_label(task)
    title = "Advance Training" if advanced else "Training"
    lead = (
        "Compare algorithms with the selected validation scheme and keep the best pipeline."
        if advanced
        else "Fit a baseline surrogate using a hold-out split, k-fold CV, or LOOCV."
    )
    page_header(f"Page 03 · {family}", title, lead, active_step=3)

    pfx, frame, features, target = _require_data(task)
    models = REGRESSORS if is_reg else CLASSIFIERS
    _preview_xy(frame, features, target)
    scheme, n_folds, val_fraction, seed = _validation_controls(pfx, len(frame))
    stratify = not is_reg and scheme != "LOOCV"
    _assignment_preview(pfx, frame, features, target, scheme, n_folds, val_fraction, seed, stratify)

    assign_ready = _assignment_is_fresh(pfx, scheme, n_folds, val_fraction, seed, len(frame), features, target)

    if assign_ready:
        if advanced:
            selected = st.multiselect(
                "Models to compare",
                list(models),
                default=[name for name in ("Ridge", "Random Forest", "Gradient Boosting", "SVR") if name in models][:3]
                or list(models)[:3],
            )
            c1, c2 = st.columns(2)
            n_iter = c1.slider("Search iterations", 4, 20, 8)
            with c2:
                scale = _bool_select("Standardize features", True, key=f"{pfx}_adv_algo_scale")
            if st.button(
                "Run comparison",
                type="primary",
                disabled=len(selected) == 0,
            ):
                if not _assignment_is_fresh(pfx, scheme, n_folds, val_fraction, seed, len(frame), features, target):
                    st.error("Refresh the assignment table before training.")
                    st.stop()
                stratify = not is_reg and scheme != "LOOCV"
                try:
                    if scheme == "Split Dataset":
                        x_train, x_test, y_train, y_test = split_xy(
                            frame, features, target, val_fraction, seed, stratify=stratify
                        )
                        inner_cv = make_cv_splitter("CV folds", 5, seed, stratify=stratify)
                        search_x, search_y = x_train, y_train
                    else:
                        x_train = x_test = frame[features]
                        y_train = y_test = frame[target]
                        inner_cv = make_cv_splitter(scheme, n_folds, seed, stratify=stratify)
                        search_x, search_y = x_train, y_train
                    with st.spinner("Cross-validating models…"):
                        table, winner, pipeline = compare_models(
                            kind, selected, search_x, search_y, scale, n_iter, seed, inner_cv
                        )
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
                y_pred_train = pipeline.predict(x_train)
                if scheme == "Split Dataset":
                    y_pred_test = pipeline.predict(x_test)
                else:
                    y_pred_test = cross_val_predict(pipeline, x_train, y_train, cv=inner_cv)
                metrics = {
                    "train": score_predictions(kind, y_train.to_numpy(), y_pred_train, n_features=x_train.shape[1]),
                    "test": score_predictions(kind, y_test.to_numpy(), y_pred_test, n_features=x_test.shape[1]),
                }
                label = "Hold-out validation" if scheme == "Split Dataset" else (
                    "LOOCV" if scheme == "LOOCV" else f"{n_folds}-fold CV"
                )
                st.session_state[f"{pfx}_comparison"] = table
                _persist_result(
                    pfx,
                    {
                        "pipeline": pipeline,
                        "metrics": metrics,
                        "X_train": x_train,
                        "X_test": x_test,
                        "y_train": y_train,
                        "y_test": y_test,
                        "y_pred_train": y_pred_train,
                        "y_pred_test": y_pred_test,
                        "metric_label": label,
                    },
                    winner,
                    kind,
                )
                st.success(f"Best model by CV: **{winner}**. Use Download fitted model to save a file.")

        else:
            model_name, scale, params = _algorithm_setup(kind, pfx, models)
            if st.button("Train model", type="primary"):
                if not _assignment_is_fresh(pfx, scheme, n_folds, val_fraction, seed, len(frame), features, target):
                    st.error("Refresh the assignment table before training.")
                    st.stop()
                try:
                    with st.spinner("Fitting…"):
                        result = fit_and_evaluate(
                            kind,
                            model_name,
                            frame,
                            features,
                            target,
                            scale,
                            scheme,
                            val_fraction,
                            n_folds,
                            seed,
                            params=params,
                        )
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
                _persist_result(pfx, result, model_name, kind)
                st.success(
                    f"Trained {model_name} with {result['metric_label']}. "
                    "Use Download fitted model to save a file."
                )

    if advanced:
        table = st.session_state.get(f"{pfx}_comparison")
        if table is not None:
            st.dataframe(table, use_container_width=True, hide_index=True)

    pipeline = st.session_state.get(f"{pfx}_pipeline")
    metrics = st.session_state.get(f"{pfx}_metrics")
    y_test = st.session_state.get(f"{pfx}_y_test")
    y_pred = st.session_state.get(f"{pfx}_y_pred_test")
    x_test = st.session_state.get(f"{pfx}_X_test")
    if pipeline is None or metrics is None:
        return

    metric_label = st.session_state.get(f"{pfx}_metric_label", "Validation metrics")
    if task == TASK_BINARY and y_test is not None and x_test is not None:
        _show_binary_results(pfx, pipeline, y_test, x_test, features, metric_label)
    else:
        _show_metrics(
            metrics,
            metric_label,
            n_samples=None if y_test is None else len(y_test),
        )
        _show_diagnostics(task, y_test, y_pred, pipeline, features, x_test=x_test, pfx=pfx)
    download_fitted_model(pfx, pipeline, key=f"{pfx}_dl_fmt_train")
