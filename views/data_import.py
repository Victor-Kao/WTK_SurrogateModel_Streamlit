"""Shared import-data subpage for regression and classification families."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.benchmarks import list_problems, load_problem
from src.state import clear_ml_data, family_label, ml_kind, prefix
from src.ui import page_header


def render(task: str) -> None:
    is_reg = ml_kind(task) == "regression"
    pfx = prefix(task)
    page_header(
        "Page 03 · " + family_label(task),
        "Import Data",
        "Load a table, choose inputs and the response, then store the dataset for training.",
        active_step=3,
    )

    source_col, clear_col = st.columns([3.2, 1], vertical_alignment="bottom")
    with source_col:
        source = st.radio(
            "Data source",
            ["Upload CSV", "Load demo dataset"],
            horizontal=True,
            key=f"{pfx}_import_source",
        )
    with clear_col:
        if st.button("Clear cache", use_container_width=True, key=f"{pfx}_clear_cache"):
            clear_ml_data(task)
            st.rerun()

    frame: pd.DataFrame | None = None
    editor_key = f"{pfx}_editor_upload"
    if source == "Upload CSV":
        uploaded = st.file_uploader("CSV file", type=["csv"], key=f"{pfx}_csv_upload")
        if uploaded is not None:
            frame = pd.read_csv(uploaded)
    else:
        problems = list_problems(task)
        selected_title = st.selectbox(
            "Benchmark problem",
            [item.title for item in problems],
            key=f"{pfx}_benchmark_select",
        )
        problem = next(item for item in problems if item.title == selected_title)
        st.caption(problem.description)
        try:
            frame = load_problem(problem)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        editor_key = f"{pfx}_editor_{problem.id}"
        st.success(f"Loaded **{problem.title}** · {len(frame)} samples · target `{problem.target}`.")

    if frame is not None:
        _import_form(task, pfx, is_reg, frame, editor_key=editor_key)

    stored = st.session_state.get(f"{pfx}_df")
    if stored is not None:
        st.divider()
        st.subheader("Currently stored")
        st.write(
            f"{len(stored)} rows · features: {', '.join(st.session_state[f'{pfx}_features'])} · "
            f"target: `{st.session_state[f'{pfx}_target']}`"
        )
        st.dataframe(stored.head(8), use_container_width=True, hide_index=True)


def _import_form(task: str, pfx: str, is_reg: bool, frame: pd.DataFrame, editor_key: str) -> None:
    st.subheader("Preview")
    edited = st.data_editor(frame, num_rows="dynamic", use_container_width=True, hide_index=True, key=editor_key)
    st.caption(f"{edited.shape[0]} rows × {edited.shape[1]} columns")

    numeric_cols = edited.select_dtypes(include="number").columns.tolist()
    all_cols = edited.columns.tolist()
    if not numeric_cols:
        st.error("Need at least one numeric column for features.")
        st.stop()

    default_target = "y" if is_reg and "y" in all_cols else ("label" if "label" in all_cols else all_cols[-1])
    default_features = [c for c in numeric_cols if c not in {default_target, "sample_id", "id"}]

    c1, c2 = st.columns(2)
    with c1:
        features = st.multiselect(
            "Feature columns",
            options=numeric_cols,
            default=default_features,
            key=f"{editor_key}_features",
        )
    with c2:
        target = st.selectbox(
            "Target column",
            options=all_cols,
            index=all_cols.index(default_target) if default_target in all_cols else 0,
            key=f"{editor_key}_target",
        )

    if not features:
        st.warning("Select one or more feature columns.")
        st.stop()
    if target in features:
        st.error("Target cannot also be a feature.")
        st.stop()
    if edited[target].isna().any() or edited[features].isna().any().any():
        st.warning("Missing values detected. Rows with NA in the selected columns will be dropped on confirm.")

    if st.button("Store dataset for training", type="primary", key=f"{pfx}_store_dataset"):
        clean = edited.dropna(subset=features + [target]).copy()
        if "id" not in clean.columns:
            if "sample_id" in clean.columns:
                clean.insert(0, "id", clean["sample_id"].to_numpy())
            else:
                clean.insert(0, "id", range(1, len(clean) + 1))
        feature_cols = [c for c in features if c not in {"id", "sample_id"}]
        if not is_reg:
            clean[target] = clean[target].astype("category")
            n_classes = int(clean[target].nunique())
            if task == "binary_classification" and n_classes != 2:
                st.error(
                    f"Binary Classification requires exactly 2 classes in `{target}`; "
                    f"found {n_classes}. Use Multiclass Classification or remap the target."
                )
                st.stop()
            if task == "multiclass_classification" and n_classes < 3:
                st.error(
                    f"Multiclass Classification expects 3+ classes in `{target}`; "
                    f"found {n_classes}. Use Binary Classification for 2-class problems."
                )
                st.stop()
        st.session_state[f"{pfx}_df"] = clean
        st.session_state[f"{pfx}_features"] = feature_cols
        st.session_state[f"{pfx}_target"] = target
        st.session_state[f"{pfx}_pipeline"] = None
        st.session_state[f"{pfx}_metrics"] = None
        # Sidebar radios are created before this page runs, so we cannot set their
        # widget keys here. Flag the jump; render_sidebar applies it on the next run.
        st.session_state.nav_main = "ml"
        st.session_state["_ml_goto_training"] = task
        st.rerun()
