"""Page 2 — DOE sampling."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.doe import (
    AUGMENT_METHOD_NOTES,
    AUGMENT_METHODS,
    METHODS,
    default_variables,
    design_bounds_from_frame,
    design_feature_columns,
    generate_next_doe_locations,
    generate_samples,
    parse_bounds_csv,
)
from src.state import clear_doe_analysis_data, clear_doe_data
from src.plots import doe_augment_scatter
from src.ui import page_header

DOE_STEPS = ("DOE Sample Generate", "DOE Sample Analysis")


def _render_generate() -> None:
    title_col, clear_col = st.columns([3.2, 1], vertical_alignment="bottom")
    with title_col:
        st.caption("Generated designs stay in session until you clear them or replace them.")
    with clear_col:
        if st.button("Clear cache", use_container_width=True, key="doe_clear_cache"):
            clear_doe_data()
            st.rerun()

    if st.session_state.doe_variables is None:
        st.session_state.doe_variables = default_variables()

    left, right = st.columns((1.05, 1.15), gap="large")
    with left:
        st.subheader("Input variables")
        edited = st.data_editor(
            st.session_state.doe_variables,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "name": st.column_config.TextColumn("Name", required=True),
                "min": st.column_config.NumberColumn("Min"),
                "max": st.column_config.NumberColumn("Max"),
                "type": st.column_config.SelectboxColumn("Type", options=["continuous", "integer"]),
            },
            key="doe_editor",
        )
        st.session_state.doe_variables = edited

        n_dim = max(len(edited.dropna(subset=["name"])), 1)
        st.caption(f"Rule of thumb for Latin Hypercube: start near {10 * n_dim} samples (10 × number of inputs).")

    with right:
        st.subheader("Sampler")
        method = st.selectbox("Method", METHODS, index=METHODS.index(st.session_state.doe_method))
        st.session_state.doe_method = method
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        levels = 3
        n_samples = 40
        if method == "Full Factorial":
            levels = st.slider("Levels per factor", min_value=2, max_value=6, value=3)
            n_dim = max(len(edited.dropna(subset=["name"])), 1)
            st.info(f"This design will contain {levels}^{n_dim} = {levels ** n_dim} rows.")
        else:
            n_samples = st.slider("Number of samples", min_value=8, max_value=2000, value=40, step=1)
            if method == "Sobol":
                power = 2 ** math.ceil(math.log2(max(n_samples, 2)))
                if power != n_samples:
                    st.caption(f"Sobol sequences are better balanced at powers of two (nearest: {power}).")

        generate = st.button("Generate design", type="primary", use_container_width=True)
        if generate:
            try:
                samples = generate_samples(edited, method, n_samples, seed=int(seed), levels=int(levels))
                st.session_state.doe_samples = samples
                st.success(f"Generated {len(samples)} samples.")
            except Exception as exc:
                st.error(str(exc))

    samples: pd.DataFrame | None = st.session_state.doe_samples
    if samples is None:
        st.info("Generate a design to preview it here. You can reuse it later on Import Data.")
        return

    matrix_col, preview_col = st.columns((1.05, 1.15), gap="large")
    with matrix_col:
        st.subheader("Design matrix")
        st.dataframe(samples, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            data=samples.to_csv(index=False).encode("utf-8"),
            file_name="doe_samples.csv",
            mime="text/csv",
        )

    with preview_col:
        numeric = samples.drop(columns=["sample_id"], errors="ignore").select_dtypes("number")
        st.subheader("Pairwise preview")
        if numeric.shape[1] >= 2:
            x_col, y_col = st.columns(2)
            with x_col:
                x_name = st.selectbox("Parameter (X axis)", numeric.columns.tolist(), index=0)
            with y_col:
                y_name = st.selectbox(
                    "Parameter (Y axis)",
                    numeric.columns.tolist(),
                    index=min(1, numeric.shape[1] - 1),
                )
            st.scatter_chart(samples, x=x_name, y=y_name)
        else:
            st.info("Need at least two numeric inputs for a pairwise preview.")


def _render_analysis() -> None:
    title_col, clear_col = st.columns([3.2, 1], vertical_alignment="bottom")
    with title_col:
        st.caption("Analyze a design from Generate, or upload a DOE sample CSV.")
    with clear_col:
        if st.button("Clear cache", use_container_width=True, key="doe_analysis_clear_cache"):
            clear_doe_analysis_data()
            st.rerun()

    source_col, _ = st.columns([3.2, 1])
    with source_col:
        source = st.radio(
            "Sample source",
            ["Generated DOE sample", "Upload DOE sample"],
            horizontal=True,
            key="doe_analysis_source",
        )

    samples: pd.DataFrame | None = None
    if source == "Generated DOE sample":
        samples = st.session_state.get("doe_samples")
        if samples is None:
            st.warning("No generated DOE sample in this session. Switch to **DOE Sample Generate**, or choose **Upload DOE sample**.")
            return
        st.success(f"Loaded generated DOE sample ({len(samples)} rows × {samples.shape[1]} columns).")
    else:
        uploaded = st.file_uploader(
            "DOE sample CSV",
            type=["csv"],
            key="doe_analysis_upload",
        )
        if uploaded is None:
            cached = st.session_state.get("doe_analysis_samples")
            if cached is None:
                st.info("Upload a DOE sample CSV to continue.")
                return
            samples = cached
            st.caption("Using previously uploaded DOE sample from this session.")
        else:
            try:
                samples = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Failed to read CSV: {exc}")
                return
            st.session_state.doe_analysis_samples = samples
            st.success(f"Uploaded DOE sample ({len(samples)} rows × {samples.shape[1]} columns).")

    if samples is None or samples.empty:
        st.warning("DOE sample is empty.")
        return

    st.subheader("Loaded design")
    st.dataframe(samples, use_container_width=True, hide_index=True)

    features = design_feature_columns(samples)
    if not features:
        st.warning("No numeric feature columns found in the loaded design.")
        return

    st.subheader("Design Next DOE location")
    st.caption(
        "Set the exploration box for each feature (defaults = current design min/max; "
        "you may enlarge or shift the space, including negative values), choose an algorithm, "
        "and generate additional points."
    )

    default_bounds = design_bounds_from_frame(samples, features)
    # Keep editor defaults in sync when the loaded design identity changes
    design_sig = (tuple(features), int(len(samples)), float(default_bounds["min"].sum()), float(default_bounds["max"].sum()))
    if st.session_state.get("doe_next_bounds_sig") != design_sig:
        st.session_state.doe_next_bounds = default_bounds.copy()
        st.session_state.doe_next_bounds_sig = design_sig
        for key in ("doe_next_bounds_upload", "doe_next_bounds_draft"):
            if key in st.session_state:
                del st.session_state[key]

    if st.session_state.get("doe_next_bounds") is None:
        st.session_state.doe_next_bounds = default_bounds.copy()

    mode_col, reset_col = st.columns([3.2, 1], vertical_alignment="bottom")
    with mode_col:
        bounds_mode = st.radio(
            "Exploration bounds",
            ["Edit in app", "Upload bounds CSV"],
            horizontal=True,
            key="doe_next_bounds_mode",
        )
    with reset_col:
        if st.button("Reset to design min/max", use_container_width=True, key="doe_next_bounds_reset"):
            st.session_state.doe_next_bounds = default_bounds.copy()
            for key in ("doe_next_bounds_upload", "doe_next_bounds_draft"):
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    if bounds_mode == "Upload bounds CSV":
        st.caption(
            "CSV columns: `name` (or feature), `min`, `max`. "
            "Values may be positive or negative; max must be greater than min for each feature."
        )
        template = st.session_state.doe_next_bounds.copy()
        st.download_button(
            "Download bounds template CSV",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="doe_bounds_template.csv",
            mime="text/csv",
            key="doe_next_bounds_template",
        )
        uploaded_bounds = st.file_uploader(
            "Bounds CSV",
            type=["csv"],
            key="doe_next_bounds_upload",
        )
        if uploaded_bounds is not None:
            try:
                raw_bounds = pd.read_csv(uploaded_bounds)
                st.session_state.doe_next_bounds = parse_bounds_csv(raw_bounds, features)
                st.success("Bounds loaded from CSV. You can switch to **Edit in app** to fine-tune.")
            except Exception as exc:
                st.error(str(exc))
        bounds_edited = st.session_state.doe_next_bounds
        st.dataframe(bounds_edited, use_container_width=True, hide_index=True)
    else:
        st.caption(
            "Edit min/max freely (any signed values). Click **Apply bounds** when finished — "
            "this prevents Streamlit from resetting cells while you type."
        )
        # Form isolates edits until Apply, avoiding data_editor session-state backtracking.
        with st.form("doe_next_bounds_form", clear_on_submit=False):
            draft = st.data_editor(
                st.session_state.doe_next_bounds,
                num_rows="fixed",
                use_container_width=True,
                hide_index=True,
                disabled=["name"],
                column_config={
                    "name": st.column_config.TextColumn("Feature"),
                    "min": st.column_config.NumberColumn(
                        "Min",
                        min_value=None,
                        max_value=None,
                        help="Lower bound (any real number, positive or negative).",
                    ),
                    "max": st.column_config.NumberColumn(
                        "Max",
                        min_value=None,
                        max_value=None,
                        help="Upper bound (any real number, positive or negative).",
                    ),
                },
            )
            applied = st.form_submit_button("Apply bounds", use_container_width=True)
        if applied:
            try:
                check = draft.copy()
                check["min"] = pd.to_numeric(check["min"], errors="coerce")
                check["max"] = pd.to_numeric(check["max"], errors="coerce")
                if check[["min", "max"]].isna().any().any():
                    raise ValueError("Min/max must be numeric.")
                if (check["max"] <= check["min"]).any():
                    raise ValueError("Each feature max must be greater than min.")
                st.session_state.doe_next_bounds = check[["name", "min", "max"]]
                st.success("Bounds applied.")
            except Exception as exc:
                st.error(str(exc))
        bounds_edited = st.session_state.doe_next_bounds

    c1, c2, c3 = st.columns(3)
    with c1:
        # Prefer LHS as default; if an old invalid session value exists, fall back.
        current_method = st.session_state.get("doe_next_method", AUGMENT_METHODS[0])
        if current_method not in AUGMENT_METHODS:
            current_method = AUGMENT_METHODS[0]
            st.session_state.doe_next_method = current_method
        method = st.selectbox(
            "Next-location algorithm",
            AUGMENT_METHODS,
            index=AUGMENT_METHODS.index(current_method),
            key="doe_next_method",
        )
    with c2:
        n_new = st.number_input(
            "Additional points",
            min_value=1,
            max_value=500,
            value=5,
            step=1,
            key="doe_next_n_new",
        )
    with c3:
        seed = st.number_input(
            "Random seed",
            min_value=0,
            value=42,
            step=1,
            key="doe_next_seed",
        )

    with st.expander("How this algorithm works", expanded=False):
        st.caption(AUGMENT_METHOD_NOTES.get(method, "No description available."))

    n_candidates = 4000
    boundary_weight_alpha = 0.1
    boundary_eps = 1e-6
    if method == "Greedy Maximin Distance (boundary-weighted)":
        st.markdown("**Boundary-weighted maximin hyperparameters**")
        h1, h2, h3 = st.columns(3)
        with h1:
            boundary_weight_alpha = st.number_input(
                "Boundary weight α",
                min_value=0.0,
                max_value=5.0,
                value=0.1,
                step=0.05,
                format="%.2f",
                key="doe_next_boundary_alpha",
                help="score = min_distance × (distance_to_boundary)^α. "
                "0 = plain maximin; larger α prefers more interior points.",
            )
        with h2:
            n_candidates = st.number_input(
                "Candidate pool size",
                min_value=200,
                max_value=20000,
                value=4000,
                step=100,
                key="doe_next_n_candidates",
                help="Number of random candidates evaluated at each greedy step.",
            )
        with h3:
            boundary_eps = st.number_input(
                "Boundary ε",
                min_value=1e-12,
                max_value=1e-2,
                value=1e-6,
                format="%.1e",
                key="doe_next_boundary_eps",
                help="Small floor added to boundary distance to keep scores finite near edges.",
            )
    elif method == "Greedy Maximin Distance":
        n_candidates = st.number_input(
            "Candidate pool size",
            min_value=200,
            max_value=20000,
            value=4000,
            step=100,
            key="doe_next_n_candidates_plain",
            help="Number of random candidates evaluated at each greedy step.",
        )

    if st.button("Generate next DOE locations", type="primary", use_container_width=True, key="doe_next_generate"):
        try:
            next_points = generate_next_doe_locations(
                samples,
                bounds_edited,
                method=method,
                n_new=int(n_new),
                seed=int(seed),
                n_candidates=int(n_candidates),
                boundary_weight_alpha=float(boundary_weight_alpha),
                boundary_eps=float(boundary_eps),
            )
            st.session_state.doe_next_samples = next_points
            st.success(f"Generated {len(next_points)} additional DOE location(s).")
        except Exception as exc:
            st.error(str(exc))

    next_points = st.session_state.get("doe_next_samples")
    if next_points is not None:
        plot_features = [f for f in features if f in next_points.columns]
        table_col, plot_col = st.columns((1.05, 1.15), gap="large")
        with table_col:
            st.subheader("Next DOE locations")
            st.dataframe(next_points, use_container_width=True, hide_index=True)
            # Align columns so the merged CSV keeps original fields + new rows
            merge_cols: list[str] = []
            for col in list(samples.columns) + list(next_points.columns):
                if col not in merge_cols:
                    merge_cols.append(col)
            merged = pd.concat(
                [samples.reindex(columns=merge_cols), next_points.reindex(columns=merge_cols)],
                ignore_index=True,
            )
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "Download new DOE locations",
                    data=next_points.to_csv(index=False).encode("utf-8"),
                    file_name="doe_next_locations.csv",
                    mime="text/csv",
                    key="doe_next_download_new",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    "Download merged DOE (original + new)",
                    data=merged.to_csv(index=False).encode("utf-8"),
                    file_name="doe_merged_locations.csv",
                    mime="text/csv",
                    key="doe_next_download_merged",
                    use_container_width=True,
                )
            st.caption(f"Merged design: {len(samples)} original + {len(next_points)} new = {len(merged)} rows.")
        with plot_col:
            st.subheader("Location preview")
            if len(plot_features) < 2:
                st.info("Need at least two numeric parameters to plot.")
            else:
                x_col, y_col = st.columns(2)
                with x_col:
                    x_name = st.selectbox(
                        "Parameter (X axis)",
                        plot_features,
                        index=0,
                        key="doe_next_plot_x",
                    )
                with y_col:
                    y_name = st.selectbox(
                        "Parameter (Y axis)",
                        plot_features,
                        index=min(1, len(plot_features) - 1),
                        key="doe_next_plot_y",
                    )
                if x_name == y_name:
                    st.warning("Choose two different parameters.")
                else:
                    st.plotly_chart(
                        doe_augment_scatter(samples, next_points, x_name, y_name),
                        use_container_width=True,
                    )


def render() -> None:
    page_header(
        "Page 02",
        "DOE Sampling",
        "Generate a design matrix or analyze an existing DOE sample set.",
        active_step=2,
    )

    current = st.session_state.get("doe_step", DOE_STEPS[0])
    if current not in DOE_STEPS:
        current = DOE_STEPS[0]
    step = st.radio(
        "DOE mode",
        DOE_STEPS,
        index=DOE_STEPS.index(current),
        horizontal=True,
        key="doe_step_radio",
        label_visibility="collapsed",
    )
    st.session_state.doe_step = step

    if step == "DOE Sample Generate":
        _render_generate()
    else:
        _render_analysis()
