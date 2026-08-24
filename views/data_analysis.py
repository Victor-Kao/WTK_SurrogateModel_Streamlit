"""Page 6 — Data analysis (visualization + correlation)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.benchmarks import PROBLEMS, load_problem
from src.plots import correlation_heatmap, feature_scatter
from src.ui import page_header

_CORR_OPTIONS = ("Pearson", "Spearman", "Kendall’s Tau")
_CORR_METHODS = {
    "Pearson": ("pearson", "r"),
    "Spearman": ("spearman", "ρ"),
    "Kendall’s Tau": ("kendall", "τ"),
}


def _all_problems():
    return (
        list(PROBLEMS["regression"])
        + list(PROBLEMS["binary_classification"])
        + list(PROBLEMS["multiclass_classification"])
    )


def _load_frame() -> pd.DataFrame | None:
    source = st.radio(
        "Data source",
        ["Upload CSV", "Load demo dataset"],
        horizontal=True,
        key="da_import_source",
    )

    if source == "Upload CSV":
        uploaded = st.file_uploader("CSV file", type=["csv"], key="da_csv_upload")
        if uploaded is None:
            st.info("Upload a CSV to explore visualizations and correlation.")
            return None
        frame = pd.read_csv(uploaded)
        st.success(f"Loaded **{uploaded.name}** · {len(frame)} rows × {frame.shape[1]} columns.")
        return frame

    problems = _all_problems()
    family = st.radio(
        "Demo family",
        ["Regression", "Binary Classification", "Multiclass Classification"],
        horizontal=True,
        key="da_demo_family",
    )
    family_map = {
        "Regression": "regression",
        "Binary Classification": "binary_classification",
        "Multiclass Classification": "multiclass_classification",
    }
    task = family_map[family]
    options = [p for p in problems if p.task == task]
    selected_title = st.selectbox(
        "Benchmark problem",
        [item.title for item in options],
        key="da_benchmark_select",
    )
    problem = next(item for item in options if item.title == selected_title)
    st.caption(problem.description)
    try:
        frame = load_problem(problem)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    st.success(
        f"Loaded **{problem.title}** · {len(frame)} samples · target `{problem.target}`."
    )
    return frame


def _default_corr_method(n_samples: int) -> str:
    """Spearman for larger samples; Kendall’s Tau when n ≤ 750."""
    return "Spearman" if n_samples > 750 else "Kendall’s Tau"


def _numeric_plot_cols(frame: pd.DataFrame) -> list[str]:
    numeric = frame.select_dtypes(include="number")
    cols = [c for c in numeric.columns if str(c).lower() not in {"id", "sample_id"}]
    return cols if cols else list(numeric.columns)


def _render_data_visualization(frame: pd.DataFrame, plot_cols: list[str]) -> None:
    st.subheader("Data visualization")
    st.caption("Choose two parameters to inspect their relationship as a 2D scatter plot.")
    if len(plot_cols) < 2:
        st.warning("Need at least two numeric columns for a scatter plot.")
        return

    c1, c2 = st.columns(2)
    x_name = c1.selectbox("X-axis", plot_cols, index=0, key="da_scatter_x")
    y_options = [c for c in plot_cols if c != x_name] or list(plot_cols)
    preferred_y = plot_cols[1] if len(plot_cols) > 1 else y_options[0]
    y_index = y_options.index(preferred_y) if preferred_y in y_options else 0
    y_name = c2.selectbox("Y-axis", y_options, index=y_index, key="da_scatter_y")
    if x_name == y_name:
        st.warning("Choose two different parameters for X and Y.")
        return

    st.plotly_chart(
        feature_scatter(frame, x_name, y_name),
        use_container_width=True,
        key="da_scatter_chart",
    )


def render() -> None:
    page_header(
        "Page 06",
        "Data Analysis",
        "Upload a CSV or load a demo dataset, then explore scatter plots and correlation.",
        active_step=6,
    )

    frame = _load_frame()
    if frame is None:
        return

    st.subheader("Preview")
    st.dataframe(frame.head(12), use_container_width=True, hide_index=True)
    st.caption(f"{frame.shape[0]} rows × {frame.shape[1]} columns")

    plot_cols = _numeric_plot_cols(frame)
    if len(plot_cols) < 2:
        st.warning("Need at least two numeric columns for visualization and correlation.")
        return

    _render_data_visualization(frame, plot_cols)

    st.divider()

    selected = st.multiselect(
        "Columns for correlation",
        options=list(frame.select_dtypes(include="number").columns),
        default=plot_cols,
        key="da_corr_columns",
    )
    if len(selected) < 2:
        st.warning("Select at least two numeric columns.")
        return

    subset = frame[selected]
    n_samples = len(subset)
    default_method = _default_corr_method(n_samples)
    size_bucket = "large" if n_samples > 750 else "small"

    st.subheader("Correlation matrix")
    st.caption(
        f"Sample size n = {n_samples}. "
        "Default is **Spearman** when n > 750, otherwise **Kendall’s Tau** "
        "(Pearson remains available)."
    )
    method_label = st.radio(
        "Correlation method",
        _CORR_OPTIONS,
        index=_CORR_OPTIONS.index(default_method),
        horizontal=True,
        key=f"da_corr_method_{size_bucket}",
    )
    method, coeff = _CORR_METHODS[method_label]
    corr = subset.corr(method=method)

    left, right = st.columns((1.15, 1), gap="large")
    with left:
        st.plotly_chart(
            correlation_heatmap(
                corr,
                title=f"{method_label} correlation",
                coeff_label=coeff,
            ),
            use_container_width=True,
            key="da_corr_chart",
        )
    with right:
        st.markdown("**Numeric correlation table**")
        st.dataframe(
            corr.round(3),
            use_container_width=True,
            height=min(720, max(380, 42 * len(corr) + 140)),
        )
