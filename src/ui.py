"""Shared layout, theme, and sidebar navigation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap");

html, body, [class*="css"]  {
    font-family: "IBM Plex Sans", sans-serif;
}

.block-container { padding-top: 1.4rem; padding-bottom: 3rem; }

[data-testid="stSidebarNav"] { display: none; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #121A24 0%, #0C1218 100%);
    border-right: 1px solid rgba(208, 140, 74, 0.18);
}

.brand-mark {
    font-family: "IBM Plex Mono", monospace;
    letter-spacing: 0.28em;
    font-size: 0.72rem;
    color: #D08C4A;
    margin-bottom: 0.15rem;
}
.brand-title {
    font-size: 1.15rem;
    font-weight: 650;
    color: #F3F6FA;
    line-height: 1.2;
    margin-bottom: 0.35rem;
}
.brand-sub {
    color: #8AA0B8;
    font-size: 0.82rem;
    margin-bottom: 0.8rem;
}

.page-kicker {
    font-family: "IBM Plex Mono", monospace;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #D08C4A;
    font-size: 0.75rem;
    margin-bottom: 0.25rem;
}
.page-title {
    font-size: 1.85rem;
    font-weight: 650;
    margin-bottom: 0.2rem;
}
.page-lead { color: #9BB0C4; margin-bottom: 1.2rem; }

.stepper {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin: 0.4rem 0 1.3rem 0;
}
.step {
    border: 1px solid rgba(255,255,255,0.08);
    background: #1A2330;
    color: #8AA0B8;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
}
.step.active {
    border-color: #D08C4A;
    color: #F3F6FA;
    background: rgba(208, 140, 74, 0.16);
}

.hero-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.8rem;
    margin: 0.6rem 0 1.4rem 0;
}
.hero-card {
    background: #1A2330;
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid #D08C4A;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    min-height: 7.2rem;
}
.hero-card h4 {
    margin: 0 0 0.35rem 0;
    font-size: 0.95rem;
}
.hero-card p {
    margin: 0;
    color: #9BB0C4;
    font-size: 0.84rem;
    line-height: 1.4;
}
.hero-num {
    font-family: "IBM Plex Mono", monospace;
    color: #D08C4A;
    font-size: 0.75rem;
}

.model-family-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.5rem 0 0.4rem 0;
}
@media (max-width: 900px) {
    .model-family-grid {
        grid-template-columns: 1fr;
    }
}
.model-family-card {
    background: #1A2330;
    border: 1px solid rgba(255,255,255,0.06);
    border-top: 3px solid #D08C4A;
    border-radius: 10px;
    padding: 0.85rem 0.95rem;
}
.model-family-card h4 {
    margin: 0 0 0.55rem 0;
    font-size: 0.92rem;
    color: #F3F6FA;
}
.model-family-card ul {
    margin: 0;
    padding-left: 1.05rem;
    color: #9BB0C4;
    font-size: 0.82rem;
    line-height: 1.45;
}
.model-family-card li {
    margin: 0.12rem 0;
}

.nested-label {
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8AA0B8;
    margin: 0.55rem 0 0.2rem 0.15rem;
}

.st-key-reg_clear_cache button,
.st-key-clf_clear_cache button,
.st-key-bin_clear_cache button,
.st-key-multi_clear_cache button,
.st-key-doe_clear_cache button,
.st-key-doe_analysis_clear_cache button {
    background: rgba(220, 53, 69, 0.12) !important;
    border: 1px solid rgba(232, 80, 90, 0.95) !important;
    color: #F07178 !important;
    font-weight: 600 !important;
}
.st-key-reg_clear_cache button:hover,
.st-key-clf_clear_cache button:hover,
.st-key-bin_clear_cache button:hover,
.st-key-multi_clear_cache button:hover,
.st-key-doe_clear_cache button:hover,
.st-key-doe_analysis_clear_cache button:hover {
    background: rgba(220, 53, 69, 0.28) !important;
    border-color: #E85A64 !important;
    color: #FFD6D9 !important;
}
</style>
"""

STEPS = [
    ("1", "Introduction"),
    ("2", "DOE Sampling"),
    ("3", "Machine Learning"),
    ("4", "Validation"),
    ("5", "Model Info"),
    ("6", "Data Analysis"),
]

MAIN_PAGES = [
    ("introduction", "1", "Introduction"),
    ("doe", "2", "DOE Sampling"),
    ("ml", "3", "Machine Learning"),
    ("validation", "4", "Validation"),
    ("performance", "5", "Model Information & Visualization"),
    ("data_analysis", "6", "Data Analysis"),
]


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(kicker: str, title: str, lead: str, active_step: int) -> None:
    chips = []
    for idx, (num, label) in enumerate(STEPS, start=1):
        cls = "step active" if idx == active_step else "step"
        chips.append(f'<span class="{cls}">{num} {label}</span>')
    st.markdown(
        f"""
        <div class="page-kicker">{kicker}</div>
        <div class="page-title">{title}</div>
        <div class="page-lead">{lead}</div>
        <div class="stepper">{''.join(chips)}</div>
        """,
        unsafe_allow_html=True,
    )


def _nav_button(label: str, key: str, selected: bool) -> bool:
    return st.button(
        label,
        key=key,
        use_container_width=True,
        type="primary" if selected else "secondary",
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-mark">WTK</div>
            <div class="brand-title">Surrogate Model</div>
            <div class="brand-sub">DOE · train · validate · visualize</div>
            """,
            unsafe_allow_html=True,
        )

        for key, number, label in MAIN_PAGES:
            clicked = _nav_button(f"{number}  {label}", f"nav_{key}", st.session_state.nav_main == key)
            if clicked:
                st.session_state.nav_main = key
                st.rerun()

        route = st.session_state.nav_main
        if route == "ml":
            # Apply pending Import → Training jump before creating subpage radios
            goto = st.session_state.pop("_ml_goto_training", None)
            if goto == "regression":
                st.session_state.ml_family = "Regression"
                st.session_state.ml_family_radio = "Regression"
                st.session_state.ml_reg_step = "Training"
                st.session_state.ml_reg_radio = "Training"
            elif goto == "binary_classification":
                st.session_state.ml_family = "Binary Classification"
                st.session_state.ml_family_radio = "Binary Classification"
                st.session_state.ml_bin_step = "Training"
                st.session_state.ml_bin_radio = "Training"
            elif goto == "multiclass_classification":
                st.session_state.ml_family = "Multiclass Classification"
                st.session_state.ml_family_radio = "Multiclass Classification"
                st.session_state.ml_multi_step = "Training"
                st.session_state.ml_multi_radio = "Training"

            st.markdown('<div class="nested-label">Family</div>', unsafe_allow_html=True)
            family_options = ["Regression", "Binary Classification", "Multiclass Classification"]
            # Migrate legacy "Classification" session value
            if st.session_state.ml_family == "Classification":
                st.session_state.ml_family = "Binary Classification"
            family = st.radio(
                "Family",
                family_options,
                index=family_options.index(st.session_state.ml_family)
                if st.session_state.ml_family in family_options
                else 0,
                label_visibility="collapsed",
                key="ml_family_radio",
            )
            st.session_state.ml_family = family

            st.markdown('<div class="nested-label">Subpage</div>', unsafe_allow_html=True)
            if family == "Regression":
                options = ["Import Data", "Training", "Hyperparameter Optimization"]
                current = st.session_state.ml_reg_step
                step = st.radio(
                    "Regression step",
                    options,
                    index=options.index(current) if current in options else 0,
                    label_visibility="collapsed",
                    key="ml_reg_radio",
                )
                st.session_state.ml_reg_step = step
                mapping = {
                    "Import Data": "reg_import",
                    "Training": "reg_train",
                    "Hyperparameter Optimization": "reg_advance",
                }
                route = mapping[step]
            elif family == "Binary Classification":
                options = ["Import Data", "Training"]
                current = st.session_state.ml_bin_step
                step = st.radio(
                    "Binary classification step",
                    options,
                    index=options.index(current) if current in options else 0,
                    label_visibility="collapsed",
                    key="ml_bin_radio",
                )
                st.session_state.ml_bin_step = step
                route = "bin_import" if step == "Import Data" else "bin_train"
            else:
                options = ["Import Data", "Training"]
                current = st.session_state.ml_multi_step
                step = st.radio(
                    "Multiclass classification step",
                    options,
                    index=options.index(current) if current in options else 0,
                    label_visibility="collapsed",
                    key="ml_multi_radio",
                )
                st.session_state.ml_multi_step = step
                route = "multi_import" if step == "Import Data" else "multi_train"

        st.divider()
        st.caption("Session data stays in memory until you reload the browser.")
    return route


def _metric_value_color(name: str, value: float) -> str | None:
    """Return hex color for R² / R² adj / MAPE traffic-light scoring."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if name in {"R²", "R² adj"}:
        if value > 0.9:
            return "#1F8A4C"  # dark green
        if value > 0.8:
            return "#7BC47F"  # light green
        if value > 0.7:
            return "#E8A838"  # orange
        if value >= 0.6:
            return "#D96B2B"  # dark orange
        return "#D64545"  # red (< 0.6)
    if name == "MAPE (%)":
        if value < 10:
            return "#1F8A4C"  # dark green
        if value < 20:
            return "#7BC47F"  # light green
        if value < 30:
            return "#E8A838"  # light orange
        if value <= 40:
            return "#D96B2B"  # dark orange
        return "#D64545"  # red (> 40)
    return None


def show_model_info(info: dict, model_label: str) -> None:
    """Shared model information card (Validation / Model information pages)."""
    from src.ml import estimator_display_name

    st.subheader("Model information")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source", model_label)
    c2.metric("Estimator", estimator_display_name(info.get("estimator")))
    c3.metric("Task", str(info.get("task") or "unknown"))
    n_feat = info.get("n_features")
    c4.metric("Features", "—" if n_feat is None else str(n_feat))

    if info.get("steps"):
        st.caption("Pipeline steps: " + " → ".join(info["steps"]))
    st.caption(
        "Input scaling: "
        + ("included (`StandardScaler` in pipeline)" if info.get("has_scaler") else "not detected in pipeline")
    )
    if info.get("feature_names"):
        st.caption("Expected features: " + ", ".join(info["feature_names"]))

    bounds = info.get("feature_bounds") or {}
    if bounds:
        st.markdown("**Supported input range** (min / max from training X)")
        rows = [
            {"Feature": name, "Min": vals.get("min"), "Max": vals.get("max")}
            for name, vals in bounds.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Supported input range: not stored in this model file.")

    importance = info.get("feature_importance") or {}
    if importance:
        method = info.get("importance_method")
        title = "**Feature importance**"
        if method:
            title += f" ({method})"
        st.markdown(title)
        imp_rows = [{"Feature": name, "Importance": value} for name, value in importance.items()]
        imp_df = pd.DataFrame(imp_rows).sort_values("Importance", ascending=False)
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

    params = info.get("params") or {}
    if params:
        with st.expander("Model hyperparameters", expanded=True):
            st.json(params)


def show_metric_row(
    metrics: dict[str, float],
    *,
    title: str | None = None,
    n_samples: int | None = None,
) -> None:
    """Show metric cards with hover help (full name, good score, short description)."""
    from src.ml import METRIC_GUIDE, metric_help_text

    if title:
        st.subheader(title)
    if not metrics and n_samples is None:
        return

    n_cards = len(metrics) + (1 if n_samples is not None else 0)
    cols = st.columns(n_cards)
    for col, (name, value) in zip(cols, metrics.items()):
        help_text = (
            metric_help_text(name)
            .replace("\n", " · ")
            .replace('"', "&quot;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if value is None or (isinstance(value, float) and value != value):  # NaN
            shown = "—"
            color = None
        else:
            shown = f"{value:.3f}"
            color = _metric_value_color(name, float(value))
        value_style = f"color:{color};" if color else "color:#E6EDF3;"
        with col:
            st.markdown(
                f"""
                <div title="{help_text}" style="
                    background:#1A2330;
                    border:1px solid rgba(255,255,255,0.08);
                    border-radius:10px;
                    padding:0.75rem 0.9rem;
                    min-height:5.2rem;
                ">
                  <div style="font-size:0.82rem;color:#9BB0C4;margin-bottom:0.25rem;">{name}</div>
                  <div style="font-size:1.55rem;font-weight:650;{value_style}">{shown}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if n_samples is not None:
        with cols[-1]:
            st.markdown(
                f"""
                <div title="Number of samples used for this validation / test score." style="
                    background:#1A2330;
                    border:1px solid rgba(255,255,255,0.08);
                    border-radius:10px;
                    padding:0.75rem 0.9rem;
                    min-height:5.2rem;
                ">
                  <div style="font-size:0.82rem;color:#9BB0C4;margin-bottom:0.25rem;">Validation samples</div>
                  <div style="font-size:1.55rem;font-weight:650;color:#E6EDF3;">{int(n_samples)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Metric guide (full names & good-score notes)", expanded=False):
        for name in metrics:
            info = METRIC_GUIDE.get(name)
            if not info:
                continue
            st.markdown(
                f"**{name} — {info['full_name']}**  \n"
                f"*Good model:* {info['good']}  \n"
                f"{info['description']}"
            )
        st.caption(
            "Color bands — R² / R² adj: dark green >0.9, light green >0.8, orange >0.7, "
            "dark orange >0.6, red ≤0.6. MAPE (%): dark green <10, light green <20, "
            "light orange <30, dark orange <40, red ≥40. "
            "Classification scores: closer to 1 is better; see notes above for when to prefer "
            "F₀.₅, F₂, Macro-F1, or Micro-F1."
        )


def download_fitted_model(pfx: str, pipeline, key: str) -> None:
    from src.ml import dump_pipeline, ensure_surrogate, is_surrogate, model_filename, save_pipeline

    # Heal stale SurrogateModel instances left in session after Streamlit reloads src.ml
    if is_surrogate(pipeline):
        pipeline = ensure_surrogate(pipeline)
        st.session_state[f"{pfx}_pipeline"] = pipeline

    fmt = st.radio(
        "Download format",
        ("pickle", "joblib"),
        horizontal=True,
        key=key,
        help="pickle uses a .pkl file. joblib uses a .joblib file, which is often more efficient for scikit-learn models. "
        "Downloaded files include the fitted pipeline, each input's training min/max, and feature importance when available.",
    )
    stem = f"{pfx}_{st.session_state.get(f'{pfx}_model_name', 'model')}".replace(" ", "_")
    filename = model_filename(stem, fmt)
    payload = dump_pipeline(pipeline, fmt)
    clicked = st.download_button(
        f"Download fitted model ({filename.rsplit('.', 1)[-1]})",
        data=payload,
        file_name=filename,
        mime="application/octet-stream",
        key=f"{key}_button",
    )
    if clicked:
        path = save_pipeline(pipeline, stem, fmt)
        st.session_state[f"{pfx}_model_path"] = str(path)
        st.success(f"Model saved to `{path}` and ready to download.")
    saved = st.session_state.get(f"{pfx}_model_path")
    if saved:
        st.caption(f"Last saved file: `{saved}`")
