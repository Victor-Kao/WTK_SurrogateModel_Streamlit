"""Plotly figures used by training, validation, and performance pages."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COPPER = "#D08C4A"
TEAL = "#4AA8A4"
STEEL = "#8AA0B8"
BG = "rgba(0,0,0,0)"
# Plotly default chart height is 450; surrogate 2D/3D plots use 1.5× that.
SURROGATE_PLOT_HEIGHT = 675
CONFUSION_PLOT_HEIGHT = int(450 * 2.5 / 2)  # half of previous 2.5× size → 562
CONFUSION_PLOT_WIDTH = CONFUSION_PLOT_HEIGHT
# Classic FEM / MATLAB jet: blue (low) → cyan → green → yellow → red (high)
FEM_JET_COLORSCALE = [
    [0.00, "#00007F"],
    [0.10, "#0000FF"],
    [0.20, "#007FFF"],
    [0.35, "#00FFFF"],
    [0.50, "#7FFF00"],
    [0.65, "#FFFF00"],
    [0.80, "#FF7F00"],
    [0.90, "#FF0000"],
    [1.00, "#7F0000"],
]


def _layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor=BG,
        plot_bgcolor="rgba(16,22,30,0.4)",
        font_color="#E6EDF3",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)")
    return fig


def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Parity plot") -> go.Figure:
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(color=COPPER, size=8, opacity=0.8),
            name="Predictions",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color=STEEL, dash="dash"),
            name="Ideal",
        )
    )
    fig.update_xaxes(title="Observed")
    fig.update_yaxes(title="Predicted")
    return _layout(fig, title)


def residual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(color=TEAL, size=8, opacity=0.8),
            name="Residual",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=STEEL)
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Observed − predicted")
    return _layout(fig, "Residuals")


def confusion_heatmap(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = pd.crosstab(
        pd.Categorical(y_true, categories=labels),
        pd.Categorical(y_pred, categories=labels),
        dropna=False,
    )
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale=["#1A2330", COPPER],
        labels=dict(x="Predicted", y="Observed", color="Count"),
    )
    fig = _layout(fig, "Confusion matrix")
    fig.update_layout(height=CONFUSION_PLOT_HEIGHT, width=CONFUSION_PLOT_WIDTH)
    fig.update_traces(textfont_size=14)
    return fig


def roc_auc_plot(roc_frame: pd.DataFrame, auc_summary: dict[str, float]) -> go.Figure:
    """Plot ROC curve(s) from ``roc_curve_table`` output."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=STEEL, dash="dash", width=1.5),
            name="Chance",
            hoverinfo="skip",
        )
    )
    colors = [COPPER, TEAL, "#7C9CBF", "#E8A87C", "#C38D9E", "#41B3A3", "#E27D60"]
    curves = list(dict.fromkeys(roc_frame["curve"].tolist()))
    for idx, curve in enumerate(curves):
        part = roc_frame.loc[roc_frame["curve"] == curve].sort_values("fpr")
        auc_val = float(part["auc"].iloc[0]) if len(part) else float("nan")
        fig.add_trace(
            go.Scatter(
                x=part["fpr"],
                y=part["tpr"],
                mode="lines",
                line=dict(color=colors[idx % len(colors)], width=2.5),
                name=f"{curve} (AUC={auc_val:.3f})",
            )
        )
    title = "ROC curve"
    if "AUC" in auc_summary:
        title = f"ROC curve · AUC = {auc_summary['AUC']:.4f}"
    elif "AUC micro" in auc_summary:
        title = f"ROC curves · micro AUC = {auc_summary['AUC micro']:.4f}"
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.05])
    return _layout(fig, title)


def importance_bar(frame: pd.DataFrame) -> go.Figure:
    ordered = frame.sort_values("importance", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=ordered["importance"],
            y=ordered["feature"],
            orientation="h",
            marker_color=COPPER,
        )
    )
    fig.update_xaxes(title="Importance")
    fig.update_yaxes(title="")
    return _layout(fig, "Feature importance")


def response_1d(
    pipeline: Any,
    x_ref: pd.DataFrame,
    feature: str,
    grid: np.ndarray,
) -> go.Figure:
    probe = pd.concat([x_ref.iloc[[0]]] * len(grid), ignore_index=True)
    probe[feature] = grid
    pred = pipeline.predict(probe)
    fig = go.Figure(
        go.Scatter(x=grid, y=pred, mode="lines", line=dict(color=COPPER, width=3), name="Surrogate")
    )
    fig.update_xaxes(title=feature)
    fig.update_yaxes(title="Predicted response")
    return _layout(fig, f"1D slice · {feature}")


def response_surface_values(
    pipeline: Any,
    x_ref: pd.DataFrame,
    fx: str,
    fy: str,
    gx: np.ndarray,
    gy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the surrogate on an fx × fy grid; other inputs stay at x_ref."""
    xx, yy = np.meshgrid(gx, gy)
    probe = pd.concat([x_ref.iloc[[0]]] * xx.size, ignore_index=True)
    probe[fx] = xx.ravel()
    probe[fy] = yy.ravel()
    zz = np.asarray(pipeline.predict(probe), dtype=float).reshape(xx.shape)
    return xx, yy, zz


def response_2d(
    pipeline: Any,
    x_ref: pd.DataFrame,
    fx: str,
    fy: str,
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    zz: np.ndarray | None = None,
) -> go.Figure:
    if zz is None:
        _xx, _yy, zz = response_surface_values(pipeline, x_ref, fx, fy, gx, gy)
    fig = go.Figure(
        go.Contour(
            x=gx,
            y=gy,
            z=zz,
            colorscale=FEM_JET_COLORSCALE,
            contours=dict(showlabels=True),
            colorbar=dict(title="ŷ"),
        )
    )
    fig.update_xaxes(title=fx)
    fig.update_yaxes(title=fy)
    fig = _layout(fig, f"2D contour · {fx} × {fy}")
    fig.update_layout(height=SURROGATE_PLOT_HEIGHT)
    return fig


def response_3d(
    pipeline: Any,
    x_ref: pd.DataFrame,
    fx: str,
    fy: str,
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    zz: np.ndarray | None = None,
) -> go.Figure:
    """Interactive 3D surface: X/Y = selected inputs, Z = predicted response.

    X and Y axes use equal visual length even when their data scales differ.
    """
    if zz is None:
        xx, yy, surface = response_surface_values(pipeline, x_ref, fx, fy, gx, gy)
    else:
        xx, yy = np.meshgrid(gx, gy)
        surface = zz

    fig = go.Figure(
        go.Surface(
            x=xx,
            y=yy,
            z=surface,
            colorscale=FEM_JET_COLORSCALE,
            colorbar=dict(title="ŷ"),
            lighting=dict(ambient=0.85, diffuse=0.7, specular=0.15, roughness=0.6),
            contours=dict(
                x=dict(show=False),
                y=dict(show=False),
                z=dict(show=False),
            ),
        )
    )
    fig.update_layout(
        title=f"3D surface · {fx} × {fy}",
        paper_bgcolor=BG,
        font_color="#E6EDF3",
        margin=dict(l=10, r=10, t=50, b=10),
        height=SURROGATE_PLOT_HEIGHT,
        scene=dict(
            xaxis_title=fx,
            yaxis_title=fy,
            zaxis_title="Predicted ŷ",
            aspectmode="manual",
            # Equal on-screen size for X and Y despite different data scales
            aspectratio=dict(x=1, y=1, z=0.75),
            bgcolor="rgba(16,22,30,0.4)",
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.12)",
                showbackground=True,
                backgroundcolor="rgba(16,22,30,0.35)",
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.12)",
                showbackground=True,
                backgroundcolor="rgba(16,22,30,0.35)",
            ),
            zaxis=dict(
                gridcolor="rgba(255,255,255,0.12)",
                showbackground=True,
                backgroundcolor="rgba(16,22,30,0.35)",
            ),
        ),
    )
    return fig


def doe_augment_scatter(
    original: pd.DataFrame,
    next_points: pd.DataFrame,
    x_name: str,
    y_name: str,
) -> go.Figure:
    """Scatter original DOE (blue) vs next locations (red)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=original[x_name],
            y=original[y_name],
            mode="markers",
            name="Original DOE",
            marker=dict(color="#3B82F6", size=9, opacity=0.85, line=dict(width=0)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=next_points[x_name],
            y=next_points[y_name],
            mode="markers",
            name="Next DOE location",
            marker=dict(color="#EF4444", size=11, opacity=0.95, symbol="diamond", line=dict(width=0)),
        )
    )
    fig.update_xaxes(title=x_name)
    fig.update_yaxes(title=y_name)
    return _layout(fig, f"DOE locations · {x_name} vs {y_name}")


def error_histogram(errors: np.ndarray, mean: float, std: float) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=errors,
            nbinsx=min(40, max(10, int(np.sqrt(len(errors)) * 2))),
            marker_color=COPPER,
            opacity=0.85,
            name="error",
        )
    )
    fig.add_vline(
        x=mean,
        line_dash="dash",
        line_color=TEAL,
        annotation_text=f"mean = {mean:.4g}",
        annotation_position="top",
    )
    fig.add_vline(x=0.0, line_dash="solid", line_color=STEEL, annotation_text="0", annotation_position="bottom right")
    if std > 0:
        fig.add_vline(x=mean - std, line_dash="dot", line_color=STEEL, annotation_text="-1σ", annotation_position="bottom")
        fig.add_vline(x=mean + std, line_dash="dot", line_color=STEEL, annotation_text="+1σ", annotation_position="bottom")
    fig.update_xaxes(title="error (observed − predicted)")
    fig.update_yaxes(title="Count")
    return _layout(fig, "Error histogram")


def feature_scatter(
    frame: pd.DataFrame,
    x_name: str,
    y_name: str,
    title: str | None = None,
) -> go.Figure:
    """2D scatter of two numeric columns."""
    clean = frame[[x_name, y_name]].dropna()
    fig = go.Figure(
        go.Scatter(
            x=clean[x_name],
            y=clean[y_name],
            mode="markers",
            marker=dict(color=COPPER, size=8, opacity=0.75, line=dict(width=0)),
            name="samples",
            hovertemplate=f"{x_name}=%{{x}}<br>{y_name}=%{{y}}<extra></extra>",
        )
    )
    fig.update_xaxes(title=x_name)
    fig.update_yaxes(title=y_name)
    return _layout(fig, title or f"{y_name} vs {x_name}")


def correlation_heatmap(
    corr: pd.DataFrame,
    title: str = "Correlation matrix",
    *,
    coeff_label: str = "r",
) -> go.Figure:
    """Correlation heatmap; values ≈0 match the dark theme panel (low contrast)."""
    n = len(corr.columns)
    text = np.round(corr.to_numpy(dtype=float), 2)
    # Midpoint blends into dark UI; ends use app blue / copper accents
    colorscale = [
        [0.00, "#3B82F6"],
        [0.25, "#2A4A6E"],
        [0.45, "#1E2A38"],
        [0.50, "#1A2330"],
        [0.55, "#2A241C"],
        [0.75, "#8A5E38"],
        [1.00, "#D08C4A"],
    ]
    fig = px.imshow(
        corr,
        text_auto=False,
        zmin=-1.0,
        zmax=1.0,
        color_continuous_scale=colorscale,
        aspect="auto",
        labels=dict(color=coeff_label),
    )
    fig.update_traces(
        text=text,
        texttemplate="%{text:.2f}",
        textfont_size=11 if n <= 12 else 9,
        textfont_color="#E6EDF3",
        hovertemplate="%{y} vs %{x}<br>" + coeff_label + "=%{z:.3f}<extra></extra>",
    )
    height = int(min(720, max(380, 42 * n + 140)))
    fig = _layout(fig, title)
    fig.update_layout(
        height=height,
        coloraxis_colorbar=dict(title=coeff_label, thickness=12, len=0.85),
        margin=dict(l=40, r=20, t=50, b=60),
    )
    fig.update_xaxes(side="bottom", tickangle=-45)
    fig.update_yaxes(autorange="reversed")
    return fig
