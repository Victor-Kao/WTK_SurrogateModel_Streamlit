"""Page 1 — Introduction."""

from __future__ import annotations

import streamlit as st

from src.ml import CLASSIFIERS, REGRESSORS, estimator_display_name
from src.ui import page_header

# Library floors from requirements.txt; comments also note a known-good env.
_SNIPPET = '''\
# Required libraries (tested versions in parentheses):
#   numpy         >= 1.24.0   (e.g. 2.4.4)
#   pandas        >= 2.0.0    (e.g. 2.2.3)
#   scikit-learn  >= 1.3.0    (e.g. 1.8.0)
#   joblib        >= 1.3.0    (e.g. 1.5.3)  # only if you use .joblib
#   pickle        (Python stdlib)

import pickle
import pandas as pd

with open("your_model.pkl", "rb") as f:
    model = pickle.load(f)

# Same feature names / order as when you trained
X = pd.DataFrame([{"x1": 0.5, "x2": 1.2}])

print(model.predict(X))

# Optional (binary classification):
# print(model.predict_proba(X)[:, 1])
'''


def _model_list_html(names: list[str]) -> str:
    items = "".join(f"<li>{estimator_display_name(name)}</li>" for name in names)
    return f"<ul>{items}</ul>"


def render() -> None:
    page_header(
        "Page 01",
        "Introduction",
        "Build a surrogate model from designed experiments, then train, validate, and inspect it.",
        active_step=1,
    )

    st.markdown(
        """
        <div class="hero-grid">
          <div class="hero-card"><div class="hero-num">01</div><h4>Introduction</h4><p>Scope, workflow, and how data moves between pages.</p></div>
          <div class="hero-card"><div class="hero-num">02</div><h4>DOE Sampling</h4><p>Create a space-filling or factorial design over your input bounds.</p></div>
          <div class="hero-card"><div class="hero-num">03</div><h4>Machine Learning</h4><p>Regression and classification tracks, each with import and training steps.</p></div>
          <div class="hero-card"><div class="hero-num">04</div><h4>Validation</h4><p>Score a held-out or independent dataset against the fitted surrogate.</p></div>
          <div class="hero-card"><div class="hero-num">05</div><h4>Model Information</h4><p>Inspect parameters, diagnostics, and response-surface visualization.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns((1.2, 1), gap="large")
    with left:
        st.subheader("Suggested Workflow")
        st.markdown(
            """
            1. Define variables and generate a design on **DOE Sampling**.
            2. Run your simulator or test plan, then bring the table into **Machine Learning → Import Data**.
            3. Fit a baseline on **Training**. Optionally refine it on **Advance Training** (regression).
            4. Confirm generalization on **Validation**.
            5. Inspect slices, importance, and diagnostics on **Model Information & Visualization**.
            """
        )

        st.subheader("Current Supported ML Models")
        st.markdown(
            f"""
            <div class="model-family-grid">
              <div class="model-family-card">
                <h4>Regression</h4>
                {_model_list_html(list(REGRESSORS))}
              </div>
              <div class="model-family-card">
                <h4>Binary Classification</h4>
                {_model_list_html(list(CLASSIFIERS))}
              </div>
              <div class="model-family-card">
                <h4>Multiclass Classification</h4>
                {_model_list_html(list(CLASSIFIERS))}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Use the downloaded pickle in your own code")
        st.caption("Minimal example for scoring new rows outside this app.")
        st.code(_SNIPPET, language="python")
