"""WTK Surrogate Model workbench."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="WTK Surrogate Model",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.state import init_state
from src.ui import inject_css, render_sidebar
from views import data_analysis, data_import, doe_sampling, introduction, performance, training, validation

inject_css()
init_state()

route = render_sidebar()
if route == "introduction":
    introduction.render()
elif route == "doe":
    doe_sampling.render()
elif route == "reg_import":
    data_import.render("regression")
elif route == "reg_train":
    training.render("regression")
elif route == "reg_advance":
    training.render("regression", advanced=True)
elif route == "bin_import":
    data_import.render("binary_classification")
elif route == "bin_train":
    training.render("binary_classification")
elif route == "multi_import":
    data_import.render("multiclass_classification")
elif route == "multi_train":
    training.render("multiclass_classification")
elif route == "clf_import":
    # Legacy route → binary
    data_import.render("binary_classification")
elif route == "clf_train":
    training.render("binary_classification")
elif route == "validation":
    validation.render()
elif route == "data_analysis":
    data_analysis.render()
elif route == "performance":
    performance.render()
else:
    performance.render()
