"""WTK Surrogate Model workbench."""

from __future__ import annotations

import streamlit as st

# Pin to 1.43.2: this app uses use_container_width (not the newer width= API).
# 1.43.0–1.43.1 had button/popover width bugs when help= was combined with
# use_container_width=True; 1.43.2 includes that fix.
_REQUIRED_STREAMLIT = "1.43.2"
if st.__version__ != _REQUIRED_STREAMLIT:
    raise RuntimeError(
        f"This app requires streamlit=={_REQUIRED_STREAMLIT} "
        f"(found {st.__version__}). Install with: pip install -r requirements.txt"
    )

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
