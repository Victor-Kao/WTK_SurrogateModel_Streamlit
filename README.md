# WTK Surrogate Model

Streamlit workbench for design-of-experiments sampling, surrogate training, validation, and visualization.

## Pages

1. **Introduction** — workflow and how the pages connect.
2. **DOE Sampling** — Latin Hypercube, Sobol, Halton, random, and full factorial designs.
3. **Machine Learning**
   - **Regression:** Import Data → Training → Advance Training
   - **Binary / Multiclass Classification:** Import Data → Training
4. **Validation** — score the stored test split or an independent labeled CSV.
5. **Model Information & Visualization** — inspect stored or uploaded models, metrics, diagnostics, feature importance, 1D/2D slices.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

After training, the model stays in session memory only. Click **Download fitted model** on Training or Model Information to save a **pickle** (`.pkl`) or **joblib** (`.joblib`) file (also written under `models/`).
