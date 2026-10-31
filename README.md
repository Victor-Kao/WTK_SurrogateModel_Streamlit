# WTK Surrogate Model

Streamlit workbench for design-of-experiments sampling, surrogate training, validation, and visualization.

**Requires Python 3.9+ and Streamlit 1.43.2** (pinned in `requirements.txt`). That release keeps the `use_container_width` API this app uses and includes the button/popover width fix from the 1.43.x line.

## Pages

1. **Introduction** — workflow and how the pages connect.
2. **DOE Sampling** — Latin Hypercube, Sobol, Halton, random, and full factorial designs.
3. **Machine Learning**
   - **Regression:** Import Data → Training → Hyperparameter Optimization
   - **Binary / Multiclass Classification:** Import Data → Training
4. **Validation** — score the stored test split or an independent labeled CSV.
5. **Model Information & Visualization** — inspect stored or uploaded models, metrics, diagnostics, feature importance, 1D/2D slices.

## How to run

Install dependencies once, then start the app. Prefer a virtual environment so Streamlit stays at **1.43.2**.

### With a virtual environment (recommended)

#### macOS / Linux

```bash
cd WTK_SurrogateModel_Streamlit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Or after the venv exists and packages are installed:

```bash
chmod +x run_app.sh   # once
./run_app.sh
```

#### Windows (PowerShell)

```powershell
cd WTK_SurrogateModel_Streamlit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If script activation is blocked, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

#### Windows (Command Prompt)

```cmd
cd WTK_SurrogateModel_Streamlit
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

Or double-click / run:

```cmd
run_app.bat
```

### Without a virtual environment

Install into your current Python (may conflict with other projects):

#### macOS / Linux

```bash
cd WTK_SurrogateModel_Streamlit
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

#### Windows (PowerShell or Command Prompt)

```bat
cd WTK_SurrogateModel_Streamlit
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Deactivate the venv (when finished)

- **macOS / Linux / Windows (activated shell):** `deactivate`

## Notes

- After training, the model stays in session memory only. Click **Download fitted model** on Training or Model Information to save a **pickle** (`.pkl`) or **joblib** (`.joblib`) file (also written under `models/`).
- If you see a runtime error about the Streamlit version, reinstall from `requirements.txt` so you get exactly `streamlit==1.43.2`.
