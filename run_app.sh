#!/usr/bin/env bash
# Launch the Streamlit app (macOS / Linux). Prefer the project venv when present.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x ".venv/bin/streamlit" ]]; then
  exec .venv/bin/streamlit run app.py "$@"
elif [[ -x "venv/bin/streamlit" ]]; then
  exec venv/bin/streamlit run app.py "$@"
else
  exec streamlit run app.py "$@"
fi
