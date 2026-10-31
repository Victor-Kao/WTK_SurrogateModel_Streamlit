@echo off
REM Launch the Streamlit app (Windows). Prefer the project venv when present.
cd /d "%~dp0"

if exist ".venv\Scripts\streamlit.exe" (
  ".venv\Scripts\streamlit.exe" run app.py %*
  exit /b %ERRORLEVEL%
)

if exist "venv\Scripts\streamlit.exe" (
  "venv\Scripts\streamlit.exe" run app.py %*
  exit /b %ERRORLEVEL%
)

streamlit run app.py %*
