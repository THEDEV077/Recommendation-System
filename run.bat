@echo off
echo 🚀 Launching Premium MovieLens Recommender...
echo 🐍 Using Python 3.11 Environment (LightFM Compatible)
echo.

if not exist venv_lfm (
    echo ⚠️ Virtual environment not found!
)

call venv_lfm\Scripts\activate
streamlit run app.py
pause
