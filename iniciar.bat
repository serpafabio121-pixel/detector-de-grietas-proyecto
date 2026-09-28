@echo off
echo Instalando librerias (solo la primera vez tarda)...
python -m pip install -r requirements.txt
echo.
echo Abriendo la aplicacion...
python -m streamlit run app/home.py
pause
