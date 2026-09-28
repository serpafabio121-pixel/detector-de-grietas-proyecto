#!/bin/bash
echo "Instalando librerias (solo la primera vez tarda)..."
python3 -m pip install -r requirements.txt
echo "Abriendo la aplicacion..."
python3 -m streamlit run app/home.py
