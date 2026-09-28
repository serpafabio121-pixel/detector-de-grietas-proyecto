# Detector de Grietas

Aplicación web hecha con **Streamlit** y **OpenCV** que detecta grietas en imágenes y video, estima su severidad (leve, medio, severo) y detecta personas en la escena.

## Estructura

```
app/home.py        # interfaz Streamlit
src/detector.py    # procesamiento de imagen y detección
src/alerts.py      # alertas
requirements.txt
```

## Uso local

```bash
pip install -r requirements.txt
python -m streamlit run app/home.py
```

En Windows también puedes hacer doble clic en `iniciar.bat`.

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub (con `app/`, `src/` y `requirements.txt` en la raíz).
2. En share.streamlit.io elige el repo y como archivo principal `app/home.py`.
3. En **Advanced settings** selecciona **Python 3.12**.
