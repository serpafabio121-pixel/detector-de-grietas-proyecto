"""
home.py
-------
Interfaz principal (Streamlit). Se ejecuta con:  streamlit run app/home.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Permite importar la carpeta "src" aunque se ejecute desde otro lugar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import alerts, detector  # noqa: E402

st.set_page_config(page_title="Detector de Grietas", page_icon="🏗️", layout="wide")


# ----------------------------------------------------------- funciones de UI
def pil_a_bgr(imagen_pil):
    """Convierte una imagen PIL (RGB) al formato de OpenCV (BGR)."""
    return cv2.cvtColor(np.array(imagen_pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_a_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mostrar_alerta(metricas, nombre="imagen"):
    """Muestra el mensaje según el nivel y activa sirena + parpadeo si es severo."""
    score = metricas["crack_score"]
    nivel = detector.clasificar(score)

    st.metric("crack_score", f"{score:.2f}")
    st.progress(min(score, 1.0))

    if nivel == "leve":
        st.success("✅ ESTRUCTURA ESTABLE: No se detectan anomalías críticas.")
    elif nivel == "medio":
        st.warning("⚠️ ADVERTENCIA: Anomalía visual de nivel MEDIO. Se sugiere inspección profesional.")
    else:
        st.error("🚨 ¡ALERTA CRÍTICA DETECTADA! El nivel de daño es SEVERO.")
        st.markdown(alerts.texto_protocolo_emergencia())
        st.markdown(alerts.css_parpadeo_rojo(), unsafe_allow_html=True)
        st.markdown(alerts.html_sirena_oculta(), unsafe_allow_html=True)
        with st.expander("🔊 ¿No suena la sirena? Presiona aquí"):
            st.caption("Los navegadores bloquean el sonido automático hasta que haces clic en la página.")
            st.audio(alerts.generar_sirena_wav(), format="audio/wav", loop=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Área afectada", f"{metricas['area_ratio'] * 100:.2f} %")
    c2.metric("Longitud aprox.", f"{metricas['longitud_px']} px")
    c3.metric("Grosor promedio", f"{metricas['grosor_px']} px")

    st.download_button("📄 Descargar reporte", alerts.formatear_reporte(nombre, metricas, nivel),
                       file_name="reporte_grieta.txt")


def analizar_y_mostrar(bgr, nombre):
    """Analiza una imagen y muestra original + resultado + alertas."""
    # Primero revisamos que no haya personas en la foto
    personas = detector.detectar_personas(bgr)
    if personas:
        st.warning("👤 Se detectó una persona en la foto. Esto NO es una grieta. "
                   "Toma la foto solo de la pared o estructura.")
        st.image(bgr_a_rgb(detector.dibujar_personas(bgr, personas)),
                 caption="Persona detectada (cuadro naranja)", use_container_width=True)
        return

    resultado, mascara, metricas = detector.analizar_imagen(bgr)
    col1, col2 = st.columns(2)
    col1.subheader("Imagen original")
    col1.image(bgr_a_rgb(detector.redimensionar(bgr)), use_container_width=True)
    col2.subheader("Grietas detectadas")
    col2.image(bgr_a_rgb(resultado), use_container_width=True)
    mostrar_alerta(metricas, nombre)


# ------------------------------------------------------------ barra lateral
st.sidebar.title("🏗️ Detector de Grietas")
fuente = st.sidebar.radio(
    "Fuente de entrada",
    ["📷 Cámara en vivo", "🖼️ Subir imagen", "🎞️ Subir video", "🧪 Muestras de prueba"],
)
st.sidebar.info("Grietas = líneas oscuras y delgadas sobre la superficie. "
                "El puntaje sube con el área, la longitud y el grosor.")

st.title("Sistema Inteligente de Detección y Análisis de Grietas Estructurales")

# ------------------------------------------------------------------ fuentes
if fuente == "📷 Cámara en vivo":
    foto = st.camera_input("Toma una foto de la estructura")
    if foto:
        analizar_y_mostrar(pil_a_bgr(Image.open(foto)), "camara")

elif fuente == "🖼️ Subir imagen":
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "webp"])
    if archivo:
        analizar_y_mostrar(pil_a_bgr(Image.open(archivo)), archivo.name)

elif fuente == "🎞️ Subir video":
    video = st.file_uploader("Sube un clip de video", type=["mp4", "avi", "mov", "mkv"])
    if video:
        # OpenCV necesita un archivo en disco, así que lo guardamos temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video.read())
            ruta = tmp.name
        barra = st.progress(0.0, text="Analizando video...")
        scores, peor, con_personas = detector.analizar_video(ruta, progreso=lambda p: barra.progress(p))
        barra.empty()
        os.remove(ruta)

        if con_personas:
            st.info(f"👤 Se omitieron {con_personas} fotogramas donde aparece una persona.")
        if not scores:
            st.error("No se pudo analizar el video: no se pudo leer o solo aparecen personas.")
        else:
            st.subheader(f"Se analizaron {len(scores)} fotogramas")
            st.line_chart(scores)
            st.caption("Se muestra el fotograma con mayor daño detectado.")
            col1, col2 = st.columns(2)
            col1.image(bgr_a_rgb(peor["original"]), caption=f"Fotograma {peor['frame']}",
                       use_container_width=True)
            col2.image(bgr_a_rgb(peor["resultado"]), caption="Grietas detectadas",
                       use_container_width=True)
            mostrar_alerta(peor["metricas"], video.name)

else:  # Muestras de prueba
    muestras = detector.generar_muestras()
    nombre = st.selectbox("Elige una muestra", list(muestras.keys()))
    analizar_y_mostrar(muestras[nombre], nombre)
