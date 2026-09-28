"""
alerts.py
---------
Funciones auxiliares para las alertas: sirena, parpadeo rojo y reportes.
La sirena se GENERA con código (no hay que descargar ningún .mp3).
"""

import base64
import io
import wave

import numpy as np


def generar_sirena_wav(duracion=4.0, frecuencia_muestreo=22050):
    """Crea un sonido de sirena (tono que sube y baja) y lo devuelve como bytes WAV."""
    t = np.linspace(0, duracion, int(frecuencia_muestreo * duracion), endpoint=False)
    # La frecuencia oscila entre 600 Hz y 1200 Hz, dos veces por segundo
    frecuencia = 900 + 300 * np.sin(2 * np.pi * 0.5 * t)
    fase = 2 * np.pi * np.cumsum(frecuencia) / frecuencia_muestreo
    onda = (0.6 * np.sin(fase) * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(frecuencia_muestreo)
        w.writeframes(onda.tobytes())
    return buffer.getvalue()


def html_sirena_oculta():
    """Devuelve un bloque HTML con <audio> oculto que suena solo y en bucle."""
    b64 = base64.b64encode(generar_sirena_wav()).decode()
    return (
        f'<audio autoplay loop style="display:none">'
        f'<source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'
    )


def css_parpadeo_rojo():
    """Devuelve CSS que hace parpadear suavemente el fondo de la app en rojo."""
    return """
    <style>
    @keyframes blinker {
        0%   { background-color: #0e1117; }
        50%  { background-color: #7a1f1f; }
        100% { background-color: #0e1117; }
    }
    .stApp { animation: blinker 1.2s ease-in-out infinite; }
    </style>
    """


def texto_protocolo_emergencia():
    """Recomendaciones urgentes para el nivel crítico."""
    return """
### 🚨 PROTOCOLO DE EMERGENCIA
1. **Evacúe** el área de inmediato y aleje a las personas de la estructura.
2. **No intente reparar** la grieta por cuenta propia.
3. **No use** la estructura hasta que sea revisada.
4. **Llame** a un ingeniero estructural y a los bomberos / defensa civil.
5. **Acordone** la zona y tome fotos desde una distancia segura.
"""


def formatear_reporte(nombre, metricas, nivel):
    """Texto de reporte que el usuario puede descargar."""
    return (
        "REPORTE - SISTEMA DE DETECCION DE GRIETAS\n"
        "=========================================\n"
        f"Fuente analizada : {nombre}\n"
        f"crack_score      : {metricas['crack_score']:.2f}\n"
        f"Nivel            : {nivel.upper()}\n"
        f"Area afectada    : {metricas['area_ratio'] * 100:.2f} %\n"
        f"Longitud aprox.  : {metricas['longitud_px']} px\n"
        f"Grosor promedio  : {metricas['grosor_px']} px\n"
    )
