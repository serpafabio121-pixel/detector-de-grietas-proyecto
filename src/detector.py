"""
detector.py
-----------
Funciones de visión por computador (OpenCV + NumPy).
No dependen de Streamlit: reciben una imagen y devuelven resultados.

Idea del algoritmo (paso a paso):
  1. Escala de grises           -> quitamos el color, solo importa la luz.
  2. Suavizado gaussiano        -> reducimos el ruido de la textura del concreto.
  3. Black-hat morfológico      -> resalta cosas OSCURAS y DELGADAS (las grietas).
  4. Umbral de OTSU             -> separa "grieta" de "no grieta" automáticamente.
  5. Contornos + filtro         -> nos quedamos solo con formas alargadas.
  6. Métricas y crack_score     -> área, longitud, grosor -> puntaje 0.0 a 1.0.
"""

import cv2
import numpy as np

ANCHO_MAX = 640  # se reduce la imagen para que el análisis sea rápido


# ---------------------------------------------------------------- utilidades
def redimensionar(bgr, ancho_max=ANCHO_MAX):
    """Reduce la imagen si es muy grande (mantiene la proporción)."""
    alto, ancho = bgr.shape[:2]
    if ancho <= ancho_max:
        return bgr
    factor = ancho_max / ancho
    return cv2.resize(bgr, (ancho_max, int(alto * factor)), interpolation=cv2.INTER_AREA)


def esqueleto(mascara):
    """
    Adelgaza la máscara hasta dejar líneas de 1 píxel de ancho.
    Sirve para medir la LONGITUD de la grieta.
    """
    img = mascara.copy()
    esq = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        erosion = cv2.erode(img, kernel)
        apertura = cv2.dilate(erosion, kernel)
        resto = cv2.subtract(img, apertura)
        esq = cv2.bitwise_or(esq, resto)
        img = erosion
        if cv2.countNonZero(img) == 0:
            break
    return esq


# ------------------------------------------------------------ preprocesado
def preprocesar(bgr):
    """Pasos 1 a 4: devuelve la máscara binaria de 'posibles grietas'."""
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gris, (5, 5), 0)

    # Black-hat: resalta zonas oscuras y delgadas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    blackhat = cv2.morphologyEx(suave, cv2.MORPH_BLACKHAT, kernel)

    # Si casi no hay contraste, no hay grieta (evita que OTSU invente una)
    if blackhat.max() < 25:
        return gris, np.zeros_like(gris)

    _, mascara = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Limpieza: quita puntitos sueltos y une pedacitos cercanos
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k3)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k3)
    return gris, mascara


# ------------------------------------------------------------ segmentación
def filtrar_contornos(mascara):
    """
    Paso 5: conserva solo contornos ALARGADOS (las grietas son largas y finas,
    las manchas son redondas). Devuelve la máscara limpia y la lista de contornos.
    """
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    limpia = np.zeros_like(mascara)
    validos = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < 30:
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        lado_largo, lado_corto = max(w, h), max(min(w, h), 1)
        alargada = lado_largo / lado_corto
        # Solidez: una grieta ramificada ocupa poco de su envolvente convexa,
        # una mancha redonda la ocupa casi toda.
        hull = cv2.contourArea(cv2.convexHull(c))
        solidez = area / hull if hull > 0 else 1.0
        if lado_largo >= 40 and (alargada >= 3 or solidez < 0.35):
            validos.append(c)
            cv2.drawContours(limpia, [c], -1, 255, thickness=cv2.FILLED)
    return limpia, validos


# ----------------------------------------------------------------- métricas
def calcular_metricas(mascara):
    """Paso 6: proporción de área, longitud, grosor y crack_score (0.0 a 1.0)."""
    alto, ancho = mascara.shape[:2]
    total = alto * ancho
    area_px = int(cv2.countNonZero(mascara))
    if area_px == 0:
        return {"area_ratio": 0.0, "longitud_px": 0, "grosor_px": 0.0, "crack_score": 0.0}

    longitud_px = int(cv2.countNonZero(esqueleto(mascara)))
    grosor_px = area_px / max(longitud_px, 1)
    diagonal = float(np.hypot(alto, ancho))
    area_ratio = area_px / total

    # Cada métrica se lleva a una escala 0-1 y luego se combinan con pesos
    s_area = min(area_ratio / 0.04, 1.0)
    s_long = min(longitud_px / (1.2 * diagonal), 1.0)
    s_grosor = min(grosor_px / 10.0, 1.0)
    score = 0.35 * s_area + 0.35 * s_long + 0.30 * s_grosor

    return {
        "area_ratio": round(area_ratio, 4),
        "longitud_px": longitud_px,
        "grosor_px": round(grosor_px, 2),
        "crack_score": round(float(min(max(score, 0.0), 1.0)), 2),
    }


# --------------------------------------------------------- función principal
def analizar_imagen(bgr):
    """
    Recibe una imagen BGR (formato OpenCV) y devuelve:
      - resultado: imagen con las grietas marcadas en rojo (BGR)
      - mascara:   máscara blanca/negra de las grietas
      - metricas:  diccionario con área, longitud, grosor y crack_score
    """
    bgr = redimensionar(bgr)
    _, mascara = preprocesar(bgr)
    mascara, contornos = filtrar_contornos(mascara)
    metricas = calcular_metricas(mascara)

    resultado = bgr.copy()
    capa_roja = resultado.copy()
    capa_roja[mascara > 0] = (0, 0, 255)
    resultado = cv2.addWeighted(capa_roja, 0.6, resultado, 0.4, 0)
    cv2.drawContours(resultado, contornos, -1, (0, 0, 255), 2)
    return resultado, mascara, metricas


def clasificar(score):
    """Devuelve 'leve', 'medio' o 'severo' según el crack_score."""
    if score < 0.48:
        return "leve"
    if score < 0.70:
        return "medio"
    return "severo"



# ------------------------------------------------------ detección de personas
_CARA_FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
_CARA_PERFIL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
_HOG = cv2.HOGDescriptor()
_HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detectar_personas(bgr):
    """
    Busca personas en la imagen (caras de frente, de perfil y cuerpos).
    Devuelve una lista de cuadros (x, y, ancho, alto) sobre la imagen
    redimensionada. Lista vacía = no hay personas.
    """
    img = redimensionar(bgr)
    gris = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cuadros = []

    for cascada in (_CARA_FRONTAL, _CARA_PERFIL):
        caras = cascada.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=6,
                                         minSize=(40, 40))
        cuadros += [tuple(int(v) for v in c) for c in caras]

    # Cuerpo completo (HOG). Solo se aceptan detecciones con buena confianza.
    rects, pesos = _HOG.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
    for r, w in zip(rects, np.array(pesos).flatten()):
        if w >= 0.9:
            cuadros.append(tuple(int(v) for v in r))
    return cuadros


def dibujar_personas(bgr, cuadros):
    """Devuelve la imagen redimensionada con cuadros azules sobre las personas."""
    img = redimensionar(bgr).copy()
    for (x, y, w, h) in cuadros:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 140, 0), 3)
    return img

# ------------------------------------------------------------------- video
def analizar_video(ruta, cada_n_frames=10, max_frames=60, progreso=None):
    """
    Procesa un video fotograma a fotograma (uno de cada 'cada_n_frames').
    Devuelve la lista de scores y los datos del peor fotograma.
    """
    cap = cv2.VideoCapture(ruta)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    scores, peor = [], None
    idx = analizados = con_personas = 0

    while analizados < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % cada_n_frames == 0:
            if detectar_personas(frame):      # si hay una persona, no es una grieta
                con_personas += 1
                idx += 1
                analizados += 1
                continue
            resultado, mascara, m = analizar_imagen(frame)
            scores.append(m["crack_score"])
            if peor is None or m["crack_score"] > peor["metricas"]["crack_score"]:
                peor = {"original": redimensionar(frame), "resultado": resultado,
                        "metricas": m, "frame": idx}
            analizados += 1
            if progreso:
                progreso(min(idx / total, 1.0))
        idx += 1
    cap.release()
    return scores, peor, con_personas


# ---------------------------------------------------- imágenes de muestra
def _textura_concreto(alto, ancho, rng):
    """Crea una textura que parece concreto/pared."""
    base = rng.normal(165, 14, (alto, ancho)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    manchas = cv2.GaussianBlur(rng.normal(0, 40, (alto, ancho)).astype(np.float32), (0, 0), 40)
    img = np.clip(base + manchas, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _dibujar_grieta(img, rng, inicio, pasos, grosor, ramas=0):
    x, y = inicio
    angulo = rng.uniform(0.3, 1.2)
    puntos = [(x, y)]
    for _ in range(pasos):
        angulo += rng.normal(0, 0.25)
        x += 14 * np.cos(angulo)
        y += 14 * np.sin(angulo)
        puntos.append((int(x), int(y)))
    pts = np.array(puntos, np.int32)
    for i in range(len(pts) - 1):
        g = max(1, int(grosor * (1 - 0.4 * i / len(pts))))
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), (25, 25, 25), g, cv2.LINE_AA)
    for _ in range(ramas):
        k = int(rng.integers(3, max(4, len(pts) - 3)))
        _dibujar_grieta(img, rng, tuple(pts[k]), int(pasos * 0.5), max(1, grosor * 0.6), 0)


def generar_muestras():
    """Devuelve un diccionario {nombre: imagen BGR} para el modo demostración."""
    alto, ancho = 480, 640
    muestras = {}

    rng = np.random.default_rng(1)
    muestras["Pared sana (sin grietas)"] = _textura_concreto(alto, ancho, rng)

    rng = np.random.default_rng(2)
    img = _textura_concreto(alto, ancho, rng)
    _dibujar_grieta(img, rng, (60, 40), 14, 2)
    muestras["Fisura fina (leve)"] = img

    rng = np.random.default_rng(3)
    img = _textura_concreto(alto, ancho, rng)
    _dibujar_grieta(img, rng, (80, 20), 28, 7, ramas=2)
    muestras["Grieta media"] = img

    rng = np.random.default_rng(4)
    img = _textura_concreto(alto, ancho, rng)
    _dibujar_grieta(img, rng, (40, 10), 32, 12, ramas=4)
    _dibujar_grieta(img, rng, (300, 0), 30, 10, ramas=3)
    muestras["Grieta severa (crítica)"] = img

    return muestras
