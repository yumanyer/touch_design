import cv2
import numpy as np
import sounddevice as sd
import random

# ================= CONFIG =================

CAMERA_INDEX = 0

# Conjunto de caracteres optimizado para CONTRASTE y SILUETA
# Se eliminan caracteres medios que ensucian la imagen
# Estructura: [Vacío] -> [Puntos] -> [Estructura] -> [Bloque]
ASCII_CHARS = " .:-=+*#%@" 

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Rango de resolución (tamaño de celda)
MIN_CELL = 4    # Máximo detalle (con ruido)
MAX_CELL = 45   # Máxima abstracción (silencio)

# Sensibilidad y suavizado
AUDIO_GAIN = 40.0
AUDIO_SMOOTH = 0.85 

# ==========================================

audio_level = 0.0

def audio_callback(indata, frames, time_info, status):
    global audio_level
    rms = np.sqrt(np.mean(indata**2))
    volume = rms * AUDIO_GAIN
    audio_level = (AUDIO_SMOOTH * audio_level) + ((1 - AUDIO_SMOOTH) * volume)

def get_params(level):
    level = np.clip(level, 0.0, 1.0)
    
    # Celda: Inversa al audio
    cell = int(MAX_CELL - (level * (MAX_CELL - MIN_CELL)))
    cell = max(cell, MIN_CELL)
    
    # Jitter reducido para no perder la forma humana
    jitter = int(level * (cell / 4.0))
    
    # Grosor dinámico
    thickness = 1 if level < 0.5 else 2
    
    # Escala de fuente
    font_scale = (cell / 22.0)
    
    return cell, jitter, thickness, font_scale

def main():
    global audio_level

    try:
        stream = sd.InputStream(
            channels=1,
            callback=audio_callback,
            samplerate=44100
        )
        stream.start()
    except Exception as e:
        print(f"Error al iniciar audio: {e}")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    cv2.namedWindow("ASCII_VISION", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("ASCII_VISION", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 1. Convertir a Gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. MEJORA DE VISIBILIDAD: Ecualización de Histograma Adaptativa (CLAHE)
        # Esto hace que te veas bien incluso con poca luz o luz de fondo
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 3. Desenfoque ligero para reducir ruido visual y suavizar caracteres
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Obtener parámetros dinámicos
        cell, jitter_max, thickness, font_scale = get_params(audio_level)

        # Canvas negro absoluto
        canvas = np.zeros((h, w), dtype=np.uint8)

        # Iterar por la cuadrícula
        for y in range(0, h, cell):
            for x in range(0, w, cell):
                block = gray[y:y+cell, x:x+cell]
                if block.size == 0: continue

                avg = np.mean(block)
                
                # Umbral de visibilidad más bajo para no perder la silueta
                if avg > 35:
                    # Mapear brillo a carácter (usando el nuevo set optimizado)
                    char_idx = int((avg / 255.0) * (len(ASCII_CHARS) - 1))
                    char = ASCII_CHARS[char_idx]
                    
                    # Jitter sutil
                    jx = x + random.randint(-jitter_max, jitter_max) if jitter_max > 0 else x
                    jy = y + cell + random.randint(-jitter_max, jitter_max) if jitter_max > 0 else y + cell
                    
                    cv2.putText(
                        canvas,
                        char,
                        (jx, jy),
                        FONT,
                        font_scale,
                        255,
                        thickness,
                        cv2.LINE_AA
                    )

        cv2.imshow("ASCII_VISION", canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
