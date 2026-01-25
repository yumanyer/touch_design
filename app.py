import cv2
import numpy as np
import sounddevice as sd
import random

# ================= CONFIG =================

CAMERA_INDEX = 0

# Conjunto de caracteres solicitado por el usuario
ASCII_CHARS = " .,:;irsXA253hMHGS#9B&@"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Rango de resolución (tamaño de celda)
MIN_CELL = 5    # Mucho ruido -> detalle fino
MAX_CELL = 50   # Silencio -> abstracción total

# Sensibilidad y suavizado
AUDIO_GAIN = 35.0
AUDIO_SMOOTH = 0.80  # Un poco más rápido para captar beats

# ==========================================

audio_level = 0.0

def audio_callback(indata, frames, time_info, status):
    global audio_level
    rms = np.sqrt(np.mean(indata**2))
    volume = rms * AUDIO_GAIN
    audio_level = (AUDIO_SMOOTH * audio_level) + ((1 - AUDIO_SMOOTH) * volume)

def get_params(level):
    """
    Calcula parámetros visuales basados en el nivel de audio.
    """
    level = np.clip(level, 0.0, 1.0)
    
    # Celda: Inversa al audio
    cell = int(MAX_CELL - (level * (MAX_CELL - MIN_CELL)))
    cell = max(cell, MIN_CELL)
    
    # Jitter: Desplazamiento aleatorio aumenta con el audio
    jitter = int(level * (cell / 2.0))
    
    # Grosor: Aumenta con el audio
    thickness = 1 if level < 0.4 else 2
    if level > 0.8: thickness = 3
    
    # Escala de fuente: Proporcional a la celda pero con un boost en picos
    font_scale = (cell / 20.0) * (0.8 + level * 0.5)
    
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

    cv2.namedWindow("SYSTEM_REACTION", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("SYSTEM_REACTION", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Sistema reactivo mejorado. Presiona 'ESC' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Procesamiento de imagen agresivo
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Umbral dinámico basado en audio para "limpiar" el negro
        threshold_val = 40 + int(audio_level * 40)
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)

        # Obtener parámetros dinámicos
        cell, jitter_max, thickness, font_scale = get_params(audio_level)

        # Canvas negro absoluto
        canvas = np.zeros((h, w), dtype=np.uint8)

        # Iterar por la cuadrícula con un pequeño offset aleatorio (Jitter)
        for y in range(0, h, cell):
            for x in range(0, w, cell):
                # Extraer bloque
                block = binary[y:y+cell, x:x+cell]
                if block.size == 0: continue

                avg = np.mean(block)
                
                # Solo dibujar si hay suficiente "luz"
                if avg > 50:
                    # Mapear brillo a carácter
                    char_idx = int((avg / 255.0) * (len(ASCII_CHARS) - 1))
                    char = ASCII_CHARS[char_idx]
                    
                    # Aplicar Jitter a la posición
                    jx = x + random.randint(-jitter_max, jitter_max) if jitter_max > 0 else x
                    jy = y + cell + random.randint(-jitter_max, jitter_max) if jitter_max > 0 else y + cell
                    
                    # Dibujar carácter
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

        # Mostrar resultado
        cv2.imshow("SYSTEM_REACTION", canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
