import cv2
import numpy as np
import sounddevice as sd
import time

# ================= CONFIG =================

CAMERA_INDEX = 0

# Conjunto de caracteres solicitado por el usuario
ASCII_CHARS = " .,:;irsXA253hMHGS#9B&@"
FONT = cv2.FONT_HERSHEY_SIMPLEX  # Fuente estándar de OpenCV

# Rango de resolución (tamaño de celda)
# Poco ruido -> caracteres grandes (MAX_CELL), imagen abstracta
# Mucho ruido -> caracteres chicos (MIN_CELL), más densidad/detalle
MIN_CELL = 6    
MAX_CELL = 40   

# Sensibilidad y suavizado
AUDIO_GAIN = 25.0
AUDIO_SMOOTH = 0.85  # Suavizado para evitar saltos bruscos de resolución

# ==========================================

audio_level = 0.0

def audio_callback(indata, frames, time_info, status):
    global audio_level
    # Calcular RMS (Root Mean Square) para medir la energía del audio
    rms = np.sqrt(np.mean(indata**2))
    volume = rms * AUDIO_GAIN
    # Aplicar suavizado exponencial
    audio_level = (AUDIO_SMOOTH * audio_level) + ((1 - AUDIO_SMOOTH) * volume)

def get_cell_size(level):
    """
    Mapea el nivel de audio al tamaño de la celda.
    Inverso: más nivel -> menor celda (más detalle).
    """
    level = np.clip(level, 0.0, 1.0)
    # Interpolar linealmente entre MAX y MIN
    cell = MAX_CELL - (level * (MAX_CELL - MIN_CELL))
    return int(cell)

def main():
    global audio_level

    # ---- Configuración de Audio ----
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

    # ---- Configuración de Video ----
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    # Ventana en pantalla completa para inmersión
    cv2.namedWindow("ASCII_SYSTEM", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("ASCII_SYSTEM", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    print("Sistema iniciado. Presiona 'ESC' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espejo para interacción natural
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Procesamiento de imagen: Blanco y Negro puro
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Normalizar y aumentar contraste para que el blanco sea "caracter" y negro "vacío"
        gray = cv2.equalizeHist(gray)
        
        # Determinar resolución basada en audio
        cell = get_cell_size(audio_level)
        cell = max(cell, MIN_CELL) # Asegurar mínimo

        # Crear canvas negro absoluto
        canvas = np.zeros((h, w), dtype=np.uint8)

        # Escalar fuente proporcional al tamaño de celda
        font_scale = cell / 22.0
        thickness = 1 if cell < 15 else 2

        # Iterar por la cuadrícula
        for y in range(0, h, cell):
            for x in range(0, w, cell):
                # Extraer bloque de la imagen original
                block = gray[y:y+cell, x:x+cell]
                if block.size == 0:
                    continue

                # El brillo promedio determina el carácter
                avg_brightness = np.mean(block)
                
                # Umbral: si es muy oscuro, es vacío (negro)
                if avg_brightness > 40:
                    # Mapear brillo a índice de ASCII_CHARS con mayor contraste
                    normalized_brightness = np.clip((avg_brightness - 40) / (255 - 40), 0, 1)
                    char_idx = int(normalized_brightness * (len(ASCII_CHARS) - 1))
                    char = ASCII_CHARS[char_idx]
                    
                    # Dibujar el carácter en el canvas
                    # El blanco es el carácter, el fondo ya es negro
                    cv2.putText(
                        canvas,
                        char,
                        (x, y + cell - 2),
                        FONT,
                        font_scale,
                        255, # Blanco
                        thickness,
                        cv2.LINE_AA
                    )

        # Mostrar el resultado
        cv2.imshow("ASCII_SYSTEM", canvas)

        # Salida con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
