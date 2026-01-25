import cv2
import numpy as np
import sounddevice as sd

# ================= CONFIG =================

CAMERA_INDEX = 0

# Conjunto de caracteres exacto solicitado
ASCII_CHARS = " .,:;irsXA253hMHGS#9B&@"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Rango de resolución (tamaño de celda)
# Poco ruido -> caracteres grandes (MAX_CELL), imagen abstracta
# Mucho ruido -> caracteres chicos (MIN_CELL), más densidad/detalle
MIN_CELL = 4    
MAX_CELL = 60   

# Sensibilidad y suavizado
AUDIO_GAIN = 40.0
AUDIO_SMOOTH = 0.7  # Menos suavizado para una reacción más nerviosa y directa

# ==========================================

audio_level = 0.0

def audio_callback(indata, frames, time_info, status):
    global audio_level
    rms = np.sqrt(np.mean(indata**2))
    volume = rms * AUDIO_GAIN
    audio_level = (AUDIO_SMOOTH * audio_level) + ((1 - AUDIO_SMOOTH) * volume)

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

    cv2.namedWindow("ASCII_SYSTEM", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("ASCII_SYSTEM", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Procesamiento de imagen para máximo contraste
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Ecualización adaptativa para asegurar que la silueta siempre sea visible
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Determinar resolución basada en audio
        level = np.clip(audio_level, 0.0, 1.0)
        cell = int(MAX_CELL - (level * (MAX_CELL - MIN_CELL)))
        cell = max(cell, MIN_CELL)

        # Canvas negro absoluto
        canvas = np.zeros((h, w), dtype=np.uint8)

        # Escala de fuente proporcional a la celda para que los caracteres se toquen
        # Ajustado para que se vea denso como en la referencia
        font_scale = cell / 18.0
        thickness = 1 if cell < 12 else 2

        # Iterar por la cuadrícula estática
        for y in range(0, h, cell):
            for x in range(0, w, cell):
                block = gray[y:y+cell, x:x+cell]
                if block.size == 0: continue

                avg = np.mean(block)
                
                # Umbral para que el negro sea vacío
                if avg > 30:
                    # Mapear brillo a carácter
                    char_idx = int((avg / 255.0) * (len(ASCII_CHARS) - 1))
                    char = ASCII_CHARS[char_idx]
                    
                    # Dibujar carácter en posición fija (rejilla)
                    cv2.putText(
                        canvas,
                        char,
                        (x, y + cell),
                        FONT,
                        font_scale,
                        255,
                        thickness,
                        cv2.LINE_AA
                    )

        cv2.imshow("ASCII_SYSTEM", canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
