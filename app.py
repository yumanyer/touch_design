import cv2
import numpy as np
import sounddevice as sd
import random

# ================= CONFIG =================

CAMERA_INDEX = 0

# Caracteres para el flujo
ASCII_CHARS = " .,:;irsXA253hMHGS#9B&@"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Parámetros de Audio
AUDIO_GAIN = 45.0
AUDIO_SMOOTH = 0.85

# ==========================================

audio_level = 0.0
# Almacenar posiciones verticales para el efecto de flujo
drops = None

def audio_callback(indata, frames, time_info, status):
    global audio_level
    rms = np.sqrt(np.mean(indata**2))
    volume = rms * AUDIO_GAIN
    audio_level = (AUDIO_SMOOTH * audio_level) + ((1 - AUDIO_SMOOTH) * volume)

def main():
    global audio_level, drops

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

    cv2.namedWindow("REACTIVE_FLOW", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("REACTIVE_FLOW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Inicializar el rastro (trail)
    trail_canvas = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Inicializar drops y trail_canvas si es la primera vez o cambió el tamaño
        if drops is None or trail_canvas is None or trail_canvas.shape[:2] != (h, w):
            cell_size = 10
            drops = [0] * (w // cell_size)
            trail_canvas = np.zeros((h, w), dtype=np.uint8)

        # Procesamiento de silueta (CLAHE para visibilidad)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        silueta = clahe.apply(gray)
        
        # Parámetros dinámicos por audio
        level = np.clip(audio_level, 0.0, 1.0)
        # cell_size cambia con el audio: más ruido -> más columnas (más detalle)
        cell_size = int(20 - (level * 12)) 
        cell_size = max(cell_size, 6)
        
        # Velocidad de caída basada en audio
        speed_boost = int(level * 15)
        
        # Efecto de persistencia (desvanecimiento lento del canvas anterior)
        # Esto crea los "trails" o rastros
        fade_speed = 40 + int(level * 60)
        trail_canvas = cv2.addWeighted(trail_canvas, 0.9, np.zeros_like(trail_canvas), 0.1, -fade_speed)
        
        # Re-inicializar drops si el cell_size cambió drásticamente
        num_cols = w // cell_size
        if len(drops) != num_cols:
            drops = [random.randint(-h, 0) for _ in range(num_cols)]

        # Dibujar el flujo sobre la silueta
        for i in range(num_cols):
            x = i * cell_size
            y = drops[i]
            
            if 0 <= y < h:
                # Solo dibujar si la silueta en esa posición tiene brillo
                # Esto hace que el flujo "revele" tu cuerpo
                brightness = silueta[y, x]
                
                if brightness > 40:
                    # Carácter basado en brillo
                    char_idx = int((brightness / 255.0) * (len(ASCII_CHARS) - 1))
                    char = ASCII_CHARS[char_idx]
                    
                    # Tamaño y grosor reactivo
                    font_scale = cell_size / 15.0
                    thickness = 1 if level < 0.6 else 2
                    
                    cv2.putText(
                        trail_canvas,
                        char,
                        (x, y),
                        FONT,
                        font_scale,
                        255,
                        thickness,
                        cv2.LINE_AA
                    )

            # Actualizar posición de la gota
            # La velocidad base es 5, más el boost del audio
            drops[i] += 5 + speed_boost
            
            # Reiniciar gota si sale de pantalla o aleatoriamente para variedad
            if drops[i] >= h or (random.random() > 0.95):
                drops[i] = random.randint(-20, 0)

        # Combinar el flujo con un fondo negro puro
        cv2.imshow("REACTIVE_FLOW", trail_canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
