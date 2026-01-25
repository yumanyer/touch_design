import cv2
import numpy as np
import sounddevice as sd
import threading

# ================= CONFIG =================

CAMERA_INDEX = 0

ASCII_CHARS = " .:-=+*#%@"
FONT = cv2.FONT_HERSHEY_SIMPLEX

MIN_CELL = 4    # música fuerte → más detalle
MAX_CELL = 16   # silencio → menos detalle

AUDIO_SMOOTH = 0.9  # estabilidad visual

# ==========================================

audio_level = 0.0


def audio_callback(indata, frames, time, status):
    global audio_level
    volume = np.linalg.norm(indata) * 10
    audio_level = AUDIO_SMOOTH * audio_level + (1 - AUDIO_SMOOTH) * volume


def audio_to_cell_size(volume):
    volume = np.clip(volume, 0.0, 1.0)
    cell = MAX_CELL - volume * (MAX_CELL - MIN_CELL)
    return int(cell)


def main():
    global audio_level

    # ---- Audio stream ----
    stream = sd.InputStream(
        channels=1,
        callback=audio_callback,
        samplerate=44100
    )
    stream.start()

    # ---- Video ----
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    cv2.namedWindow("ASCII", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Blanco y negro duro
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        # Tamaño de celda según audio
        cell = audio_to_cell_size(audio_level)

        # Canvas negro
        canvas = np.zeros((h, w), dtype=np.uint8)

        for y in range(0, h, cell):
            for x in range(0, w, cell):
                block = bw[y:y+cell, x:x+cell]
                if block.size == 0:
                    continue

                mean = block.mean()

                if mean > 200:
                    idx = int((mean / 255) * (len(ASCII_CHARS) - 1))
                    char = ASCII_CHARS[idx]
                else:
                    char = " "

                cv2.putText(
                    canvas,
                    char,
                    (x, y + cell),
                    FONT,
                    cell / 25,
                    255,
                    1,
                    cv2.LINE_AA
                )

        cv2.imshow("ASCII", canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

