import cv2
import mediapipe as mp
try:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_draw
except ImportError:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_draw

import numpy as np
import pygame
import sys
import time
import threading
from scipy.fftpack import fft
from pydub import AudioSegment
import pyaudio
import os

# --- CONFIGURACIÓN ---
WIDTH, HEIGHT = 1280, 720
FPS = 60
AUDIO_FILE = "audio.mp3"  # Cambia esto al nombre de tu archivo
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100

# --- INICIALIZACIÓN DE MEDIAPIPE ---
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# --- CLASE PARA MANEJO DE AUDIO ---
class AudioAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.playing = False
        self.data_chunk = np.zeros(CHUNK)
        self.fft_data = np.zeros(CHUNK // 2)
        self.amplitude = 0
        
        if os.path.exists(file_path):
            self.audio_segment = AudioSegment.from_file(file_path)
            self.audio_segment = self.audio_segment.set_frame_rate(RATE).set_channels(CHANNELS)
            self.raw_data = self.audio_segment.raw_data
        else:
            print(f"Archivo {file_path} no encontrado. Usando modo silencio.")
            self.raw_data = None

    def start(self):
        if self.raw_data:
            self.playing = True
            threading.Thread(target=self._play_thread, daemon=True).start()

    def _play_thread(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)
        
        offset = 0
        bytes_per_sample = 2 * CHANNELS
        
        while self.playing and offset < len(self.raw_data):
            chunk_data = self.raw_data[offset:offset + CHUNK * bytes_per_sample]
            if not chunk_data: break
            
            # Convertir a numpy para análisis
            audio_data = np.frombuffer(chunk_data, dtype=np.int16)
            if len(audio_data) > 0:
                # Promediar canales si es estéreo
                if CHANNELS == 2:
                    audio_data = (audio_data[0::2] + audio_data[1::2]) / 2
                
                self.data_chunk = audio_data
                self.amplitude = np.abs(audio_data).mean() / 32768.0
                
                # FFT
                fast_fourier = np.abs(fft(audio_data)[:CHUNK // 2])
                self.fft_data = fast_fourier / (CHUNK * 32768.0)

            stream.write(chunk_data)
            offset += CHUNK * bytes_per_sample
            
        stream.stop_stream()
        stream.close()
        p.terminate()
        self.playing = False

# --- CLASE PARA PARTÍCULAS ---
class Particle:
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.vel = np.random.randn(2) * 2
        self.acc = np.zeros(2)
        self.color = [255, 255, 255]
        self.size = np.random.randint(2, 6)
        self.life = 255

    def apply_force(self, force):
        self.acc += force

    def update(self, amp):
        self.vel += self.acc
        self.pos += self.vel
        self.acc *= 0
        self.vel *= 0.95  # Fricción
        self.life -= 2
        self.size_mod = self.size * (1 + amp * 10)

    def draw(self, surface):
        if self.life > 0:
            alpha_color = (*self.color, self.life)
            pygame.draw.circle(surface, self.color, (int(self.pos[0]), int(self.pos[1])), int(self.size_mod))

# --- PROGRAMA PRINCIPAL ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Manus Touch Design - Audio & Hand Reactive")
    clock = pygame.time.Clock()
    
    # Intentar abrir la cámara (Iriun suele ser 0 o 1 en Linux)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    analyzer = AudioAnalyzer(AUDIO_FILE)
    analyzer.start()

    particles = []
    hand_pos = []

    running = True
    while running:
        screen.fill((10, 10, 20)) # Fondo oscuro
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 1. Captura de Cámara y Seguimiento de Manos
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        hand_pos = []
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # Obtener posición de la punta del índice (Landmark 8) y pulgar (Landmark 4)
                lm8 = hand_lms.landmark[8]
                lm4 = hand_lms.landmark[4]
                hx, hy = int(lm8.x * WIDTH), int(lm8.y * HEIGHT)
                px, py = int(lm4.x * WIDTH), int(lm4.y * HEIGHT)
                hand_pos.append((hx, hy))
                
                # Detectar "pellizco" (distancia corta entre índice y pulgar)
                dist_pinch = np.sqrt((hx-px)**2 + (hy-py)**2)
                if dist_pinch < 40:
                    # Crear explosión de partículas en el pellizco
                    for _ in range(5):
                        p = Particle(hx, hy)
                        p.vel = np.random.randn(2) * 10
                        particles.append(p)
                
                # Dibujar esqueleto de la mano en una superficie pequeña si se desea
                # mp_draw.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS)

        # 2. Lógica de Partículas
        amp = analyzer.amplitude
        fft_vals = analyzer.fft_data
        
        # Crear partículas basadas en el audio
        if amp > 0.05:
            for _ in range(int(amp * 50)):
                px = np.random.randint(0, WIDTH)
                py = np.random.randint(0, HEIGHT)
                particles.append(Particle(px, py))

        # Actualizar y dibujar partículas
        for p in particles[:]:
            # Atracción a las manos
            for hx, hy in hand_pos:
                dist = np.linalg.norm(p.pos - [hx, hy])
                if dist < 300:
                    force = (np.array([hx, hy]) - p.pos) * 0.05
                    p.apply_force(force)
            
            # Color basado en FFT (frecuencias bajas = rojo, altas = azul)
            if len(fft_vals) > 0:
                low_freq = np.mean(fft_vals[:10]) * 1000
                high_freq = np.mean(fft_vals[100:200]) * 1000
                p.color = [
                    min(255, int(100 + low_freq)),
                    min(255, int(150 + amp * 500)),
                    min(255, int(200 + high_freq))
                ]

            p.update(amp)
            p.draw(screen)
            if p.life <= 0:
                particles.remove(p)

        # Limitar número de partículas para rendimiento
        if len(particles) > 1000:
            particles = particles[-1000:]

        # 3. Visualización de FFT en el fondo
        if len(fft_vals) > 0:
            bar_width = WIDTH // len(fft_vals[:100])
            for i, val in enumerate(fft_vals[:100]):
                h = int(val * HEIGHT * 5)
                pygame.draw.rect(screen, (50, 50, 100), (i * bar_width, HEIGHT - h, bar_width - 1, h))

        # 4. Dibujar punteros de manos
        for hx, hy in hand_pos:
            pygame.draw.circle(screen, (0, 255, 255), (hx, hy), 15, 2)
            pygame.draw.circle(screen, (0, 255, 255), (hx, hy), 5)

        pygame.display.flip()
        clock.tick(FPS)

    analyzer.playing = False
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
