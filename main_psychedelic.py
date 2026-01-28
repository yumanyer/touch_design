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
import colorsys
from scipy.fftpack import fft
from pydub import AudioSegment
import pyaudio
import os

# --- CONFIGURACIÓN ---
WIDTH, HEIGHT = 1280, 720
FPS = 60
AUDIO_FILE = "audio.mp3"
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
        self.amplitude = 0
        self.fft_data = np.zeros(CHUNK // 2)
        self.bass = 0
        self.mid = 0
        self.high = 0
        
        if os.path.exists(file_path):
            try:
                self.audio_segment = AudioSegment.from_file(file_path)
                self.audio_segment = self.audio_segment.set_frame_rate(RATE).set_channels(CHANNELS)
                self.raw_data = self.audio_segment.raw_data
            except Exception as e:
                print(f"Error cargando audio: {e}")
                self.raw_data = None
        else:
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
            
            audio_data = np.frombuffer(chunk_data, dtype=np.int16)
            if len(audio_data) > 0:
                if CHANNELS == 2:
                    audio_data = (audio_data[0::2].astype(float) + audio_data[1::2].astype(float)) / 2
                
                self.amplitude = np.abs(audio_data).mean() / 32768.0
                fast_fourier = np.abs(fft(audio_data)[:CHUNK // 2])
                self.fft_data = fast_fourier / (CHUNK * 32768.0)
                
                # Bandas de frecuencia
                self.bass = np.mean(self.fft_data[:10]) * 100
                self.mid = np.mean(self.fft_data[10:50]) * 100
                self.high = np.mean(self.fft_data[50:]) * 100

            stream.write(chunk_data)
            offset += CHUNK * bytes_per_sample
            
        stream.stop_stream()
        stream.close()
        p.terminate()

# --- MOTOR VISUAL PSICODÉLICO ---
class PsychedelicEngine:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.feedback_surface = pygame.Surface((width, height))
        self.hue = 0
        self.rotation = 0

    def update(self, amp, bass, hand_pos):
        self.hue = (self.hue + 0.005 + bass * 0.1) % 1.0
        self.rotation += 0.5 + bass * 5
        
        # Efecto de Feedback (Estela)
        # Escalar ligeramente la superficie anterior para crear túnel
        scale_factor = 1.005 + amp * 0.02
        scaled_w = int(self.width * scale_factor)
        scaled_h = int(self.height * scale_factor)
        temp_surf = pygame.transform.smoothscale(self.surface, (scaled_w, scaled_h))
        
        # Centrar y aplicar transparencia
        self.surface.fill((0, 0, 0))
        self.surface.blit(temp_surf, ((self.width - scaled_w)//2, (self.height - scaled_h)//2))
        
        # Oscurecer para que la estela desaparezca
        darken = pygame.Surface((self.width, self.height))
        darken.set_alpha(int(20 - bass * 10))
        darken.fill((0, 0, 0))
        self.surface.blit(darken, (0, 0))

    def draw_shapes(self, amp, bass, hand_pos):
        color = [int(c * 255) for c in colorsys.hsv_to_rgb(self.hue, 1, 1)]
        
        for hx, hy in hand_pos:
            # Dibujar mandalas en las manos
            num_points = 8
            radius = 50 + bass * 200
            for i in range(num_points):
                angle = np.radians(i * (360 / num_points) + self.rotation)
                px = hx + np.cos(angle) * radius
                py = hy + np.sin(angle) * radius
                
                # Líneas conectadas
                pygame.draw.line(self.surface, color, (hx, hy), (px, py), int(2 + amp * 10))
                pygame.draw.circle(self.surface, color, (int(px), int(py)), int(5 + bass * 20))
                
                # Simetría (Espejo)
                m_hx, m_hy = self.width - hx, hy
                m_px, m_py = self.width - px, py
                pygame.draw.line(self.surface, color, (m_hx, m_hy), (m_px, m_py), int(2 + amp * 10))

    def apply_glitch(self, bass):
        if bass > 0.8:
            # Desplazamiento aleatorio de filas
            for _ in range(5):
                y = np.random.randint(0, self.height)
                h = np.random.randint(10, 50)
                shift = np.random.randint(-50, 50)
                rect = pygame.Rect(0, y, self.width, h)
                sub = self.surface.subsurface(rect).copy()
                self.surface.blit(sub, (shift, y))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PSYCHEDELIC TOUCH DESIGN")
    clock = pygame.time.Clock()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)

    analyzer = AudioAnalyzer(AUDIO_FILE)
    analyzer.start()
    
    engine = PsychedelicEngine(WIDTH, HEIGHT)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

        # 1. Cámara y Manos
        success, frame = cap.read()
        if not success: continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        hand_pos = []
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                lm8 = hand_lms.landmark[8]
                hand_pos.append((int(lm8.x * WIDTH), int(lm8.y * HEIGHT)))

        # 2. Actualizar Motor Psicodélico
        amp = analyzer.amplitude
        bass = analyzer.bass
        
        engine.update(amp, bass, hand_pos)
        engine.draw_shapes(amp, bass, hand_pos)
        engine.apply_glitch(bass)

        # 3. Mezclar con la cámara (opcional, aquí lo haremos sutil)
        # Convertir frame de OpenCV a Pygame
        frame_pygame = pygame.image.frombuffer(frame_rgb.tostring(), frame_rgb.shape[1::-1], "RGB")
        frame_pygame = pygame.transform.scale(frame_pygame, (WIDTH, HEIGHT))
        frame_pygame.set_alpha(60) # Transparencia para ver la cámara de fondo
        
        screen.fill((0, 0, 0))
        screen.blit(engine.surface, (0, 0))
        screen.blit(frame_pygame, (0, 0), special_flags=pygame.BLEND_ADD)

        pygame.display.flip()
        clock.tick(FPS)

    analyzer.playing = False
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
