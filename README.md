# Touch Design Interactivo con Python

Este programa es una aplicación de diseño interactivo que utiliza la cámara para detectar manos y reacciona al ritmo de la música cargada.

## Características
- **Seguimiento de Manos**: Utiliza MediaPipe para detectar la posición de tus manos.
- **Audio Reactivo**: Analiza frecuencias (FFT) y amplitud en tiempo real para generar visuales.
- **Partículas Interactivas**: Las partículas son atraídas por tus manos y cambian de color/tamaño según la música.
- **Compatible con Iriun Webcam**: Configurado para detectar dispositivos de video en Linux.

## Requisitos del Sistema (Linux)
Asegúrate de tener instaladas las dependencias del sistema:
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev ffmpeg python3.11-dev gcc g++
```

## Instalación de Librerías Python
```bash
pip install mediapipe opencv-python pygame pyaudio numpy scipy pydub
```

## Cómo usar
1. Coloca un archivo de audio llamado `audio.mp3` en la misma carpeta que el script (o cambia el nombre en `main.py`).
2. Conecta tu celular con **Iriun Webcam** y asegúrate de que Linux lo reconozca (usualmente como `/dev/video0` o `/dev/video1`).
3. Ejecuta el programa:
   ```bash
   python main.py
   ```

## Controles
- **Mover las manos**: Las partículas seguirán la punta de tu dedo índice.
- **Música**: Los visuales del fondo y el comportamiento de las partículas cambiarán con el ritmo.
- **ESC**: Salir del programa.
