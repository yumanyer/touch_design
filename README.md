# Touch Design: Audio-Reactive ASCII System

Este sistema transforma la entrada de video en una representación puramente de caracteres ASCII, donde la **resolución visual es controlada dinámicamente por el audio**.

## Concepto
- ❌ **Sin video "normal"**: La imagen existe solo como una matriz de caracteres.
- ✅ **ASCII Puro**: El blanco son caracteres, el negro es vacío absoluto.
- 🎵 **Audio como Control de Resolución**:
  - **Poco ruido**: Caracteres grandes, imagen abstracta y pobre en detalle.
  - **Mucho ruido**: Caracteres pequeños, alta densidad y máximo detalle.
- 🧠 **Reactividad Orgánica**: El sistema no decora, reacciona. El sonido es la fuente de la información visual.

## Requisitos
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- SoundDevice
- PortAudio (Librería del sistema)

## Instalación
```bash
# Instalar dependencias de Python
pip install opencv-python numpy sounddevice

# En Linux (Ubuntu/Debian), instalar PortAudio
sudo apt-get install libportaudio2
```

## Uso
Ejecuta el script principal:
```bash
python app.py
```
- **ESC**: Salir del sistema.
- El sistema se iniciará en modo ventana (ajustable a pantalla completa en el código).

## Configuración (app.py)
Puedes ajustar los siguientes parámetros en la sección `CONFIG`:
- `MIN_CELL`: Tamaño mínimo de celda (máximo detalle con ruido fuerte).
- `MAX_CELL`: Tamaño máximo de celda (mínimo detalle en silencio).
- `AUDIO_GAIN`: Sensibilidad del micrófono.
- `AUDIO_SMOOTH`: Suavizado de la transición de resolución.
