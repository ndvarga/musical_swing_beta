# Musical Swing (Raspberry Pi, real hardware)

## Run (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If audio fails on macOS (PortAudio), install:
```bash
brew install portaudio
pip install --force-reinstall sounddevice
```

## Tests
```bash
python -m swing_sim.tests
```
