# ADAS 1.0 — Guia Rápido

## Requisitos
- Python 3.9+
- GPU NVIDIA opcional (recomendado) com drivers + CUDA/cuDNN
- Webcam ou arquivo de vídeo de teste

## Instalação
```bash
python -m venv .venv
# Ative a venv...
pip install --upgrade pip
pip install -r requirements.txt
```
> Dica: `ultralytics` baixará o modelo YOLO automaticamente na primeira execução.

## Execução
Webcam (padrão):
```bash
python adas.py
```

Arquivo de vídeo:
```bash
python adas.py --source caminho/do/video.mp4
```

Ajustes úteis:
```bash
# Mais precisão vs. mais velocidade
python adas.py --conf 0.35 --skip-frames 2 --imgsz 640 --device auto --half
# Zona de proximidade (normalizada 0-1)
python adas.py --proximity-x-min 0.1 --proximity-x-max 0.9 --proximity-y-min 0.3 --proximity-y-max 0.8
# Desligar/ligar detecção de faixa
python adas.py --lane-detection
```

Atalhos no vídeo:
- `ESC`: sair
- `P`: pausar/retomar
- `L`: ligar/desligar detecção de faixa
- `H`: ajuda no console

## Sons de alerta
Coloque um `beep.wav` na mesma pasta do `adas.py`. No Windows, `winsound` é usado automaticamente. Em outros sistemas, caímos para `playsound`; se não encontrar o arquivo, o alerta sonoro é ignorado.

## Observações
- Não commitar a pasta `.venv` no repositório.
- Para gerar executável com PyInstaller, use o helper `resource_path()` que já está no código sugerido.
