import argparse
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import os
import json
import easyocr

def setup_logging():
    """Configura o logging para o sistema."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

def beep_thread():
    """Executa beep sonoro usando pygame em thread."""
    try:
        import pygame
        def play_beep():
            pygame.mixer.init()
            beep_path = os.path.join(os.path.dirname(__file__), 'beep.wav')
            if os.path.exists(beep_path):
                pygame.mixer.music.load(beep_path)
                pygame.mixer.music.play()
                time.sleep(0.2)
                pygame.mixer.music.stop()
            else:
                logging.warning('Arquivo beep.wav não encontrado.')
        threading.Thread(target=play_beep, daemon=True).start()
    except Exception as e:
        logging.warning(f'Beep não pôde ser executado: {e}')


# Tamanho médio dos veículos em metros

# Carrega parâmetros do config_adas.json
def load_config(path='config_adas.json'):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

CONFIG = load_config()
VEHICLE_LENGTH = {
    'car': CONFIG.get('risk_config', {}).get('car', {}).get('length', 1.2),
    'truck': CONFIG.get('risk_config', {}).get('truck', {}).get('length', 5.0),
    'bus': CONFIG.get('risk_config', {}).get('bus', {}).get('length', 3.0),
    'motorcycle': CONFIG.get('risk_config', {}).get('motorcycle', {}).get('length', 0.7),
    'person': CONFIG.get('risk_config', {}).get('person', {}).get('length', 0.5),
}
FOCAL_LENGTH = CONFIG.get('camera_focal_length', 800)


def estimate_distance(cls_name, y1, y2):
    """Estima a distância do objeto detectado."""
    h_pixel = y2 - y1
    L_real = VEHICLE_LENGTH.get(cls_name, 1.5)
    if h_pixel == 0:
        return 1000
    return (FOCAL_LENGTH * L_real) / h_pixel

def detect_plate(frame, box):
    """Detecta e lê placa de veículo usando EasyOCR."""
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    reader = easyocr.Reader(['pt', 'en'], gpu=False)
    results = reader.readtext(crop)
    if results:
        return results[0][1]
    return None

def decide_risk(cls_name, distance):
    """Decide o nível de risco com base na classe e distância."""
    if cls_name in ('car', 'truck', 'bus'):
        if distance <= 5.0:
            return 2, f'🚨 {cls_name.title()} muito próximo ({distance:.1f}m)!'
        if distance <= 12.0:
            return 1, f'⚠️ {cls_name.title()} à frente ({distance:.1f}m)'
    elif cls_name == 'motorcycle':
        if distance <= 3.0:
            return 2, f'🚨 Moto muito próxima ({distance:.1f}m)!'
        if distance <= 6.0:
            return 1, f'⚠️ Moto à frente ({distance:.1f}m)'
    elif cls_name == 'person':
        if distance <= 2.0:
            return 2, f'🚨 Pedestre muito próximo ({distance:.1f}m)!'
        if distance <= 4.0:
            return 1, f'⚠️ Pedestre à frente ({distance:.1f}m)'
    return -1, ''

def draw_overlay(frame, risks, fps):
    """Desenha sobreposição de informações no frame."""
    H, W = frame.shape[:2]
    msg = ''
    lvl = -1
    for level, text in risks:
        if level > lvl:
            lvl, msg = level, text
    if lvl >= 0 and msg:
        color = (0,255,255) if lvl==1 else (0,0,255)
        cv2.rectangle(frame, (0,0), (W,45), color, -1)
        cv2.putText(frame, msg, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
    cv2.putText(frame, f'FPS: {fps:.1f}', (10,H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
    return frame

def in_focus_zone(x1, y1, x2, y2, W, H):
    """Verifica se o centro do objeto está na zona de interesse definida no config."""
    zone = CONFIG.get('zones', {}).get('proximity', {})
    # Valores relativos (0-1) para largura/altura
    x_min = zone.get('x_min', 0.3)
    x_max = zone.get('x_max', 0.7)
    y_min = zone.get('y_min', 0.3)
    y_max = zone.get('y_max', 0.7)
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    return x_min <= cx <= x_max and y_min <= cy <= y_max

def preprocess_frame(frame):
    """Aplica aumento de contraste e nitidez ao frame."""
    # Aumento de contraste
    alpha = 1.5  # contraste
    beta = 10    # brilho
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # Nitidez
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)
    return frame

def main():
    """Função principal do sistema ADAS."""
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0', help='0=webcam, vídeo ou URL')
    parser.add_argument('--conf', type=float, default=CONFIG.get('performance', {}).get('confidence_threshold', 0.2))
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--skip', type=int, default=CONFIG.get('performance', {}).get('skip_frames', 2), help='Pula frames para aumentar FPS')
    parser.add_argument('--resize', type=int, default=640, help='Redimensiona largura do frame')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        logging.error(f'Modelo {args.model} não encontrado.')
        return

    try:
        model = YOLO(args.model)
    except Exception as e:
        logging.error(f'Erro ao carregar modelo: {e}')
        return

    src = 0 if args.source=='0' else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logging.error(f'Não consegui abrir: {args.source}')
        return

    fps = 0.0
    t0 = time.time()
    frame_count = 0

    logging.info("🚗 Sistema ADAS otimizado iniciado")

    # Classes de interesse
    target_classes = set(CONFIG.get('target_classes', ['car','truck','bus','motorcycle','license plate']))

    try:
        reader = easyocr.Reader(['pt', 'en'], gpu=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                logging.warning('Frame não pôde ser lido. Encerrando.')
                break
            frame_count += 1

            if frame_count % args.skip != 0:
                continue  # pula frames

            H_orig, W_orig = frame.shape[:2]
            # Pré-processamento para melhorar detecção
            frame = preprocess_frame(frame)
            # Redimensiona para processamento rápido
            scale = args.resize / W_orig
            frame_small = cv2.resize(frame, (args.resize, int(H_orig*scale)))

            try:
                results = model.predict(frame_small, conf=args.conf, verbose=False)[0]
            except Exception as e:
                logging.error(f'Erro na predição: {e}')
                continue
            risks = []

            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, str(cls_id))
                    if cls_name not in target_classes:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Reescala para frame original
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)

                    # Foco: só considera objetos na zona central
                    if not in_focus_zone(x1, y1, x2, y2, W_orig, H_orig):
                        continue

                    # Cálculo de distância ajustado
                    distance = estimate_distance(cls_name, y1, y2)
                    level, text = decide_risk(cls_name, distance)

                    color = (0,255,0)
                    if level==1: color=(0,255,255)
                    elif level==2: color=(0,0,255)
                    thickness = 2 if level < 2 else 4

                    cv2.rectangle(frame, (x1,y1),(x2,y2),color,thickness)
                    label = f'{cls_name} {distance:.1f}m'

                    # OCR apenas em regiões detectadas como placa
                    if cls_name == 'license plate':
                        plate = None
                        try:
                            crop = frame[y1:y2, x1:x2]
                            results_plate = reader.readtext(crop)
                            if results_plate:
                                plate = results_plate[0][1]
                                label += f' | Placa: {plate}'
                        except Exception as e:
                            logging.debug(f'Erro OCR: {e}')

                    cv2.putText(frame, label, (x1,max(20,y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1)

                    if level>=1: risks.append((level,text))

            now = time.time()
            fps = 1.0/(now-t0) if now!=t0 else 0
            t0 = now

            frame = draw_overlay(frame, risks, fps)

            if any(r[0]>=1 for r in risks):
                beep_thread()  # beep em thread, sem travar

            cv2.imshow('ADAS Otimizado', frame)
            key = cv2.waitKey(1) & 0xFF
            if key==27: break  # ESC para sair
    except Exception as e:
        logging.error(f'Erro inesperado: {e}')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("✅ Sistema ADAS finalizado")

if __name__ == '__main__':
    main()
