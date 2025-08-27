VEHICLE_HISTORY = {}

import argparse
import time
from collections import deque
import cv2
import numpy as np
import os
import json
import pygame
import easyocr
from icons import ADASIcons

def beep():
    try:
        pygame.mixer.init()
        beep_path = os.path.join(os.path.dirname(__file__), 'beep.wav')
        if os.path.exists(beep_path):
            pygame.mixer.music.load(beep_path)
            pygame.mixer.music.play()
            time.sleep(0.2)
            pygame.mixer.music.stop()
    except Exception as e:
        pass

def load_config(path='config_adas.json'):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

CONFIG = load_config()
TARGET_NAMES = set(CONFIG.get('target_classes', ['car','truck','bus','motorcycle','stop sign','traffic light','person','license plate']))
PROXIMITY_ZONE = CONFIG.get('zones', {}).get('proximity', {'x_min': 0.2, 'x_max': 0.8, 'y_min': 0.2, 'y_max': 0.8})
RISK_CFG = CONFIG.get('risk_config', {
    'car': {'area_ratio_high': 0.05, 'area_ratio_mid': 0.08},
    'truck': {'area_ratio_high': 0.08, 'area_ratio_mid': 0.01},
    'bus': {'area_ratio_high': 0.25, 'area_ratio_mid': 0.12},
    'motorcycle': {'area_ratio_high': 0.1, 'area_ratio_mid': 0.05},
    'person': {'area_ratio_high': 0.08, 'area_ratio_mid': 0.04}
})
RECENT_RISKS = deque(maxlen=3)
VEHICLE_LENGTH = {
    'car': CONFIG.get('risk_config', {}).get('car', {}).get('length', 0.5),
    'truck': CONFIG.get('risk_config', {}).get('truck', {}).get('length', 0.5),
    'bus': CONFIG.get('risk_config', {}).get('bus', {}).get('length', 0.5),
    'motorcycle': CONFIG.get('risk_config', {}).get('motorcycle', {}).get('length', 0.5),
    'person': CONFIG.get('risk_config', {}).get('person', {}).get('length', 0.5),
}
FOCAL_LENGTH = CONFIG.get('camera_focal_length', 300)
COLLISION_DISTANCE_CRITICAL = CONFIG.get('alerts', {}).get('collision_distance_critical', 0.7)

def preprocess_frame(frame):
    alpha = 1.5
    beta = 10
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)
    return frame

def estimate_distance(cls_name, y1, y2):
    FOCAL_LENGTH = CONFIG.get('camera_focal_length', 800)
    h = y2 - y1
    if h is None or h <= 0:
        return 1000.0
    try:
        if cls_name in VEHICLE_LENGTH:
            L_real = VEHICLE_LENGTH.get(cls_name, 1.5)
        elif cls_name == 'person':
            L_real = 1.7
        else:
            L_real = 1.5
        return float((FOCAL_LENGTH * L_real) / h)
    except Exception:
        return 1000.0

def in_proximity_zone(x1, y1, x2, y2, W, H):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    rx1 = PROXIMITY_ZONE['x_min'] * W
    rx2 = PROXIMITY_ZONE['x_max'] * W
    ry1 = PROXIMITY_ZONE['y_min'] * H
    ry2 = PROXIMITY_ZONE['y_max'] * H
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

def area_ratio(x1, y1, x2, y2, W, H):
    return max(0, x2 - x1) * max(0, y2 - y1) / float(W * H)

def decide_risk(cls_name, aratio, proximity_hit):
    if not proximity_hit:
        return -1, ''
    
    if cls_name in ('car', 'truck', 'bus'):
        if aratio >= RISK_CFG.get(cls_name, RISK_CFG['car'])['area_ratio_high']:
            return 2, f'🚨 PERIGO: {cls_name.title()} MUITO próximo no parabrisa!'
        if aratio >= RISK_CFG.get(cls_name, RISK_CFG['car'])['area_ratio_mid']:
            return 1, f'⚠️ Atenção: {cls_name.title()} à frente no parabrisa'
    elif cls_name == 'motorcycle':
        if aratio >= RISK_CFG['motorcycle']['area_ratio_high']:
            return 2, '🚨 PERIGO: Moto MUITO próxima no parabrisa!'
        if aratio >= RISK_CFG['motorcycle']['area_ratio_mid']:
            return 1, '⚠️ Atenção: Moto à frente no parabrisa'
    elif cls_name == 'stop sign':
        return 1, '🛑 PLACA DE PARE no parabrisa - Reduza velocidade!'
    elif cls_name == 'traffic light':
        return 0, '🚦 Semáforo detectado no parabrisa'
    elif cls_name == 'person':
        if aratio >= RISK_CFG['person']['area_ratio_high']:
            return 2, '🚨 PERIGO: Pedestre MUITO perto no parabrisa!'
        if aratio >= RISK_CFG['person']['area_ratio_mid']:
            return 1, '⚠️ Atenção: Pedestre à frente no parabrisa'
    
    return -1, ''

def draw_overlay(frame, risks, fps, lane_warning, host_speed=None, proximity_zone_visible=True):
    H, W = frame.shape[:2]
    icons = ADASIcons(icon_size=32)
    rx1 = int(PROXIMITY_ZONE['x_min'] * W)
    rx2 = int(PROXIMITY_ZONE['x_max'] * W)
    ry1 = int(PROXIMITY_ZONE['y_min'] * H)
    ry2 = int(PROXIMITY_ZONE['y_max'] * H)
    overlay = frame.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 255), -1)
    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 3)
    cv2.putText(frame, '🚗 PARABRISA - ZONA DE RISCO', (rx1, ry1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    square_size = int(min(W, H) * 0.35)
    sq_cx = W // 2
    sq_cy = int(H * 0.55)
    sq1 = (sq_cx - square_size // 2, sq_cy - square_size // 2)
    sq2 = (sq_cx + square_size // 2, sq_cy + square_size // 2)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, sq1, sq2, (0, 0, 255), -1)
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    cv2.rectangle(frame, sq1, sq2, (0, 0, 255), 2)
    cv2.putText(frame, '⚠️ ZONA CRÍTICA', (sq1[0], sq1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.rectangle(frame, (0, 0), (W, 70), (0, 0, 0), -1)
    cv2.putText(frame, 'ADAS - SISTEMA AVANÇADO DE ALERTA', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f'FPS: {fps:.1f}', (W - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if host_speed is not None:
        speed_kmh = host_speed * 3.6
        speed_color = (0, 255, 0)
        if speed_kmh > 80:
            speed_color = (0, 255, 255)
        if speed_kmh > 100:
            speed_color = (0, 0, 255)
            
        cv2.putText(frame, f'Velocidade: {speed_kmh:.1f} km/h', (W - 300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 2)
    
    msg = ''
    lvl = -1
    alert_messages = []
    
    for level, text in risks:
        if level > lvl:
            lvl, msg = level, text
        alert_messages.append((level, text))
    
    y_offset = 50
    for level, text in alert_messages[:3]:
        color = (0, 255, 255) if level == 1 else (0, 0, 255)
        
        icon = None
        if 'car' in text.lower() or 'truck' in text.lower() or 'bus' in text.lower():
            icon = icons.icons['car']
        elif 'moto' in text.lower():
            icon = icons.icons['motorcycle']
        elif 'pedestre' in text.lower():
            icon = icons.icons['person']
        elif 'placa' in text.lower() or 'pare' in text.lower():
            icon = icons.icons['stop_sign']
        elif 'semáforo' in text.lower():
            icon = icons.icons['traffic_light']
        elif level == 2:
            icon = icons.icons['danger']
        elif level == 1:
            icon = icons.icons['warning']
        else:
            icon = icons.icons['info']
        
        if icon is not None:
            icon_x = 10
            icon_y = y_offset - 25
            if icon.shape[0] != 24:
                icon = cv2.resize(icon, (24, 24))
            roi = frame[icon_y:icon_y+24, icon_x:icon_x+24]
            roi = cv2.addWeighted(icon, 1.0, roi, 0.5, 0)
            frame[icon_y:icon_y+24, icon_x:icon_x+24] = roi
        
        cv2.putText(frame, text, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25
    
    if lvl >= 0 and msg:
        color = (0, 255, 255) if lvl == 1 else (0, 0, 255)
        thickness = 4 if lvl == 1 else 6
        
        cv2.rectangle(frame, (0, H-80), (W, H), color, -1)
        
        main_icon = icons.icons['danger'] if lvl == 2 else icons.icons['warning']
        main_icon = cv2.resize(main_icon, (60, 60))
        icon_x = W//2 - 200
        icon_y = H - 70
        roi = frame[icon_y:icon_y+60, icon_x:icon_x+60]
        roi = cv2.addWeighted(main_icon, 1.0, roi, 0.3, 0)
        frame[icon_y:icon_y+60, icon_x:icon_x+60] = roi
        
        cv2.putText(frame, '🚨 ALERTA PRINCIPAL 🚨', (W//2 - 120, H-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, msg, (W//2 - 140, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), thickness)
    
    if lane_warning:
        cv2.rectangle(frame, (0, H-120), (W, H-80), (0, 0, 255), -1)
        cv2.putText(frame, '⚠️ AVISO DE FAIXA', (10, H-95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, lane_warning, (10, H-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    status_icon = icons.icons['info']
    status_icon = cv2.resize(status_icon, (20, 20))
    icon_x = 10
    icon_y = H - 20
    roi = frame[icon_y-20:icon_y, icon_x:icon_x+20]
    roi = cv2.addWeighted(status_icon, 1.0, roi, 0.5, 0)
    frame[icon_y-20:icon_y, icon_x:icon_x+20] = roi
    
    cv2.putText(frame, '✅ SISTEMA ATIVO - FOCADO NO PARABRISA', (35, H-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sistema ADAS Avançado - Detecção de Faixa e Proximidade Horizontal')
    parser.add_argument('--source', default=None, help='0 (webcam), caminho de vídeo, ou URL RTSP/HTTP')
    parser.add_argument('--conf', type=float, default=0.4, help='Confiança mínima para detecção')
    parser.add_argument('--model', default='yolov8n.pt', help='Modelo YOLO a usar')
    parser.add_argument('--show-names', nargs='*', default=None, help='Classes específicas para mostrar')
    parser.add_argument('--proximity-x-min', type=float, default=0.1, help='Limite esquerdo da zona de proximidade (0-1)')
    parser.add_argument('--proximity-x-max', type=float, default=0.9, help='Limite direito da zona de proximidade (0-1)')
    parser.add_argument('--proximity-y-min', type=float, default=0.3, help='Limite superior da zona de proximidade (0-1)')
    parser.add_argument('--proximity-y-max', type=float, default=0.8, help='Limite inferior da zona de proximidade (0-1)')
    parser.add_argument('--skip-frames', type=int, default=3, help='Pular frames para melhorar FPS (recomendado: 3-5)')
    parser.add_argument('--lane-detection', action='store_true', help='Ativar detecção de mudança de faixa')
    parser.add_argument('--video-delay', type=int, default=1, help='Tempo de espera entre frames em ms (aumente para rodar mais devagar, ex: 30, 50, 100)')
    args = parser.parse_args()

    if args.proximity_x_min != 0.1 or args.proximity_x_max != 0.9 or args.proximity_y_min != 0.3 or args.proximity_y_max != 0.8:
        PROXIMITY_ZONE.update({
            'x_min': args.proximity_x_min,
            'x_max': args.proximity_x_max,
            'y_min': args.proximity_y_min,
            'y_max': args.proximity_y_max
        })

    from ultralytics import YOLO
    model = YOLO(args.model)

    src = 0 if args.source is None or args.source == '0' else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f'Não consegui abrir a fonte: {args.source}')

    names = model.names
    allowed = set(args.show_names) if args.show_names else TARGET_NAMES

    t_last_beep = 0
    fps = 0.0
    t0 = time.time()
    frames = 0
    frame_count = 0
    lane_warning = ""
    host_speed = None
    prev_nearest_distance = None

    print(f"🚗 Sistema ADAS AVANÇADO iniciado com foco em: {', '.join(allowed)}")
    print(f"📍 Zona de proximidade (parabrisa): X({PROXIMITY_ZONE['x_min']:.2f}-{PROXIMITY_ZONE['x_max']:.2f}) Y({PROXIMITY_ZONE['y_min']:.2f}-{PROXIMITY_ZONE['y_max']:.2f})")
    print(f"🛣️ Detecção de faixa: {'ATIVADA' if args.lane_detection else 'DESATIVADA'}")
    print(f"⚡ Skip frames: {args.skip_frames} (FPS otimizado)")
    print("Pressione 'ESC' para sair, 'P' para pausar/despausar, 'L' para toggle detecção de faixa")

    paused = False
    lane_detection_enabled = args.lane_detection

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame_count += 1
            if frame_count % args.skip_frames != 0:
                continue
            
            H, W = frame.shape[:2]
            rx1 = int(PROXIMITY_ZONE['x_min'] * W)
            rx2 = int(PROXIMITY_ZONE['x_max'] * W)
            ry1 = int(PROXIMITY_ZONE['y_min'] * H)
            ry2 = int(PROXIMITY_ZONE['y_max'] * H)
            results = model.predict(frame, conf=args.conf, verbose=False)[0]
            risks = []
            nearest_distance = None
            nearest_obj_id = None
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = names.get(cls_id, str(cls_id))
                    if cls_name not in allowed:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    distance = estimate_distance(cls_name, y1, y2)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        cv2.circle(frame, (cx, cy), 30, (0, 255, 255), 3)
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2 and distance is not None and distance <= 1.5:
                        cv2.circle(frame, (cx, cy), 40, (0, 0, 255), 4)
                        label = f'🚨 ALERTA: {cls_name} muito próximo! ({distance:.1f}m)'
                        risks.append((2, label))
            if prev_nearest_distance is not None and nearest_distance is not None and fps > 0:
                host_speed = abs(prev_nearest_distance - nearest_distance) * fps
            prev_nearest_distance = nearest_distance
            
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = names.get(cls_id, str(cls_id))
                    if cls_name not in allowed:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf_score = float(box.conf[0])
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    proximity_hit = in_proximity_zone(x1, y1, x2, y2, W, H)
                    if not proximity_hit:
                        continue
                    
                    ar = area_ratio(x1, y1, x2, y2, W, H)
                    distance = estimate_distance(cls_name, y1, y2)
                    
                    level, text = decide_risk(cls_name, ar, proximity_hit)
                    
                    color = (0,255,0)
                    if level==1: color=(0,255,255)
                    elif level==2: color=(0,0,255)
                    thickness = 2 if level < 2 else 4

                    cv2.rectangle(frame, (x1,y1),(x2,y2),color,thickness)
                    label = f'{cls_name} {conf_score:.2f}'

                    if cls_name == 'license plate':
                        plate = None
                        try:
                            crop = frame[y1:y2, x1:x2]
                            results_plate = reader.readtext(crop)
                            if results_plate:
                                plate = results_plate[0][1]
                                label += f' | Placa: {plate}'
                        except Exception as e:
                            pass

                    cv2.putText(frame, label, (x1,max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1)

                    if level>=1: risks.append((level,text))

            now = time.time()
            fps = 1.0/(now-t0) if now!=t0 else 0
            t0 = now

            frame = draw_overlay(frame, risks, fps, lane_warning, host_speed)

            if any(r[0]>=1 for r in risks):
                beep()

            cv2.imshow('ADAS YOLOv8 - Sistema Avançado', frame)
            key = cv2.waitKey(args.video_delay) & 0xFF
            
            if key == 27:
                break
            elif key == ord('p') or key == ord('P'):
                paused = not paused
                print("⏸️ Pausado" if paused else "▶️ Despausado")
            elif key == ord('l') or key == ord('L'):
                lane_detection_enabled = not lane_detection_enabled
                print(f"🛣️ Detecção de faixa: {'ATIVADA' if lane_detection_enabled else 'DESATIVADA'}")
            elif key == ord('h') or key == ord('H'):
                print("🎮 Controles:\n- ESC: Sair\n- P: Pausar/Despausar\n- L: Toggle detecção de faixa\n- H: Mostrar esta ajuda")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Sistema ADAS finalizado")
