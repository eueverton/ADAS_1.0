# Histórico de distâncias para cada objeto
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

def beep():
    """Alerta sonoro usando pygame."""
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


# Carrega parâmetros do config_adas.json
def load_config(path='config_adas.json'):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

CONFIG = load_config()
TARGET_NAMES = set(CONFIG.get('target_classes', ['car','truck','bus','motorcycle','stop sign','traffic light','person','license plate']))
PROXIMITY_ZONE = CONFIG.get('zones', {}).get('proximity', {'x_min': 0.2, 'x_max': 0.8, 'y_min': 0.2, 'y_max': 0.8})
LANE_ZONE = CONFIG.get('zones', {}).get('lane', {'x_min': 0.35, 'x_max': 0.65, 'y_min': 0.4, 'y_max': 0.9})
RISK_CFG = CONFIG.get('risk_config', {
    'car': {'area_ratio_high': 0.05, 'area_ratio_mid': 0.08},
    'truck': {'area_ratio_high': 0.08, 'area_ratio_mid': 0.01},
    'bus': {'area_ratio_high': 0.25, 'area_ratio_mid': 0.12},
    'motorcycle': {'area_ratio_high': 0.1, 'area_ratio_mid': 0.05},
    'person': {'area_ratio_high': 0.08, 'area_ratio_mid': 0.04}
})
RECENT_RISKS = deque(maxlen=3)
LANE_HISTORY = deque(maxlen=10)
VEHICLE_LENGTH = {
    'car': CONFIG.get('risk_config', {}).get('car', {}).get('length', 0.5),
    'truck': CONFIG.get('risk_config', {}).get('truck', {}).get('length', 0.5),
    'bus': CONFIG.get('risk_config', {}).get('bus', {}).get('length', 0.5),
    'motorcycle': CONFIG.get('risk_config', {}).get('motorcycle', {}).get('length', 0.5),
    'person': CONFIG.get('risk_config', {}).get('person', {}).get('length', 0.5),
}
FOCAL_LENGTH = CONFIG.get('camera_focal_length', 300)  # Valor sugerido para dashcam 720p/1080p. Ajuste conforme sua câmera.
COLLISION_DISTANCE_CRITICAL = CONFIG.get('alerts', {}).get('collision_distance_critical', 0.7)
def preprocess_frame(frame):
    """Aumenta contraste e nitidez do frame."""
    alpha = 1.5
    beta = 10
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)
    return frame

def estimate_distance(cls_name, y1, y2):
    """Estima distância real do objeto. Calibre FOCAL_LENGTH e length para maior precisão."""
    # Sistema de metragem do adas_patched.py
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
    """Verifica se objeto está na zona de proximidade horizontal (parabrisa)"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    rx1 = PROXIMITY_ZONE['x_min'] * W
    rx2 = PROXIMITY_ZONE['x_max'] * W
    ry1 = PROXIMITY_ZONE['y_min'] * H
    ry2 = PROXIMITY_ZONE['y_max'] * H
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

def in_lane_zone(x1, y1, x2, y2, W, H):
    """Verifica se objeto está na zona central da faixa"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    lx1 = LANE_ZONE['x_min'] * W
    lx2 = LANE_ZONE['x_max'] * W
    ly1 = LANE_ZONE['y_min'] * H
    ly2 = LANE_ZONE['y_max'] * H
    return lx1 <= cx <= lx2 and ly1 <= cy <= ly2

def detect_lane_change(center_x, W):
    """Detecta mudança de faixa baseada na posição horizontal"""
    normalized_x = center_x / W
    LANE_HISTORY.append(normalized_x)
    
    if len(LANE_HISTORY) < 5:
        return False, "Analisando faixa..."
    
    # Calcula tendência de movimento
    recent_positions = list(LANE_HISTORY)[-5:]
    left_trend = sum(1 for x in recent_positions if x < 0.4)
    right_trend = sum(1 for x in recent_positions if x > 0.6)
    
    if left_trend >= 3:
        return True, "⚠️ MUDANDO PARA FAIXA ESQUERDA!"
    elif right_trend >= 3:
        return True, "⚠️ MUDANDO PARA FAIXA DIREITA!"
    
    return False, "✅ Mantendo faixa"

def area_ratio(x1, y1, x2, y2, W, H):
    return max(0, x2 - x1) * max(0, y2 - y1) / float(W * H)

def decide_risk(cls_name, aratio, proximity_hit, lane_hit):
    """Sistema de risco otimizado APENAS para objetos na área do parabrisa"""
    # Apenas avalia risco se o objeto estiver na zona de proximidade (parabrisa)
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
    
    # Ignora detecção de faixa para focar apenas no parabrisa
    return -1, ''

def draw_overlay(frame, risks, fps, lane_warning, proximity_zone_visible=True):
    """Overlay otimizado APENAS para zona de parabrisa - detecção exclusiva"""
    H, W = frame.shape[:2]
    
    # Desenha área retangular do parabrisa com destaque
    rx1 = int(PROXIMITY_ZONE['x_min'] * W)
    rx2 = int(PROXIMITY_ZONE['x_max'] * W)
    ry1 = int(PROXIMITY_ZONE['y_min'] * H)
    ry2 = int(PROXIMITY_ZONE['y_max'] * H)
    
    # Destaca a zona do parabrisa com cor mais intensa
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 3)
    cv2.putText(frame, 'PARABRISA - ZONA EXCLUSIVA DE RISCO', (rx1, ry1-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Desenha área quadrada central (zona crítica) apenas para referência
    square_size = int(min(W, H) * 0.25)
    sq_cx = W // 2
    sq_cy = int(H * 0.65)
    sq1 = (sq_cx - square_size // 2, sq_cy - square_size // 2)
    sq2 = (sq_cx + square_size // 2, sq_cy + square_size // 2)
    cv2.rectangle(frame, sq1, sq2, (0, 0, 255), 2)
    cv2.putText(frame, 'ZONA CRÍTICA (referência)', (sq1[0], sq1[1]-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Informação sobre foco exclusivo no parabrisa
    cv2.putText(frame, 'SISTEMA FOCADO APENAS NO PARABRISA', (10, H-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # FPS e informações
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Alertas de risco
    msg = ''
    lvl = -1
    for level, text in risks:
        if level > lvl:
            lvl, msg = level, text
    
    if lvl >= 0 and msg:
        color = (0, 255, 255) if lvl == 1 else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (W, 45), color, -1)
        cv2.putText(frame, msg, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Aviso de mudança de faixa
    if lane_warning:
        cv2.rectangle(frame, (0, H-60), (W, H), (0, 0, 255), -1)
        cv2.putText(frame, lane_warning, (10, H-35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
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

    # Atualiza zona de proximidade apenas se os valores padrão foram alterados
    # (mantém valores do config_adas.json a menos que explicitamente sobrescritos)
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
    host_speed = None  # Velocidade estimada do carro da câmera (m/s)
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
            # Pula frames para melhorar FPS
            if frame_count % args.skip_frames != 0:
                continue
                
            H, W = frame.shape[:2]
            # Calcula zona do parabrisa
            rx1 = int(PROXIMITY_ZONE['x_min'] * W)
            rx2 = int(PROXIMITY_ZONE['x_max'] * W)
            ry1 = int(PROXIMITY_ZONE['y_min'] * H)
            ry2 = int(PROXIMITY_ZONE['y_max'] * H)
            # Calcula zona quadrada central
            square_size = int(min(W, H) * 0.25)
            sq_cx = W // 2
            sq_cy = int(H * 0.65)
            sq1 = (sq_cx - square_size // 2, sq_cy - square_size // 2)
            sq2 = (sq_cx + square_size // 2, sq_cy + square_size // 2)
            # Otimização: processa apenas a zona de interesse para melhor FPS
            results = model.predict(frame, conf=args.conf, verbose=False)[0]
            risks = []
            lane_objects = []
            nearest_distance = None
            nearest_obj_id = None
            # Identifica o veículo mais próximo à frente
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = names.get(cls_id, str(cls_id))
                    if cls_name not in allowed:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    distance = estimate_distance(cls_name, y1, y2)
                    # Centro do objeto
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    # Checa se está no parabrisa
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        # Circula com atenção amarela
                        cv2.circle(frame, (cx, cy), 30, (0, 255, 255), 3)
                    # Checa se está na zona quadrada central E está muito próximo
                    if sq1[0] <= cx <= sq2[0] and sq1[1] <= cy <= sq2[1] and distance is not None and distance <= 1.5:
                        # Alerta vermelho apenas se realmente próximo
                        cv2.circle(frame, (cx, cy), 40, (0, 0, 255), 4)
                        label = f'🚨 ALERTA: {cls_name} muito próximo! ({distance:.1f}m)'
                        risks.append((2, label))
                    # ...existing code...
            # Estima velocidade do host vehicle (carro da câmera)
            if prev_nearest_distance is not None and nearest_distance is not None and fps > 0:
                # Δdistância/frame * FPS (aproximação)
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
                    
                    # Verifica se o objeto está na zona de proximidade antes de processar
                    proximity_hit = in_proximity_zone(x1, y1, x2, y2, W, H)
                    if not proximity_hit:
                        continue  # Ignora objetos fora da zona de proximidade
                    
                    ar = area_ratio(x1, y1, x2, y2, W, H)
                    lane_hit = in_lane_zone(x1, y1, x2, y2, W, H)
                    # Estima distância real
                    distance = estimate_distance(cls_name, y1, y2)
                    
                    # Detecção de mudança de faixa
                    if lane_detection_enabled and cls_name in ('car', 'truck', 'bus') and lane_hit:
                        lane_change, warning_msg = detect_lane_change(center_x, W)
                        if lane_change:
                            lane_warning = warning_msg
                            beep()  # Alerta sonoro para mudança de faixa
                        lane_objects.append((center_x, center_y, cls_name))
                    
                    level, text = decide_risk(cls_name, ar, proximity_hit, lane_hit)
                    
                    # Sistema anti-colisão: alerta se distância crítica for atingida
                    collision_alert = False
                    label = f'{cls_name} {conf_score:.2f}'
                    if distance is not None:
                        label += f' {distance:.1f}m'
                    else:
                        label += ' --- m'
                        # Adiciona cálculo de velocidade relativa
                        relative_speed = None
                        if cls_name in ('car', 'truck', 'bus', 'motorcycle') and distance is not None and 'obj_id' in locals() and obj_id == nearest_obj_id and host_speed is not None and fps > 0:
                            prev_dist = VEHICLE_HISTORY.get(obj_id, distance)
                            speed_approach = prev_dist - distance
                            relative_speed = speed_approach * fps - (host_speed if host_speed is not None else 0)
                            # Alerta se aproximação for perigosa considerando velocidade do host
                            if relative_speed > 5.0 and distance <= COLLISION_DISTANCE_CRITICAL * 2:
                                color = (255, 0, 0)
                                thickness = 6
                                label = f'⬆️ APROXIMAÇÃO PERIGOSA! {cls_name} {distance:.1f}m | Vrel: {relative_speed:.1f}m/s'
                                risks.append((2, label))
                                overlay = frame.copy()
                                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (255, 0, 0), -1)
                                frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
                                cv2.putText(frame, f'APROXIMAÇÃO PERIGOSA: {cls_name} a {distance:.1f}m | Vrel: {relative_speed:.1f}m/s', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
                                now = time.time()
                                if now - t_last_beep > 1.0:
                                    beep()
                                    t_last_beep = now

                    if cls_name in ('car', 'truck', 'bus', 'motorcycle') and distance is not None:
                        # Histórico de distância para prever aproximação rápida
                        obj_id = f'{cls_name}_{x1}_{y1}_{x2}_{y2}'
                        prev_dist = VEHICLE_HISTORY.get(obj_id, distance)
                        speed_approach = prev_dist - distance
                        VEHICLE_HISTORY[obj_id] = distance
                        # Se aproximação rápida (>1.5m/frame), alerta extra
                        if speed_approach > 1.5 and distance <= COLLISION_DISTANCE_CRITICAL * 2:
                            color = (255, 0, 0)
                            thickness = 6
                            label = f'⬆️ APROXIMAÇÃO RÁPIDA! {cls_name} {distance:.1f}m'
                            risks.append((2, label))
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (255, 0, 0), -1)
                            frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
                            cv2.putText(frame, f'APROXIMAÇÃO RÁPIDA: {cls_name} a {distance:.1f}m', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
                            now = time.time()
                            if now - t_last_beep > 1.0:
                                beep()
                                t_last_beep = now
                        elif distance <= COLLISION_DISTANCE_CRITICAL:
                            color = (0, 0, 255)
                            thickness = 6
                            collision_alert = True
                            label = f'🚨 COLISÃO IMINENTE! {cls_name} {distance:.1f}m'
                            risks.append((2, label))
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
                            frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
                            cv2.putText(frame, f'COLISÃO IMINENTE: {cls_name} a {distance:.1f}m', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
                            now = time.time()
                            if now - t_last_beep > 1.0:
                                beep()
                                t_last_beep = now
                        elif distance <= COLLISION_DISTANCE_CRITICAL * 2:
                            color = (0, 165, 255) # Laranja
                            thickness = 4
                            label = f'⚠️ RISCO: {cls_name} {distance:.1f}m'
                            risks.append((1, label))
                            # Seta de aproximação se velocidade relativa for alta
                            if speed_approach > 1.0:
                                cv2.arrowedLine(frame, (x1, y1-20), (x1, y1-60), (0,0,255), 4, tipLength=0.5)
                        else:
                            color = (0, 255, 0)
                            thickness = 2
                    else:
                        # Realce visual para maior risco
                        color = (0, 255, 0)
                        thickness = 2
                        if level == 1:
                            color = (0, 255, 255)
                            thickness = 2
                        elif level == 2:
                            color = (0, 0, 255)
                            thickness = 4

                    # OCR para placas
                    if cls_name == 'license plate':
                        try:
                            reader = easyocr.Reader(['pt', 'en'], gpu=True)
                            crop = frame[y1:y2, x1:x2]
                            results_plate = reader.readtext(crop)
                            if results_plate:
                                plate = results_plate[0][1]
                                label += f' | Placa: {plate}'
                        except Exception as e:
                            pass

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(frame, label, (x1, max(20, y1-6)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    if not collision_alert and level >= 0:
                        risks.append((level, text))

            # Sistema de alertas otimizado
            RECENT_RISKS.append(max([r[0] for r in risks], default=-1))
            averaged = int(round(sum([x for x in RECENT_RISKS if x >= 0])/max(1, len([x for x in RECENT_RISKS if x >= 0])))) if any(x >= 0 for x in RECENT_RISKS) else -1

            # Pré-processamento antes do overlay
            frame = preprocess_frame(frame)
            frame = draw_overlay(frame, risks, fps, lane_warning)
            
            # Alerta sonoro com debounce
            now = time.time()
            if averaged >= 1 and now - t_last_beep > 1.0:
                beep()
                t_last_beep = now

            # Cálculo de FPS otimizado
            frames += 1
            if frames % 20 == 0:
                t1 = time.time()
                fps = 20.0 / (t1 - t0)
                t0 = t1

        # Interface de controle
        cv2.imshow('ADAS YOLOv8 - Sistema Avançado', frame)
        key = cv2.waitKey(args.video_delay) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('p') or key == ord('P'):  # P para pausar
            paused = not paused
            print("⏸️ Pausado" if paused else "▶️ Despausado")
        elif key == ord('l') or key == ord('L'):  # L para toggle detecção de faixa
            lane_detection_enabled = not lane_detection_enabled
            print(f"🛣️ Detecção de faixa: {'ATIVADA' if lane_detection_enabled else 'DESATIVADA'}")
        elif key == ord('h') or key == ord('H'):  # H para ajuda
            print("""
🎮 Controles:
- ESC: Sair
- P: Pausar/Despausar
- L: Toggle detecção de faixa
- H: Mostrar esta ajuda
            """)

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Sistema ADAS finalizado")