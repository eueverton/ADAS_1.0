import argparse
import time
from collections import deque
import cv2
import numpy as np

try:
    import winsound
    def beep():
        winsound.Beep(1500, 120)
except:
    try:
        from playsound import playsound
        import os
        _BEEP_PATH = 'beep.wav'
        def beep():
            if os.path.exists(_BEEP_PATH):
                playsound(_BEEP_PATH, block=False)
    except:
        def beep():
            pass

# Foco em carros e placas - objetos mais relevantes para ADAS
TARGET_NAMES = {'car', 'truck', 'bus', 'motorcycle', 'stop sign', 'traffic light', 'person'}
# Área de proximidade horizontal - cobrindo todo o parabrisa
PROXIMITY_ZONE = {'x_min': 1.01, 'x_max': 1.9, 'y_min': 0.01, 'y_max': 1.8}
# Zona de faixa - área central para detecção de mudança de faixa
LANE_ZONE = {'x_min': 0.35, 'x_max': 0.65, 'y_min': 0.4, 'y_max': 0.9}
# Configurações de risco otimizadas para carros
RISK_CFG = {
    'car': {'area_ratio_high': 0.15, 'area_ratio_mid': 0.08},
    'truck': {'area_ratio_high': 0.2, 'area_ratio_mid': 0.1},
    'bus': {'area_ratio_high': 0.25, 'area_ratio_mid': 0.12},
    'motorcycle': {'area_ratio_high': 0.1, 'area_ratio_mid': 0.05},
    'person': {'area_ratio_high': 0.08, 'area_ratio_mid': 0.04}
}
RECENT_RISKS = deque(maxlen=3)  # Reduzido para máxima responsividade
LANE_HISTORY = deque(maxlen=10)  # Histórico para detecção de mudança de faixa

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
    """Sistema de risco otimizado para carros e placas com detecção de faixa"""
    if cls_name in ('car', 'truck', 'bus'):
        if proximity_hit and aratio >= RISK_CFG.get(cls_name, RISK_CFG['car'])['area_ratio_high']:
            return 2, f'🚨 PERIGO: {cls_name.title()} MUITO próximo!'
        if proximity_hit and aratio >= RISK_CFG.get(cls_name, RISK_CFG['car'])['area_ratio_mid']:
            return 1, f'⚠️ Atenção: {cls_name.title()} à frente'
    elif cls_name == 'motorcycle':
        if proximity_hit and aratio >= RISK_CFG['motorcycle']['area_ratio_high']:
            return 2, '🚨 PERIGO: Moto MUITO próxima!'
        if proximity_hit and aratio >= RISK_CFG['motorcycle']['area_ratio_mid']:
            return 1, '⚠️ Atenção: Moto à frente'
    elif cls_name == 'stop sign' and proximity_hit:
        return 1, '🛑 PLACA DE PARE - Reduza velocidade!'
    elif cls_name == 'traffic light' and proximity_hit:
        return 0, '🚦 Semáforo detectado'
    elif cls_name == 'person' and proximity_hit:
        if aratio >= RISK_CFG['person']['area_ratio_high']:
            return 2, '🚨 PERIGO: Pedestre MUITO perto!'
        if aratio >= RISK_CFG['person']['area_ratio_mid']:
            return 1, '⚠️ Atenção: Pedestre à frente'
    
    # Detecção de mudança de faixa
    if lane_hit:
        return 1, '🛣️ Objeto na faixa detectado'
    
    return -1, ''

def draw_overlay(frame, risks, fps, lane_warning, proximity_zone_visible=True):
    """Overlay otimizado com zona de proximidade horizontal e detecção de faixa"""
    H, W = frame.shape[:2]
    
    # Desenha zona de proximidade horizontal (parabrisa)
    if proximity_zone_visible:
        rx1 = int(PROXIMITY_ZONE['x_min'] * W)
        rx2 = int(PROXIMITY_ZONE['x_max'] * W)
        ry1 = int(PROXIMITY_ZONE['y_min'] * H)
        ry2 = int(PROXIMITY_ZONE['y_max'] * H)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
        cv2.putText(frame, 'PARABRISA - ZONA DE PROXIMIDADE', (rx1, ry1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Desenha zona de faixa central
    lx1 = int(LANE_ZONE['x_min'] * W)
    lx2 = int(LANE_ZONE['x_max'] * W)
    ly1 = int(LANE_ZONE['y_min'] * H)
    ly2 = int(LANE_ZONE['y_max'] * H)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 0, 255), 1)
    cv2.putText(frame, 'ZONA DE FAIXA', (lx1, ly1-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    
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
    args = parser.parse_args()

    # Atualiza zona de proximidade com argumentos da linha de comando
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
            
            # Otimização: processa apenas a zona de interesse para melhor FPS
            results = model.predict(frame, conf=args.conf, verbose=False)[0]
            risks = []
            lane_objects = []
            
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
                    
                    ar = area_ratio(x1, y1, x2, y2, W, H)
                    proximity_hit = in_proximity_zone(x1, y1, x2, y2, W, H)
                    lane_hit = in_lane_zone(x1, y1, x2, y2, W, H)
                    
                    # Detecção de mudança de faixa
                    if lane_detection_enabled and cls_name in ('car', 'truck', 'bus') and lane_hit:
                        lane_change, warning_msg = detect_lane_change(center_x, W)
                        if lane_change:
                            lane_warning = warning_msg
                            beep()  # Alerta sonoro para mudança de faixa
                        lane_objects.append((center_x, center_y, cls_name))
                    
                    level, text = decide_risk(cls_name, ar, proximity_hit, lane_hit)
                    
                    # Cores baseadas no nível de risco
                    color = (0, 255, 0)  # Verde
                    if level == 1: color = (0, 255, 255)  # Amarelo
                    elif level == 2: color = (0, 0, 255)  # Vermelho
                    
                    # Desenha bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label otimizado
                    label = f'{cls_name} {conf_score:.2f}'
                    cv2.putText(frame, label, (x1, max(20, y1-6)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    if level >= 0:
                        risks.append((level, text))

            # Sistema de alertas otimizado
            RECENT_RISKS.append(max([r[0] for r in risks], default=-1))
            averaged = int(round(sum([x for x in RECENT_RISKS if x >= 0])/max(1, len([x for x in RECENT_RISKS if x >= 0])))) if any(x >= 0 for x in RECENT_RISKS) else -1

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
        key = cv2.waitKey(1) & 0xFF
        
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