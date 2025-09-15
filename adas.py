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
from icons import ADASIcons

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

def recognize_license_plate(frame, x1, y1, x2, y2):
    """Recognizes license plates in the given frame with enhanced HD preprocessing."""
    try:
        reader = easyocr.Reader(['pt', 'en'], gpu=True)
        crop = frame[y1:y2, x1:x2]

        # Enhanced preprocessing for HD video
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # CLAHE for better contrast in HD
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Bilateral filter for noise reduction while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Resize for better OCR (configurable)
        resize_factor = CONFIG.get('hd_settings', {}).get('ocr_resize_factor', 2.0)
        if resize_factor != 1.0:
            h, w = gray.shape
            gray = cv2.resize(gray, (int(w * resize_factor), int(h * resize_factor)), interpolation=cv2.INTER_CUBIC)

        # Adaptive thresholding for better text extraction
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        results_plate = reader.readtext(thresh)

        if results_plate:
            # Filter results by confidence
            min_conf = CONFIG.get('ocr', {}).get('min_confidence', 0.5)
            filtered_results = [res for res in results_plate if res[2] > min_conf]
            if filtered_results:
                plate = filtered_results[0][1].upper().replace(' ', '')
                return plate
    except Exception as e:
        print(f"Error recognizing license plate: {e}")
    return None

def detect_traffic_light_state(frame, x1, y1, x2, y2):
    """Detects the state of a traffic light (red, yellow, green)."""
    try:
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Define color ranges for traffic lights
        red_lower1 = np.array([0, 70, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 70, 50])
        red_upper2 = np.array([180, 255, 255])

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])

        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])

        # Create masks
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_green = cv2.inRange(hsv, green_lower, green_upper)

        # Count pixels for each color
        red_pixels = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        green_pixels = cv2.countNonZero(mask_green)

        total_pixels = red_pixels + yellow_pixels + green_pixels
        if total_pixels == 0:
            return None

        # Determine dominant color
        thresholds = CONFIG.get('traffic_light', {})
        red_threshold = thresholds.get('red_threshold', 0.4)
        yellow_threshold = thresholds.get('yellow_threshold', 0.3)
        green_threshold = thresholds.get('green_threshold', 0.4)

        if red_pixels / total_pixels > red_threshold:
            return 'VERMELHO'
        elif yellow_pixels / total_pixels > yellow_threshold:
            return 'AMARELO'
        elif green_pixels / total_pixels > green_threshold:
            return 'VERDE'

    except Exception as e:
        print(f"Error detecting traffic light state: {e}")
    return None

def recognize_traffic_sign(frame, x1, y1, x2, y2, cls_name):
    """Recognizes traffic signs and extracts information like speed limits with enhanced HD preprocessing."""
    try:
        # For stop signs and traffic lights, we don't need OCR
        if cls_name == 'stop sign':
            return 'PARE'
        elif cls_name == 'traffic light':
            # Try state detection first
            if CONFIG.get('traffic_light', {}).get('state_detection_enabled', False):
                state = detect_traffic_light_state(frame, x1, y1, x2, y2)
                if state:
                    return f'SEMAFORO {state}'
            return 'SEMAFORO'

        # For other signs, use OCR to detect numbers (speed limits)
        reader = easyocr.Reader(['pt', 'en'], gpu=True)
        crop = frame[y1:y2, x1:x2]

        # Enhanced preprocessing for HD sign recognition
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # CLAHE for better contrast in HD
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Bilateral filter for noise reduction while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Resize for better OCR (configurable)
        resize_factor = CONFIG.get('hd_settings', {}).get('ocr_resize_factor', 2.0)
        if resize_factor != 1.0:
            h, w = gray.shape
            gray = cv2.resize(gray, (int(w * resize_factor), int(h * resize_factor)), interpolation=cv2.INTER_CUBIC)

        # Adaptive thresholding for better text extraction
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        results = reader.readtext(thresh)

        if results:
            # Filter results by confidence
            min_conf = CONFIG.get('ocr', {}).get('min_confidence', 0.5)
            filtered_results = [res for res in results if res[2] > min_conf]
            if filtered_results:
                text = filtered_results[0][1].upper()

                # Look for speed limit patterns (numbers like 30, 40, 50, 60, 80, 100, 120)
                import re
                speed_match = re.search(r'\b(30|40|50|60|80|90|100|110|120)\b', text)
                if speed_match:
                    speed = speed_match.group(1)
                    return f'LIMITE {speed}km/h'

                # Look for other common traffic sign text
                if any(word in text for word in ['PROIBIDO', 'PROIBIDA']):
                    return 'PROIBIDO'
                elif any(word in text for word in ['PARE', 'STOP']):
                    return 'PARE'
                elif any(word in text for word in ['PREFERENCIAL']):
                    return 'PREFERENCIAL'
                elif any(word in text for word in ['CEDA', 'YIELD']):
                    return 'CEDA A PASSAGEM'
                elif any(word in text for word in ['SENTIDO', 'UNICO']):
                    return 'SENTIDO UNICO'

    except Exception as e:
        print(f"Error recognizing traffic sign: {e}")
    return None

# Global variables for traffic sign warnings
TRAFFIC_SIGN_WARNINGS = {}
TRAFFIC_SIGN_DURATION = 5  # seconds

# Global variables for temporary displays
LICENSE_PLATE_DISPLAYS = {}  # {plate_text: detection_time}
PEDESTRIAN_POSITIONS = {}  # Track pedestrian positions for movement detection


# Carrega parâmetros do config_adas.json
def load_config(path='config_adas.json'):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# Carrega configuração do fluxo óptico
def load_optical_flow_config(path='optical_flow_config.json'):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

CONFIG = load_config()
TARGET_NAMES = set(CONFIG.get('target_classes', ['car','truck','bus','motorcycle','stop sign','traffic light','person','license plate']))
PROXIMITY_ZONE = CONFIG.get('zones', {}).get('proximity', {'x_min': 0.2, 'x_max': 0.8, 'y_min': 0.2, 'y_max': 0.8})

RISK_CFG = CONFIG.get('risk_config', {
    'car': {'area_ratio_high': 0.05, 'area_ratio_mid': 0.08},
    'truck': {'area_ratio_high': 0.25, 'area_ratio_mid': 0.12},
    'bus': {'area_ratio_high': 0.25, 'area_ratio_mid': 0.12},
    'motorcycle': {'area_ratio_high': 0.1, 'area_ratio_mid': 0.05},
    'person': {'area_ratio_high': 0.08, 'area_ratio_mid': 0.04}
})
RECENT_RISKS = deque(maxlen=3)
VEHICLE_LENGTH = {
    'car': CONFIG.get('risk_config', {}).get('car', {}).get('length', 0.8),
    'truck': CONFIG.get('risk_config', {}).get('truck', {}).get('length', 0.9),
    'bus': CONFIG.get('risk_config', {}).get('bus', {}).get('length', 0.9),
    'motorcycle': CONFIG.get('risk_config', {}).get('motorcycle', {}).get('length', 0.5),
    'person': CONFIG.get('risk_config', {}).get('person', {}).get('length', 0.5),
}
FOCAL_LENGTH = CONFIG.get('hd_settings', {}).get('focal_length', 600)  # Ajustado para HD (300-800)
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


# Função de detecção de mudança de faixa removida

def area_ratio(x1, y1, x2, y2, W, H):
    return max(0, x2 - x1) * max(0, y2 - y1) / float(W * H)

def decide_risk(cls_name, aratio, proximity_hit):
    """Sistema de risco otimizado APENAS para objetos na área do parabrisa"""
    # Apenas avalia risco se o objeto estiver na zona de proximidade (parabrisa)
    if not proximity_hit:
        return -1, ''
    
    if cls_name in ('car', 'truck', 'bus'):
        if aratio >= RISK_CFG.get(cls_name, RISK_CFG['car'])['area_ratio_high']:
            return 2, f' PERIGO: {cls_name.title()} MUITO proximo no parabrisa!'
        if aratio >= RISK_CFG.get(cls_name, RISK_CFG['car'])['area_ratio_mid']:
            return 1, f' Atencao: {cls_name.title()} a frente no parabrisa'
    elif cls_name == 'motorcycle':
        if aratio >= RISK_CFG['motorcycle']['area_ratio_high']:
            return 2, ' PERIGO: Moto MUITO proxima no parabrisa!'
        if aratio >= RISK_CFG['motorcycle']['area_ratio_mid']:
            return 1, ' Atencao: Moto a frente no parabrisa'
    elif cls_name == 'stop sign':
        return 1, ' PLACA DE PARE no parabrisa - Reduza velocidade!'
    elif cls_name == 'traffic light':
        return 0, ' Semaforo detectado no parabrisa'
    elif cls_name == 'person':
        if aratio >= RISK_CFG['person']['area_ratio_high']:
            return 2, ' PERIGO: Pedestre MUITO perto no parabrisa!'
        if aratio >= RISK_CFG['person']['area_ratio_mid']:
            return 1, ' Atencao: Pedestre a frente no parabrisa'
    
    return -1, ''


def draw_clean_overlay(frame, risks, fps, lane_warning, host_speed=None, proximity_zone_visible=True, detection_stats=None):
    """Interface clean e moderna para avisos do ADAS"""
    H, W = frame.shape[:2]
    
    # Inicializa sistema de ícones
    icons = ADASIcons(icon_size=24)
    
    # Painel de status superior (transparente)
    status_panel = frame.copy()
    cv2.rectangle(status_panel, (0, 0), (W, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(status_panel, 0.6, frame, 0.4, 0)
    
    # Título do sistema
    cv2.putText(frame, 'ADAS', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f'FPS: {fps:.1f}', (W - 100, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Estatísticas de detecção
    if detection_stats:
        stats_text = f"Obj: {detection_stats.get('total_objects', 0)}"
        cv2.putText(frame, stats_text, (W - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Velocidade do veículo (desativada)
    
    # Processa alertas por prioridade
    alert_levels = {2: [], 1: [], 0: []}
    collision_alerts = []
    other_critical_alerts = []
    
    for level, text in risks:
        if level in alert_levels:
            alert_levels[level].append(text)
            # Separa alertas de colisão iminente
            if level == 2 and 'COLISAO IMINENTE' in text:
                collision_alerts.append(text)
            elif level == 2:
                other_critical_alerts.append(text)
    
    # Exibe alertas de colisão iminente - banner vermelho na parte inferior
    if collision_alerts:
        # Banner vermelho na parte inferior
        collision_bg = frame.copy()
        cv2.rectangle(collision_bg, (0, H-80), (W, H), (0, 0, 200), -1)
        frame = cv2.addWeighted(collision_bg, 0.7, frame, 0.3, 0)
        
        # Ícone de perigo grande
        danger_icon = icons.icons['danger']
        danger_icon = cv2.resize(danger_icon, (50, 50))
        frame[H-65:H-15, 20:70] = danger_icon
        
        # Texto do alerta de colisão
        collision_text = collision_alerts[0]  # Mostra apenas o primeiro alerta de colisão
        simplified_text = _simplify_alert_text(collision_text)
        cv2.putText(frame, ' COLISAO IMINENTE', (90, H-45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, simplified_text, (90, H-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Exibe outros alertas críticos (nível 2) - acima do banner de colisão
    other_critical_alerts = other_critical_alerts[:2]  # Máximo 2 alertas
    if other_critical_alerts and not collision_alerts:
        y_pos = H - 120
        for alert in other_critical_alerts:
            # Fundo semi-transparente vermelho
            alert_bg = frame.copy()
            cv2.rectangle(alert_bg, (0, y_pos), (W, y_pos + 40), (0, 0, 180), -1)
            frame = cv2.addWeighted(alert_bg, 0.5, frame, 0.5, 0)
            
            # Ícone de perigo
            danger_icon = icons.icons['danger']
            danger_icon = cv2.resize(danger_icon, (30, 30))
            frame[10:40, y_pos+5:y_pos+35] = danger_icon
            
            # Texto do alerta (simplificado)
            simplified_text = _simplify_alert_text(alert)
            cv2.putText(frame, simplified_text, (50, y_pos + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y_pos -= 45
    
    # Exibe alertas de atenção (nível 1) - lado direito
    attention_alerts = alert_levels[1][:3]
    if attention_alerts:
        x_pos = W - 300
        y_pos = 70
        for alert in attention_alerts:
            # Card flutuante amarelo
            card_bg = frame.copy()
            cv2.rectangle(card_bg, (x_pos, y_pos), (x_pos + 280, y_pos + 35), (0, 165, 255), -1)
            frame = cv2.addWeighted(card_bg, 0.4, frame, 0.6, 0)
            cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 280, y_pos + 35), (0, 165, 255), 2)
            
            # Ícone de atenção
            warning_icon = icons.icons['warning']
            warning_icon = cv2.resize(warning_icon, (25, 25))
            frame[y_pos+5:y_pos+30, x_pos+5:x_pos+30] = warning_icon
            
            # Texto simplificado
            simplified_text = _simplify_alert_text(alert)
            cv2.putText(frame, simplified_text, (x_pos + 35, y_pos + 23), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_pos += 40
    
    # Avisos de trânsito ativos
    current_time = time.time()
    active_warnings = []
    
    # Limpa avisos expirados
    for sign_info, detection_time in list(TRAFFIC_SIGN_WARNINGS.items()):
        if current_time - detection_time > TRAFFIC_SIGN_DURATION:
            del TRAFFIC_SIGN_WARNINGS[sign_info]
        else:
            active_warnings.append(sign_info)
    
    # Exibe avisos de trânsito ativos
    if active_warnings:
        x_pos = 10
        y_pos = H - 180
        for warning in active_warnings:
            # Card informativo azul
            info_bg = frame.copy()
            cv2.rectangle(info_bg, (x_pos, y_pos), (x_pos + 250, y_pos + 30), (255, 165, 0), -1)
            frame = cv2.addWeighted(info_bg, 0.4, frame, 0.6, 0)
            cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 250, y_pos + 30), (255, 165, 0), 2)
            
            # Ícone baseado no tipo de aviso
            warning_icon = _get_traffic_sign_icon(warning)
            warning_icon = cv2.resize(warning_icon, (20, 20))
            frame[y_pos+5:y_pos+25, x_pos+5:x_pos+25] = warning_icon
            
            # Texto e temporizador
            time_left = TRAFFIC_SIGN_DURATION - (current_time - TRAFFIC_SIGN_WARNINGS[warning])
            display_text = f"{warning} ({time_left:.0f}s)"
            cv2.putText(frame, display_text, (x_pos + 30, y_pos + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_pos -= 35
    
    # Display license plates temporarily
    current_time = time.time()
    active_plates = []

    # Clean up expired license plates
    for plate, detection_time in list(LICENSE_PLATE_DISPLAYS.items()):
        if current_time - detection_time > 10:  # 10 seconds display
            del LICENSE_PLATE_DISPLAYS[plate]
        else:
            active_plates.append(plate)

    # Display active license plates
    if active_plates:
        x_pos = W - 300
        y_pos = H - 50
        for plate in active_plates[:3]:  # Show max 3 plates
            # License plate card
            plate_bg = frame.copy()
            cv2.rectangle(plate_bg, (x_pos, y_pos), (x_pos + 280, y_pos + 30), (128, 128, 128), -1)
            frame = cv2.addWeighted(plate_bg, 0.6, frame, 0.4, 0)
            cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 280, y_pos + 30), (128, 128, 128), 2)

            # License plate icon
            plate_icon = icons.icons['info']  # Using info icon for license plates
            plate_icon = cv2.resize(plate_icon, (20, 20))
            frame[y_pos+5:y_pos+25, x_pos+5:x_pos+25] = plate_icon

            # Plate text and timer
            time_left = 10 - (current_time - LICENSE_PLATE_DISPLAYS[plate])
            display_text = f"Placa: {plate} ({time_left:.0f}s)"
            cv2.putText(frame, display_text, (x_pos + 30, y_pos + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y_pos -= 35

    # Status do sistema (discreto)
    cv2.putText(frame, ' Sistema Ativo', (10, H - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

    return frame

def _simplify_alert_text(text):
    """Simplifica textos de alerta para interface mais clean"""
    simplifications = {
        'PERIGO:': '⚠️',
        'MUITO próximo': 'Próximo',
        'no parabrisa': '',
        'Atenção:': '👁️',
        'Reduza velocidade': '↘️ Velocidade',
        'detectado': ''
    }
    
    simplified = text
    for old, new in simplifications.items():
        simplified = simplified.replace(old, new)
    
    # Remove palavras redundantes
    words_to_remove = ['muito', 'bastante', 'extremamente']
    for word in words_to_remove:
        simplified = simplified.replace(word, '')
    
    return simplified.strip()

def _get_traffic_sign_icon(warning_text):
    """Retorna ícone apropriado para aviso de trânsito"""
    icons = ADASIcons(icon_size=24).icons
    
    if 'LIMITE' in warning_text:
        return icons['warning']
    elif 'PARE' in warning_text:
        return icons['stop_sign']
    elif 'SEMAFORO' in warning_text:
        return icons['traffic_light']
    elif 'PROIBIDO' in warning_text:
        return icons['danger']
    else:
        return icons['info']



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
    print(f"⚡ Skip frames: {args.skip_frames} (FPS otimizado)")
    print("Pressione 'ESC' para sair, 'P' para pausar/despausar, 'L' para toggle detecção de faixa")

    paused = False

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
            sq_cy = int(H * 0.55)  # Aumentado de 0.65 para 0.55 para subir o quadrado
            sq1 = (sq_cx - square_size // 2, sq_cy - square_size // 2)
            sq2 = (sq_cx + square_size // 2, sq_cy + square_size // 2)
            # Otimização: processa apenas a zona de interesse para melhor FPS
            results = model.predict(frame, conf=args.conf, verbose=False)[0]
            risks = []
            nearest_distance = None
            nearest_obj_id = None
            # Identifica o veículo mais próximo à frente
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = names.get(cls_id, str(cls_id))
                    if cls_name not in allowed or cls_name not in ['car', 'truck', 'bus', 'motorcycle']:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    distance = estimate_distance(cls_name, y1, y2)
                    
                    # Atualiza o veículo mais próximo
                    if distance is not None and (nearest_distance is None or distance < nearest_distance):
                        nearest_distance = distance
                        nearest_obj_id = f'{cls_name}_{x1}_{y1}_{x2}_{y2}'
                    
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
            
            # Velocidade do host desativada
            host_speed = None
            
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
                    # Para pedestres, permite detecção em toda a tela
                    if cls_name == 'person':
                        proximity_hit = True  # Pedestres são detectados em toda a tela
                    else:
                        proximity_hit = in_proximity_zone(x1, y1, x2, y2, W, H)
                        if not proximity_hit:
                            continue  # Ignora objetos fora da zona de proximidade
                    
                    ar = area_ratio(x1, y1, x2, y2, W, H)
                    # Estima distância real
                    distance = estimate_distance(cls_name, y1, y2)
                    
                    # Detecção de mudança de faixa removida
                    
                    level, text = decide_risk(cls_name, ar, proximity_hit)
                    
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
                                label = f' APROXIMACAO PERIGOSA! {cls_name} {distance:.1f}m | Vrel: {relative_speed:.1f}m/s'
                                risks.append((2, label))
                                overlay = frame.copy()
                                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (255, 0, 0), -1)
                                frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
                                cv2.putText(frame, f'APROXIMACAO PERIGOSA: {cls_name} a {distance:.1f}m | Vrel: {relative_speed:.1f}m/s', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
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
                            label = f'⬆ APROXIMACAO RÁPIDA! {cls_name} {distance:.1f}m'
                            risks.append((2, label))
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (255, 0, 0), -1)
                            frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
                            cv2.putText(frame, f'APROXIMACAO RÁPIDA: {cls_name} a {distance:.1f}m', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
                            now = time.time()
                            if now - t_last_beep > 1.0:
                                beep()
                                t_last_beep = now
                        elif distance <= COLLISION_DISTANCE_CRITICAL:
                            color = (0, 0, 255)
                            thickness = 6
                            collision_alert = True
                            label = f' COLISAO IMINENTE! {cls_name} {distance:.1f}m'
                            risks.append((2, label))
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
                            frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
                            cv2.putText(frame, f'COLISAO IMINENTE: {cls_name} a {distance:.1f}m', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
                            now = time.time()
                            if now - t_last_beep > 1.0:
                                beep()
                                t_last_beep = now
                        elif distance <= COLLISION_DISTANCE_CRITICAL * 2:
                            color = (0, 165, 255) # Laranja
                            thickness = 4
                            label = f' RISCO: {cls_name} {distance:.1f}m'
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

                    # License plate recognition for detected license plates
                    if cls_name == 'license plate':
                        plate = recognize_license_plate(frame, x1, y1, x2, y2)
                        if plate:
                            label += f' | Placa: {plate}'
                            # Add to temporary display for 10 seconds
                            LICENSE_PLATE_DISPLAYS[plate] = time.time()
                    
                    # Traffic sign recognition for stop signs and traffic lights
                    if cls_name in ['stop sign', 'traffic light']:
                        sign_info = recognize_traffic_sign(frame, x1, y1, x2, y2, cls_name)
                        if sign_info:
                            label += f' | {sign_info}'
                            # Add to warnings for 5 seconds
                            TRAFFIC_SIGN_WARNINGS[sign_info] = time.time()

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(frame, label, (x1, max(20, y1-6)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    if not collision_alert and level >= 0:
                        risks.append((level, text))

            # Sistema de alertas otimizado
            RECENT_RISKS.append(max([r[0] for r in risks], default=-1))
            averaged = int(round(sum([x for x in RECENT_RISKS if x >= 0])/max(1, len([x for x in RECENT_RISKS if x >= 0])))) if any(x >= 0 for x in RECENT_RISKS) else -1

            # Coleta estatísticas de detecção
            detection_stats = {
                'total_objects': len(results.boxes) if results.boxes is not None else 0,
                'risks_count': len(risks),
                'critical_risks': sum(1 for r in risks if r[0] == 2)
            }

            # Pré-processamento antes do overlay
            frame = preprocess_frame(frame)
            frame = draw_clean_overlay(frame, risks, fps, lane_warning, host_speed, detection_stats)
            
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
        cv2.imshow('ADAS YOLOv8 - Sistema Avancado', frame)
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
