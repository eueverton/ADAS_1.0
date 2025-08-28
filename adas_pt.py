# Sistema ADAS Avançado - Reconhecimento de Placas de Trânsito e Veículos
# Versão completamente traduzida para português

# Histórico de distâncias para cada objeto
HISTORICO_VEICULOS = {}

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

# Variáveis globais para avisos de placas de trânsito
AVISOS_PLACAS_TRANSITO = {}
DURACAO_AVISO_PLACA = 5  # segundos

def bip():
    """Alerta sonoro usando pygame."""
    try:
        pygame.mixer.init()
        caminho_bip = os.path.join(os.path.dirname(__file__), 'beep.wav')
        if os.path.exists(caminho_bip):
            pygame.mixer.music.load(caminho_bip)
            pygame.mixer.music.play()
            time.sleep(0.2)
            pygame.mixer.music.stop()
    except Exception as e:
        pass

def reconhecer_placa_veiculo(frame, x1, y1, x2, y2):
    """Reconhece placas de veículos no frame fornecido usando OCR."""
    try:
        leitor = easyocr.Reader(['pt', 'en'], gpu=True)
        recorte = frame[y1:y2, x1:x2]
        
        # Pré-processamento para melhores resultados de OCR
        cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        _, binarizado = cv2.threshold(cinza, 150, 255, cv2.THRESH_BINARY_INV)
        resultados_placa = leitor.readtext(binarizado)
        
        if resultados_placa:
            placa = resultados_placa[0][1]
            return placa
    except Exception as e:
        print(f"Erro ao reconhecer placa de veículo: {e}")
    return None

def reconhecer_placa_transito(frame, x1, y1, x2, y2, nome_classe):
    """Reconhece placas de trânsito e extrai informações como limites de velocidade."""
    try:
        # Para placas de PARE e semáforos, não precisamos de OCR
        if nome_classe == 'stop sign':
            return 'PARE'
        elif nome_classe == 'traffic light':
            return 'SEMAFORO'
        
        # Para outras placas, usa OCR para detectar números (limites de velocidade)
        leitor = easyocr.Reader(['pt', 'en'], gpu=True)
        recorte = frame[y1:y2, x1:x2]
        
        # Pré-processamento aprimorado para reconhecimento de placas
        cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        suavizado = cv2.GaussianBlur(cinza, (5, 5), 0)
        _, binarizado = cv2.threshold(suavizado, 120, 255, cv2.THRESH_BINARY)
        
        resultados = leitor.readtext(binarizado)
        
        if resultados:
            texto = resultados[0][1].upper()
            
            # Procura por padrões de limite de velocidade (números como 30, 40, 50, 60, 80, 100, 120)
            import re
            correspondencia_velocidade = re.search(r'\b(30|40|50|60|80|90|100|110|120)\b', texto)
            if correspondencia_velocidade:
                velocidade = correspondencia_velocidade.group(1)
                return f'LIMITE {velocidade}km/h'
            
            # Procura por outros textos comuns em placas de trânsito
            if any(palavra in texto for palavra in ['PROIBIDO', 'PROIBIDA']):
                return 'PROIBIDO'
            elif any(palavra in texto for palavra in ['PARE', 'STOP']):
                return 'PARE'
            elif any(palavra in texto for palavra in ['PREFERENCIAL']):
                return 'PREFERENCIAL'
                
    except Exception as e:
        print(f"Erro ao reconhecer placa de trânsito: {e}")
    return None

# Carrega parâmetros do config_adas.json
def carregar_configuracao(caminho='config_adas.json'):
    if os.path.exists(caminho):
        with open(caminho, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

CONFIG = carregar_configuracao()
CLASSES_ALVO = set(CONFIG.get('target_classes', ['car','truck','bus','motorcycle','stop sign','traffic light','person','license plate']))
ZONA_PROXIMIDADE = CONFIG.get('zones', {}).get('proximity', {'x_min': 0.2, 'x_max': 0.8, 'y_min': 0.2, 'y_max': 0.8})

CONFIG_RISCO = CONFIG.get('risk_config', {
    'car': {'area_ratio_high': 0.05, 'area_ratio_mid': 0.08},
    'truck': {'area_ratio_high': 0.08, 'area_ratio_mid': 0.01},
    'bus': {'area_ratio_high': 0.25, 'area_ratio_mid': 0.12},
    'motorcycle': {'area_ratio_high': 0.1, 'area_ratio_mid': 0.05},
    'person': {'area_ratio_high': 0.08, 'area_ratio_mid': 0.04}
})

RISCO_RECENTE = deque(maxlen=3)
COMPRIMENTO_VEICULO = {
    'car': CONFIG.get('risk_config', {}).get('car', {}).get('length', 0.5),
    'truck': CONFIG.get('risk_config', {}).get('truck', {}).get('length', 0.5),
    'bus': CONFIG.get('risk_config', {}).get('bus', {}).get('length', 0.5),
    'motorcycle': CONFIG.get('risk_config', {}).get('motorcycle', {}).get('length', 0.5),
    'person': CONFIG.get('risk_config', {}).get('person', {}).get('length', 0.5),
}

DISTANCIA_FOCAL = CONFIG.get('camera_focal_length', 700)  # Valor sugerido para dashcam 720p/1080p
DISTANCIA_COLISAO_CRITICA = CONFIG.get('alerts', {}).get('collision_distance_critical', 2.0)

def preprocessar_frame(frame):
    """Aumenta contraste e nitidez do frame."""
    alpha = 1.5
    beta = 10
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)
    return frame

def estimar_distancia(nome_classe, y1, y2):
    """Estima distância real do objeto. Calibre DISTANCIA_FOCAL e comprimento para maior precisão."""
    DISTANCIA_FOCAL = CONFIG.get('camera_focal_length', 800)
    altura = y2 - y1
    if altura is None or altura <= 0:
        return 1000.0
    try:
        if nome_classe in COMPRIMENTO_VEICULO:
            comprimento_real = COMPRIMENTO_VEICULO.get(nome_classe, 1.5)
        elif nome_classe == 'person':
            comprimento_real = 1.7
        else:
            comprimento_real = 1.5
        return float((DISTANCIA_FOCAL * comprimento_real) / altura)
    except Exception:
        return 1000.0

def na_zona_proximidade(x1, y1, x2, y2, LARGURA, ALTURA):
    """Verifica se objeto está na zona de proximidade horizontal (parabrisa)"""
    centro_x = (x1 + x2) / 2
    centro_y = (y1 + y2) / 2
    rx1 = ZONA_PROXIMIDADE['x_min'] * LARGURA
    rx2 = ZONA_PROXIMIDADE['x_max'] * LARGURA
    ry1 = ZONA_PROXIMIDADE['y_min'] * ALTURA
    ry2 = ZONA_PROXIMIDADE['y_max'] * ALTURA
    return rx1 <= centro_x <= rx2 and ry1 <= centro_y <= ry2

def in_proximity_zone(x1, y1, x2, y2, W, H):
    """Verifica se objeto está na zona de proximidade horizontal (parabrisa)"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    rx1 = ZONA_PROXIMIDADE['x_min'] * W
    rx2 = ZONA_PROXIMIDADE['x_max'] * W
    ry1 = ZONA_PROXIMIDADE['y_min'] * H
    ry2 = ZONA_PROXIMIDADE['y_max'] * H
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

def razao_area(x1, y1, x2, y2, LARGURA, ALTURA):
    return max(0, x2 - x1) * max(0, y2 - y1) / float(LARGURA * ALTURA)

def decidir_risco(nome_classe, razao_area, na_proximidade):
    """Sistema de risco otimizado APENAS para objetos na área do parabrisa"""
    # Apenas avalia risco se o objeto estiver na zona de proximidade (parabrisa)
    if not na_proximidade:
        return -1, ''
    
    if nome_classe in ('car', 'truck', 'bus'):
        if razao_area >= CONFIG_RISCO.get(nome_classe, CONFIG_RISCO['car'])['area_ratio_high']:
            return 2, f'🚨 PERIGO: {nome_classe.title()} MUITO próximo no parabrisa!'
        if razao_area >= CONFIG_RISCO.get(nome_classe, CONFIG_RISCO['car'])['area_ratio_mid']:
            return 1, f'⚠️ Atenção: {nome_classe.title()} à frente no parabrisa'
    elif nome_classe == 'motorcycle':
        if razao_area >= CONFIG_RISCO['motorcycle']['area_ratio_high']:
            return 2, '🚨 PERIGO: Moto MUITO próxima no parabrisa!'
        if razao_area >= CONFIG_RISCO['motorcycle']['area_ratio_mid']:
            return 1, '⚠️ Atenção: Moto à frente no parabrisa'
    elif nome_classe == 'stop sign':
        return 1, '🛑 PLACA DE PARE no parabrisa - Reduza velocidade!'
    elif nome_classe == 'traffic light':
        return 0, '🚦 Semáforo detectado no parabrisa'
    elif nome_classe == 'person':
        if razao_area >= CONFIG_RISCO['person']['area_ratio_high']:
            return 2, '🚨 PERIGO: Pedestre MUITO perto no parabrisa!'
        if razao_area >= CONFIG_RISCO['person']['area_ratio_mid']:
            return 1, '⚠️ Atenção: Pedestre à frente no parabrisa'
    
    return -1, ''

def desenhar_overlay(frame, riscos, fps, aviso_faixa, velocidade_veiculo=None, zona_proximidade_visivel=True):
    """Overlay otimizado APENAS para zona de parabrisa - detecção exclusiva"""
    ALTURA, LARGURA = frame.shape[:2]
    
    # Inicializa sistema de ícones
    icones = ADASIcons(icon_size=32)
    
    # Desenha área retangular do parabrisa com destaque
    rx1 = int(ZONA_PROXIMIDADE['x_min'] * LARGURA)
    rx2 = int(ZONA_PROXIMIDADE['x_max'] * LARGURA)
    ry1 = int(ZONA_PROXIMIDADE['y_min'] * ALTURA)
    ry2 = int(ZONA_PROXIMIDADE['y_max'] * ALTURA)
    
    # Destaca a zona do parabrisa com cor mais intensa e transparência
    overlay = frame.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 255), -1)
    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 3)
    cv2.putText(frame, '🚗 PARABRISA - ZONA DE RISCO', (rx1, ry1-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Desenha área quadrada central (zona crítica) com destaque
    tamanho_quadrado = int(min(LARGURA, ALTURA) * 0.65)
    centro_x_quad = LARGURA // 2
    centro_y_quad = int(ALTURA * 0.55)
    quad1 = (centro_x_quad - tamanho_quadrado // 2, centro_y_quad - tamanho_quadrado // 2)
    quad2 = (centro_x_quad + tamanho_quadrado // 2, centro_y_quad + tamanho_quadrado // 2)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, quad1, quad2, (0, 0, 255), -1)
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    cv2.rectangle(frame, quad1, quad2, (0, 0, 255), 2)
    cv2.putText(frame, '⚠️ ZONA CRÍTICA', (quad1[0], quad1[1]-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Painel de informações no topo
    cv2.rectangle(frame, (0, 0), (LARGURA, 70), (0, 0, 0), -1)
    cv2.putText(frame, 'ADAS - SISTEMA AVANÇADO DE ALERTA', (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # FPS e velocidade do veículo
    cv2.putText(frame, f'FPS: {fps:.1f}', (LARGURA - 120, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Indicador de velocidade do veículo (km/h)
    if velocidade_veiculo is not None:
        velocidade_kmh = velocidade_veiculo * 3.6
        cor_velocidade = (0, 255, 0)  # Verde para velocidade normal
        if velocidade_kmh > 80:
            cor_velocidade = (0, 255, 255)  # Amarelo para velocidade moderada
        if velocidade_kmh > 100:
            cor_velocidade = (0, 0, 255)  # Vermelho para velocidade alta
            
        cv2.putText(frame, f'Velocidade: {velocidade_kmh:.1f} km/h', (LARGURA - 300, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_velocidade, 2)
    
    # Alertas de risco - sistema aprimorado com ícones
    mensagem = ''
    nivel = -1
    mensagens_alerta = []
    
    for nivel_alerta, texto in riscos:
        if nivel_alerta > nivel:
            nivel, mensagem = nivel_alerta, texto
        mensagens_alerta.append((nivel_alerta, texto))
    
    # Exibe múltiplos alertas simultaneamente com ícones
    deslocamento_y = 50
    for nivel_alerta, texto in mensagens_alerta[:3]:  # Mostra até 3 alertas
        cor = (0, 255, 255) if nivel_alerta == 1 else (0, 0, 255)
        
        # Adiciona ícone apropriado ao alerta
        icone = None
        if 'car' in texto.lower() or 'truck' in texto.lower() or 'bus' in texto.lower():
            icone = icones.icons['car']
        elif 'moto' in texto.lower():
            icone = icones.icons['motorcycle']
        elif 'pedestre' in texto.lower():
            icone = icones.icons['person']
        elif 'placa' in texto.lower() or 'pare' in texto.lower():
            icone = icones.icons['stop_sign']
        elif 'semáforo' in texto.lower():
            icone = icones.icons['traffic_light']
        elif nivel_alerta == 2:
            icone = icones.icons['danger']
        elif nivel_alerta == 1:
            icone = icones.icons['warning']
        else:
            icone = icones.icons['info']
        
        # Desenha ícone ao lado do texto
        if icone is not None:
            pos_x_icone = 10
            pos_y_icone = deslocamento_y - 25
            # Redimensiona ícone se necessário
            if icone.shape[0] != 24:
                icone = cv2.resize(icone, (24, 24))
            # Adiciona ícone ao frame
            roi = frame[pos_y_icone:pos_y_icone+24, pos_x_icone:pos_x_icone+24]
            roi = cv2.addWeighted(icone, 1.0, roi, 0.5, 0)
            frame[pos_y_icone:pos_y_icone+24, pos_x_icone:pos_x_icone+24] = roi
        
        cv2.putText(frame, texto, (40, deslocamento_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
        deslocamento_y += 25
    
    # Exibe avisos de placas de trânsito por 5 segundos
    tempo_atual = time.time()
    avisos_ativos = []
    
    # Limpa avisos expirados
    for info_placa, tempo_deteccao in list(AVISOS_PLACAS_TRANSITO.items()):
        if tempo_atual - tempo_deteccao > DURACAO_AVISO_PLACA:
            del AVISOS_PLACAS_TRANSITO[info_placa]
        else:
            avisos_ativos.append(info_placa)
    
    # Exibe avisos ativos de placas de trânsito
    if avisos_ativos:
        pos_y_aviso = ALTURA - 150
        for aviso in avisos_ativos:
            # Desenha fundo do aviso
            cv2.rectangle(frame, (LARGURA//2 - 200, pos_y_aviso - 30), (LARGURA//2 + 200, pos_y_aviso + 10), (0, 165, 255), -1)
            
            # Desenha texto do aviso
            cv2.putText(frame, f'🚸 {aviso}', (LARGURA//2 - 190, pos_y_aviso), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Desenha contador regressivo
            tempo_restante = DURACAO_AVISO_PLACA - (tempo_atual - AVISOS_PLACAS_TRANSITO[aviso])
            cv2.putText(frame, f'{tempo_restante:.1f}s', (LARGURA//2 + 150, pos_y_aviso), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            pos_y_aviso -= 40
    
    # Alerta principal com destaque e ícone grande
    if nivel >= 0 and mensagem:
        cor = (0, 255, 255) if nivel == 1 else (0, 0, 255)
        espessura = 4 if nivel == 1 else 6
        
        # Fundo para o alerta principal
        cv2.rectangle(frame, (0, ALTURA-80), (LARGURA, ALTURA), cor, -1)
        
        # Ícone grande para alerta principal
        icone_principal = icones.icons['danger'] if nivel == 2 else icones.icons['warning']
        icone_principal = cv2.resize(icone_principal, (60, 60))
        pos_x_icone = LARGURA//2 - 200
        pos_y_icone = ALTURA - 70
        roi = frame[pos_y_icone:pos_y_icone+60, pos_x_icone:pos_x_icone+60]
        roi = cv2.addWeighted(icone_principal, 1.0, roi, 0.3, 0)
        frame[pos_y_icone:pos_y_icone+60, pos_x_icone:pos_x_icone+60] = roi
        
        cv2.putText(frame, '🚨 ALERTA PRINCIPAL 🚨', (LARGURA//2 - 120, ALTURA-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, mensagem, (LARGURA//2 - 140, ALTURA-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), espessura)
    
    # Aviso de mudança de faixa (mantido para compatibilidade)
    if aviso_faixa:
        cv2.rectangle(frame, (0, ALTURA-120), (LARGURA, ALTURA-80), (0, 0, 255), -1)
        cv2.putText(frame, '⚠️ AVISO DE FAIXA', (10, ALTURA-95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, aviso_faixa, (10, ALTURA-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Status do sistema na parte inferior com ícone
    icone_status = icones.icons['info']
    icone_status = cv2.resize(icone_status, (20, 20))
    pos_x_icone = 10
    pos_y_icone = ALTURA - 20
    roi = frame[pos_y_icone-20:pos_y_icone, pos_x_icone:pos_x_icone+20]
    roi = cv2.addWeighted(icone_status, 1.0, roi, 0.5, 0)
    frame[pos_y_icone-20:pos_y_icone, pos_x_icone:pos_x_icone+20] = roi
    
    cv2.putText(frame, '✅ SISTEMA ATIVO - FOCADO NO PARABRISA', (35, ALTURA-5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sistema ADAS Avançado - Detecção de Faixa e Proximidade Horizontal')
    parser.add_argument('--fonte', default=None, help='0 (webcam), caminho de vídeo, ou URL RTSP/HTTP')
    parser.add_argument('--confianca', type=float, default=0.4, help='Confiança mínima para detecção')
    parser.add_argument('--modelo', default='yolov8n.pt', help='Modelo YOLO a usar')
    parser.add_argument('--mostrar-nomes', nargs='*', default=None, help='Classes específicas para mostrar')
    parser.add_argument('--proximidade-x-min', type=float, default=0.1, help='Limite esquerdo da zona de proximidade (0-1)')
    parser.add_argument('--proximidade-x-max', type=float, default=0.9, help='Limite direito da zona de proximidade (0-1)')
    parser.add_argument('--proximidade-y-min', type=float, default=0.3, help='Limite superior da zona de proximidade (0-1)')
    parser.add_argument('--proximidade-y-max', type=float, default=0.8, help='Limite inferior da zona de proximidade (0-1)')
    parser.add_argument('--pular-frames', type=int, default=3, help='Pular frames para melhorar FPS (recomendado: 3-5)')
    parser.add_argument('--deteccao-faixa', action='store_true', help='Ativar detecção de mudança de faixa')
    parser.add_argument('--atraso-video', type=int, default=1, help='Tempo de espera entre frames em ms (aumente para rodar mais devagar, ex: 30, 50, 100)')
    args = parser.parse_args()

    # Atualiza zona de proximidade apenas se os valores padrão foram alterados
    if args.proximidade_x_min != 0.1 or args.proximidade_x_max != 0.9 or args.proximidade_y_min != 0.3 or args.proximidade_y_max != 0.8:
        ZONA_PROXIMIDADE.update({
            'x_min': args.proximidade_x_min,
            'x_max': args.proximidade_x_max,
            'y_min': args.proximidade_y_min,
            'y_max': args.proximidade_y_max
        })

    from ultralytics import YOLO
    modelo = YOLO(args.modelo)

    fonte = 0 if args.fonte is None or args.fonte == '0' else args.fonte
    captura = cv2.VideoCapture(fonte)
    if not captura.isOpened():
        raise RuntimeError(f'Não consegui abrir a fonte: {args.fonte}')

    nomes = modelo.names
    permitidos = set(args.mostrar_nomes) if args.mostrar_nomes else CLASSES_ALVO

    ultimo_bip = 0
    fps = 0.0
    tempo_inicial = time.time()
    frames = 0
    contador_frames = 0
    aviso_faixa = ""
    velocidade_veiculo = None  # Velocidade estimada do carro da câmera (m/s)
    distancia_mais_proxima_anterior = None

    print(f"🚗 Sistema ADAS AVANÇADO iniciado com foco em: {', '.join(permitidos)}")
    print(f"📍 Zona de proximidade (parabrisa): X({ZONA_PROXIMIDADE['x_min']:.2f}-{ZONA_PROXIMIDADE['x_max']:.2f}) Y({ZONA_PROXIMIDADE['y_min']:.2f}-{ZONA_PROXIMIDADE['y_max']:.2f})")
    print(f"🛣️ Detecção de faixa: {'ATIVADA' if args.deteccao_faixa else 'DESATIVADA'}")
    print(f"⚡ Pular frames: {args.pular_frames} (FPS otimizado)")
    print("Pressione 'ESC' para sair, 'P' para pausar/despausar, 'L' para toggle detecção de faixa")

    pausado = False
    deteccao_faixa_ativa = args.deteccao_faixa

    while True:
        if not pausado:
            ok, frame = captura.read()
            if not ok:
                break
            
            contador_frames += 1
            # Pula frames para melhorar FPS
            if contador_frames % args.pular_frames != 0:
                continue
                
            ALTURA, LARGURA = frame.shape[:2]
            # Calcula zona do parabrisa
            rx1 = int(ZONA_PROXIMIDADE['x_min'] * LARGURA)
            rx2 = int(ZONA_PROXIMIDADE['x_max'] * LARGURA)
            ry1 = int(ZONA_PROXIMIDADE['y_min'] * ALTURA)
            ry2 = int(ZONA_PROXIMIDADE['y_max'] * ALTURA)
            # Calcula zona quadrada central
            tamanho_quadrado = int(min(LARGURA, ALTURA) * 0.35)  # Aumentando para 35%
            centro_x_quad = LARGURA // 2
            centro_y_quad = int(ALTURA * 0.55)
            quad1 = (centro_x_quad - tamanho_quadrado // 2, centro_y_quad - tamanho_quadrado // 2)
            quad2 = (centro_x_quad + tamanho_quadrado // 2, centro_y_quad + tamanho_quadrado // 2)
            # Otimização: processa apenas a zona de interesse para melhor FPS
            resultados = modelo.predict(frame, conf=args.confianca, verbose=False)[0]
            riscos = []
            distancia_mais_proxima = None
            id_objeto_mais_proximo = None
            
            # Identifica o veículo mais próximo à frente
            if resultados.boxes is not None and len(resultados.boxes) > 0:
                for caixa in resultados.boxes:
                    id_classe = int(caixa.cls[0])
                    nome_classe = nomes.get(id_classe, str(id_classe))
                    if nome_classe not in permitidos:
                        continue
                    x1, y1, x2, y2 = map(int, caixa.xyxy[0].tolist())
                    distancia = estimar_distancia(nome_classe, y1, y2)
                    # Centro do objeto
                    centro_x = (x1 + x2) // 2
                    centro_y = (y1 + y2) // 2
                    # Checa se está no parabrisa E está próximo (até 2 metros)
                    if rx1 <= centro_x <= rx2 and ry1 <= centro_y <= ry2 and distancia is not None and distancia <= DISTANCIA_COLISAO_CRITICA:
                        # Circula com atenção amarela para objetos próximos
                        cv2.circle(frame, (centro_x, centro_y), 30, (0, 255, 255), 3)
                    # Checa se está na zona de proximidade horizontal (parabrisa)
                    if in_proximity_zone(x1, y1, x2, y2, LARGURA, ALTURA):
                        # Circula com atenção amarela para objetos próximos
                        cv2.circle(frame, (centro_x, centro_y), 30, (0, 255, 255), 3)
                    # Checa se está na zona quadrada central E está muito próximo (até 2 metros)
                    if in_proximity_zone(x1, y1, x2, y2, LARGURA, ALTURA) and distancia is not None and distancia <= DISTANCIA_COLISAO_CRITICA:
                        # Alerta vermelho apenas se realmente próximo
                        cv2.circle(frame, (centro_x, centro_y), 40, (0, 0, 255), 4)
                        rotulo = f'🚨 ALERTA: {nome_classe} muito próximo! ({distancia:.1f}m)'
                        riscos.append((2, rotulo))
            
            # Estima velocidade do veículo host (carro da câmera)
            if distancia_mais_proxima_anterior is not None and distancia_mais_proxima is not None and fps > 0:
                # Δdistância/frame * FPS (aproximação)
                velocidade_veiculo = abs(distancia_mais_proxima_anterior - distancia_mais_proxima) * fps
            distancia_mais_proxima_anterior = distancia_mais_proxima
            
            if resultados.boxes is not None and len(resultados.boxes) > 0:
                for caixa in resultados.boxes:
                    id_classe = int(caixa.cls[0])
                    nome_classe = nomes.get(id_classe, str(id_classe))
                    if nome_classe not in permitidos:
                        continue
                    
                    x1, y1, x2, y2 = map(int, caixa.xyxy[0].tolist())
                    pontuacao_confianca = float(caixa.conf[0])
                    centro_x = (x1 + x2) / 2
                    centro_y = (y1 + y2) / 2
                    
                    # Verifica se o objeto está na zona de proximidade antes de processar
                    na_proximidade = na_zona_proximidade(x1, y1, x2, y2, LARGURA, ALTURA)
                    if not na_proximidade:
                        continue  # Ignora objetos fora da zona de proximidade
                    
                    razao = razao_area(x1, y1, x2, y2, LARGURA, ALTURA)
                    # Estima distância real
                    distancia = estimar_distancia(nome_classe, y1, y2)
                    
                    nivel, texto = decidir_risco(nome_classe, razao, na_proximidade)
                    
                    # Sistema anti-colisão: alerta se distância crítica for atingida
                    alerta_colisao = False
                    rotulo = f'{nome_classe} {pontuacao_confianca:.2f}'
                    if distancia is not None:
                        rotulo += f' {distancia:.1f}m'
                        if distancia <= DISTANCIA_COLISAO_CRITICA:
                            alerta_colisao = True  # Ativa alerta de colisão
                    else:
                        rotulo += ' --- m'
                    
                    # Reconhecimento de placa de veículo para placas detectadas
                    if nome_classe == 'license plate':
                        placa = reconhecer_placa_veiculo(frame, x1, y1, x2, y2)
                        if placa:
                            rotulo += f' | Placa: {placa}'
                    
                    # Reconhecimento de placa de trânsito para placas de PARE e semáforos
                    if nome_classe in ['stop sign', 'traffic light']:
                        info_placa = reconhecer_placa_transito(frame, x1, y1, x2, y2, nome_classe)
                        if info_placa:
                            rotulo += f' | {info_placa}'
                            # Adiciona aos avisos por 5 segundos
                            AVISOS_PLACAS_TRANSITO[info_placa] = time.time()

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, rotulo, (x1, max(20, y1-6)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    if not alerta_colisao and nivel >= 0:
                        riscos.append((nivel, texto))

            # Sistema de alertas otimizado
            RISCO_RECENTE.append(max([r[0] for r in riscos], default=-1))
            media = int(round(sum([x for x in RISCO_RECENTE if x >= 0])/max(1, len([x for x in RISCO_RECENTE if x >= 0])))) if any(x >= 0 for x in RISCO_RECENTE) else -1

            # Pré-processamento antes do overlay
            frame = preprocessar_frame(frame)
            frame = desenhar_overlay(frame, riscos, fps, aviso_faixa)
            
            # Alerta sonoro com debounce
            agora = time.time()
            if media >= 1 and agora - ultimo_bip > 1.0:
                bip()
                ultimo_bip = agora

            # Cálculo de FPS otimizado
            frames += 1
            if frames % 20 == 0:
                tempo_final = time.time()
                fps = 20.0 / (tempo_final - tempo_inicial)
                tempo_inicial = tempo_final

        # Interface de controle
        cv2.imshow('ADAS YOLOv8 - Sistema Avançado', frame)
        tecla = cv2.waitKey(args.atraso_video) & 0xFF
        
        if tecla == 27:  # ESC
            break
        elif tecla == ord('p') or tecla == ord('P'):  # P para pausar
            pausado = not pausado
            print("⏸️ Pausado" if pausado else "▶️ Despausado")
        elif tecla == ord('l') or tecla == ord('L'):  # L para toggle detecção de faixa
            deteccao_faixa_ativa = not deteccao_faixa_ativa
            print(f"🛣️ Detecção de faixa: {'ATIVADA' if deteccao_faixa_ativa else 'DESATIVADA'}")
        elif tecla == ord('h') or tecla == ord('H'):  # H para ajuda
            print("""
🎮 Controles:
- ESC: Sair
- P: Pausar/Despausar
- L: Toggle detecção de faixa
- H: Mostrar esta ajuda
            """)

    captura.release()
    cv2.destroyAllWindows()
    print("✅ Sistema ADAS finalizado")
