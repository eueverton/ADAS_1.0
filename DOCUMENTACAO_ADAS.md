# Documentação Completa do Sistema ADAS 1.0

## Visão Geral do Sistema

O Sistema ADAS (Advanced Driver Assistance System) é uma solução de assistência ao condutor que utiliza visão computacional para detectar objetos, estimar distâncias e gerar alertas de segurança em tempo real.

## Arquitetura do Sistema

### Módulos Principais

1. **adas.py** - Script principal do sistema
2. **icons.py** - Sistema de ícones personalizados
3. **config_adas.json** - Configurações do sistema
4. **requirements.txt** - Dependências do projeto

### Fluxo de Processamento

1. Captura de vídeo (webcam/arquivo/stream)
2. Pré-processamento do frame
3. Detecção de objetos com YOLOv8
4. Análise de risco e proximidade
5. Geração de alertas visuais e sonoros
6. Renderização do overlay com informações

## Configuração (config_adas.json)

```json
{
  "target_classes": ["car", "truck", "bus", "motorcycle", "stop sign", "traffic light", "person", "license plate"],
  "zones": {
    "proximity": {"x_min": 0.2, "x_max": 0.8, "y_min": 0.2, "y_max": 0.8}
  },
  "risk_config": {
    "car": {"area_ratio_high": 0.05, "area_ratio_mid": 0.08},
    "truck": {"area_ratio_high": 0.08, "area_ratio_mid": 0.01},
    "bus": {"area_ratio_high": 0.25, "area_ratio_mid": 0.12},
    "motorcycle": {"area_ratio_high": 0.1, "area_ratio_mid": 0.05},
    "person": {"area_ratio_high": 0.08, "area_ratio_mid": 0.04}
  },
  "camera_focal_length": 300,
  "alerts": {
    "collision_distance_critical": 0.7
  }
}
```

## Funções Principais

### 1. `load_config(path='config_adas.json')`
Carrega as configurações do sistema a partir do arquivo JSON.

### 2. `preprocess_frame(frame)`
Realça contraste e nitidez do frame para melhor detecção.

### 3. `estimate_distance(cls_name, y1, y2)`
Estima a distância real do objeto usando geometria de câmera.

### 4. `in_proximity_zone(x1, y1, x2, y2, W, H)`
Verifica se o objeto está na zona de risco (parabrisa).

### 5. `area_ratio(x1, y1, x2, y2, W, H)`
Calcula a proporção da área do objeto em relação ao frame.

### 6. `decide_risk(cls_name, aratio, proximity_hit)`
Decide o nível de risco baseado no tipo de objeto e proximidade.

### 7. `draw_overlay(frame, risks, fps, lane_warning, host_speed, proximity_zone_visible)`
Renderiza o overlay com informações, alertas e ícones.

### 8. `beep()`
Gera alerta sonoro usando pygame.

## Sistema de Ícones Personalizados

### Classe `ADASIcons`
Gerencia a criação e acesso aos ícones visuais.

**Métodos de criação:**
- `_create_warning_icon()` - Ícone de atenção (amarelo)
- `_create_danger_icon()` - Ícone de perigo (vermelho) 
- `_create_info_icon()` - Ícone informativo (azul)
- `_create_car_icon()` - Ícone de carro
- `_create_motorcycle_icon()` - Ícone de moto
- `_create_truck_icon()` - Ícone de caminhão
- `_create_bus_icon()` - Ícone de ônibus
- `_create_person_icon()` - Ícone de pedestre
- `_create_stop_sign_icon()` - Ícone de placa de pare
- `_create_traffic_light_icon()` - Ícone de semáforo

## Parâmetros de Execução

### Argumentos de Linha de Comando

```bash
--source          # Fonte de vídeo (0=webcam, caminho, URL)
--conf            # Confiança mínima para detecção (0.0-1.0)
--model           # Modelo YOLO a usar (padrão: yolov8n.pt)
--show-names      # Classes específicas para mostrar
--proximity-x-min # Limite esquerdo da zona de proximidade
--proximity-x-max # Limite direito da zona de proximidade  
--proximity-y-min # Limite superior da zona de proximidade
--proximity-y-max # Limite inferior da zona de proximidade
--skip-frames     # Pular frames para melhorar FPS
--lane-detection  # Ativar detecção de mudança de faixa
--video-delay     # Tempo de espera entre frames (ms)
```

### Exemplos de Uso

```bash
# Webcam com configuração padrão
python adas.py

# Vídeo arquivo com alta precisão
python adas.py --source Dashcam.mp4 --conf 0.4

# Otimizado para performance
python adas.py --skip-frames 3 --video-delay 30

# Zona de risco personalizada
python adas.py --proximity-x-min 0.1 --proximity-x-max 0.9 --proximity-y-min 0.4 --proximity-y-max 0.8
```

## Sistema de Alertas

### Níveis de Risco

1. **Nível 0**: Informativo (semáforos detectados)
2. **Nível 1**: Atenção (objetos próximos)
3. **Nível 2**: Perigo (risco de colisão iminente)

### Tipos de Alertas

- **Proximidade**: Objetos na zona de risco
- **Aproximação rápida**: Objetos se aproximando rapidamente
- **Colisão iminente**: Distância crítica atingida
- **Placas detectadas**: OCR de placas de veículos

## Estrutura de Dados

### Variáveis Globais

```python
VEHICLE_HISTORY = {}        # Histórico de distâncias por objeto
CONFIG = {}                 # Configurações carregadas
TARGET_NAMES = set()        # Classes alvo para detecção
PROXIMITY_ZONE = {}         # Zona de proximidade configurada
RISK_CFG = {}               # Configurações de risco
RECENT_RISKS = deque()      # Histórico recente de riscos
VEHICLE_LENGTH = {}         # Comprimentos médios de veículos
FOCAL_LENGTH = 300          # Distância focal da câmera
COLLISION_DISTANCE_CRITICAL = 0.7  # Distância crítica para colisão
```

## Dependências

### Bibliotecas Principais

```python
ultralytics    # YOLOv8 para detecção de objetos
opencv-python  # Processamento de imagem e vídeo
numpy          # Computação numérica
pygame         # Sistema de áudio
easyocr        # OCR para placas
```

### Instalação

```bash
pip install ultralytics opencv-python numpy pygame easyocr
```

## Performance e Otimização

### Técnicas Implementadas

1. **Skip Frames**: Processa apenas 1 a cada N frames
2. **Pré-processamento**: Realce de contraste para melhor detecção
3. **Zona de Interesse**: Foca apenas na área relevante do frame
4. **Debouncing**: Evita alertas sonoros excessivos

### Métricas de Performance

- FPS: 15-30 (dependendo da configuração)
- Latência: < 100ms
- Uso de CPU: Moderado
- Uso de GPU: Alto (se disponível)

## Troubleshooting

### Problemas Comuns

1. **Webcam não detectada**
   - Verifique permissões do sistema
   - Teste com `--source 0`

2. **Performance baixa**
   - Aumente `--skip-frames`
   - Reduza `--conf`
   - Use modelo menor (yolov8n.pt)

3. **Alertas sonoros não funcionam**
   - Verifique arquivo `beep.wav` no diretório
   - Teste com pygame instalado

4. **Detecção imprecisa**
   - Ajuste `FOCAL_LENGTH` no config_adas.json
   - Calibre `VEHICLE_LENGTH` para seu ambiente

## Próximas Melhorias

1. Sistema de logging de eventos
2. Exportação de relatórios em CSV/JSON
3. Modo noturno com ajuste automático
4. Integração com GPS e sensores
5. Dashboard web para monitoramento
6. Machine learning para previsão de trajetórias

## Contribuição

Para contribuir com o projeto:

1. Faça fork do repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

Este projeto está sob licença MIT. Veja o arquivo LICENSE para detalhes.

## Contato

Para dúvidas e sugestões:
- Email: [seu-email]
- GitHub: [seu-usuario]
- Documentação: [link-para-docs]
