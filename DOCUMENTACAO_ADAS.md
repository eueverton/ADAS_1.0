# 📄 Documentação do Sistema ADAS Avançado

## 📋 Índice
1. [Visão Geral](#visão-geral)
2. [Funcionalidades](#funcionalidades)
3. [Instalação e Configuração](#instalação-e-configuração)
4. [Uso do Sistema](#uso-do-sistema)
5. [Reconhecimento de Placas](#reconhecimento-de-placas)
6. [Sistema de Alertas](#sistema-de-alertas)
7. [Configuração Avançada](#configuração-avançada)
8. [Exemplos de Uso](#exemplos-de-uso)
9. [Solução de Problemas](#solução-de-problemas)

## 🔍 Visão Geral

O **Sistema ADAS Avançado** é uma solução de Assistência ao Condutor que utiliza visão computacional para detectar e alertar sobre potenciais riscos na estrada. O sistema combina detecção de objetos com YOLOv8 e reconhecimento óptico de caracteres (OCR) para fornecer alertas em tempo real.

## 🚀 Funcionalidades

### ✅ Detecção de Veículos
- Carros, caminhões, ônibus e motocicletas
- Estimativa de distância em tempo real
- Alertas de proximidade e risco de colisão

### 🛑 Reconhecimento de Placas de Trânsito
- **Placas de PARE**: Detecção automática com aviso "PARE"
- **Semáforos**: Identificação com aviso "SEMAFORO"  
- **Limites de Velocidade**: OCR para números (30, 40, 50, 60, 80, 100, 120 km/h)
- **Placas Regulatórias**: Reconhecimento de "PROIBIDO", "PREFERENCIAL"

### 📊 Sistema de Alertas Visuais
- Overlay com informações em tempo real
- Avisos coloridos por nível de risco
- Contador regressivo de 5 segundos para placas detectadas
- Indicador de velocidade do veículo

### 🔊 Alertas Sonoros
- Bip sonoro para alertas de alto risco
- Sistema anti-spam com intervalo mínimo de 1 segundo

## 🛠️ Instalação e Configuração

### Pré-requisitos
```bash
# Instalar dependências
pip install ultralytics opencv-python pygame easyocr numpy
```

### Estrutura de Arquivos
```
ADAS_1.0/
├── adas.py              # Versão original em inglês
├── adas_pt.py           # Versão traduzida para português
├── config_adas.json     # Arquivo de configuração
├── icons.py             # Sistema de ícones
├── beep.wav             # Som de alerta (opcional)
├── test_*.py            # Testes unitários
└── DOCUMENTACAO_ADAS.md # Esta documentação
```

## 🎮 Uso do Sistema

### Comando Básico
```bash
# Usar webcam
python adas_pt.py --fonte 0

# Usar arquivo de vídeo
python adas_pt.py --fonte dashcam2.mp4

# Com configurações personalizadas
python adas_pt.py --fonte 0 --confianca 0.5 --pular-frames 2
```

### Parâmetros de Linha de Comando
| Parâmetro | Descrição | Valor Padrão |
|-----------|-----------|--------------|
| `--fonte` | Fonte de vídeo (0=webcam, caminho do arquivo) | None |
| `--confianca` | Confiança mínima para detecção | 0.4 |
| `--modelo` | Modelo YOLO a ser usado | yolov8n.pt |
| `--pular-frames` | Pular frames para melhorar FPS | 3 |
| `--atraso-video` | Delay entre frames (ms) | 1 |

### Controles Durante a Execução
- **ESC**: Sair do programa
- **P**: Pausar/Despausar
- **L**: Ativar/Desativar detecção de faixa
- **H**: Mostrar ajuda

## 🔍 Reconhecimento de Placas

### Tipos de Placas Suportadas
1. **Placas de PARE**
   - Detecção automática via YOLO
   - Aviso: "PARE" por 5 segundos

2. **Semáforos**
   - Detecção automática via YOLO  
   - Aviso: "SEMAFORO" por 5 segundos

3. **Limites de Velocidade**
   - OCR para números: 30, 40, 50, 60, 80, 90, 100, 110, 120
   - Formato: "LIMITE Xkm/h"

4. **Placas Regulatórias**
   - OCR para texto: "PROIBIDO", "PARE", "PREFERENCIAL"
   - Avisos correspondentes ao texto detectado

### Processamento de OCR
```python
# Exemplo do fluxo de reconhecimento
1. Detecção YOLO → "stop sign"
2. Função reconhecer_placa_transito() → "PARE"
3. Adiciona aos avisos ativos por 5 segundos
4. Exibe overlay com contador regressivo
```

## ⚠️ Sistema de Alertas

### Níveis de Risco
| Nível | Cor | Descrição |
|-------|-----|-----------|
| 2 | 🔴 Vermelho | Perigo iminente |
| 1 | 🟡 Amarelo | Atenção necessária |
| 0 | 🔵 Azul | Informação |

### Zonas de Detecção
- **Zona do Parabrisa**: Área retangular amarela
- **Zona Crítica**: Área quadrada central vermelha
- Apenas objetos nestas zonas geram alertas

### Alertas de Velocidade
- Verde: ≤ 80 km/h
- Amarelo: 81-100 km/h  
- Vermelho: > 100 km/h

## ⚙️ Configuração Avançada

### Arquivo config_adas.json
```json
{
  "target_classes": ["car", "truck", "bus", "motorcycle", "stop sign", "traffic light", "person", "license plate"],
  "zones": {
    "proximity": {"x_min": 0.2, "x_max": 0.8, "y_min": 0.2, "y_max": 0.8}
  },
  "risk_config": {
    "car": {"area_ratio_high": 0.05, "area_ratio_mid": 0.08, "length": 0.5},
    "truck": {"area_ratio_high": 0.08, "area_ratio_mid": 0.01, "length": 0.5},
    "bus": {"area_ratio_high": 0.25, "area_ratio_mid": 0.12, "length": 0.5},
    "motorcycle": {"area_ratio_high": 0.1, "area_ratio_mid": 0.05, "length": 0.5},
    "person": {"area_ratio_high": 0.08, "area_ratio_mid": 0.04, "length": 0.5}
  },
  "camera_focal_length": 300,
  "alerts": {
    "collision_distance_critical": 0.7
  }
}
```

### Personalização
- **target_classes**: Classes que o sistema deve detectar
- **zones**: Áreas de interesse na tela
- **risk_config**: Limiares para diferentes tipos de risco
- **camera_focal_length**: Calibrar para sua câmera específica

## 🧪 Exemplos de Uso

### Teste com Webcam
```bash
python adas_pt.py --fonte 0 --confianca 0.4 --pular-frames 3
```

### Teste com Arquivo de Vídeo  
```bash
python adas_pt.py --fonte dashcam2.mp4 --atraso-video 30
```

### Alta Performance
```bash
python adas_pt.py --fonte 0 --pular-frames 5 --confianca 0.6
```

### Baixa Latência
```bash
python adas_pt.py --fonte 0 --pular-frames 1 --confianca 0.3
```

## 🔧 Solução de Problemas

### Problemas Comuns

1. **Webcam não detectada**
   ```bash
   # Verificar se a webcam está funcionando
   python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
   ```

2. **Performance baixa**
   ```bash
   # Aumentar pulo de frames
   python adas_pt.py --pular-frames 5
   ```

3. **Muitos falsos positivos**
   ```bash
   # Aumentar confiança mínima
   python adas_pt.py --confianca 0.6
   ```

4. **OCR não funcionando**
   - Verificar se easyocr está instalado corretamente
   - Testar com imagens mais claras e bem iluminadas

### Otimização de Performance

- **CPU**: Usar `--pular-frames 3-5`
- **GPU**: Instalar CUDA para melhor performance do EasyOCR
- **Modelo**: Usar yolov8s.pt para melhor precisão

## 📊 Testes e Validação

### Testes Unitários
```bash
# Testar reconhecimento de placas de trânsito
python test_traffic_sign.py

# Testar reconhecimento de placas de veículos  
python test_license_plate.py

# Teste de integração completo
python test_adas_integration.py
```

### Validação com Dados Reais
1. Coletar vídeos de diferentes condições
2. Testar com várias placas de trânsito
3. Validar precisão do OCR
4. Ajustar parâmetros conforme necessário

## 📈 Próximos Passos

### Melhorias Futuras
- [ ] Suporte a mais tipos de placas de trânsito
- [ ] Detecção de pedestres cruzando a rua
- [ ] Sistema de alerta de sonolência do motorista
- [ ] Integração com GPS para alertas contextuais
- [ ] Modo noturno com ajustes automáticos

### Personalização
- Adicionar novos tipos de placas no código OCR
- Ajustar limiares de risco no config_adas.json
- Customizar overlay visual conforme necessidade

---

## 📞 Suporte

Para dúvidas ou problemas:
1. Verificar a documentação acima
2. Testar com os exemplos fornecidos
3. Ajustar parâmetros conforme seu hardware
4. Verificar logs de erro no console

**⚠️ Nota**: Este é um sistema de assistência e não substitui a atenção do motorista. Use sempre com cautela e conforme as leis de trânsito locais.
