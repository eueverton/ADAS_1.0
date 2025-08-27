# Sistema de Ícones ADAS

## Visão Geral
Este sistema implementa ícones personalizados para o sistema ADAS, proporcionando uma interface visual mais rica e intuitiva para os alertas de segurança.

## Ícones Disponíveis

### Ícones de Alerta
- **warning**: Triângulo amarelo com ponto de exclamação - para alertas de atenção
- **danger**: Triângulo vermelho com X - para alertas de perigo crítico
- **info**: Círculo azul com letra i - para informações gerais

### Ícones de Veículos
- **car**: Ícone de carro verde
- **motorcycle**: Ícone de motocicleta amarela
- **truck**: Ícone de caminhão laranja
- **bus**: Ícone de ônibus verde

### Ícones de Sinalização
- **person**: Ícone de pedestre branco
- **stop_sign**: Placa de pare vermelha com texto "PARE"
- **traffic_light**: Semáforo com luzes verde, amarela e vermelha

## Integração com o Sistema ADAS

Os ícones são automaticamente associados aos tipos de alertas:

- Carros, caminhões e ônibus → Ícone de carro
- Motos → Ícone de moto
- Pedestres → Ícone de pessoa
- Placas de pare → Ícone de placa de pare
- Semáforos → Ícone de semáforo
- Alertas nível 2 (perigo) → Ícone de perigo
- Alertas nível 1 (atenção) → Ícone de atenção
- Outros alertas → Ícone de informação

## Como Usar

### Instalação
Nenhuma instalação adicional é necessária. Os ícones já estão integrados ao sistema ADAS.

### Execução
Execute o sistema ADAS normalmente:
```bash
python adas.py --source Dashcam.mp4 --conf 0.4 --skip-frames 3 --video-delay 30
```

### Personalização
Para modificar os ícones, edite o arquivo `icons.py`:

1. **Tamanho dos ícones**: Modifique o parâmetro `icon_size` no construtor `ADASIcons`
2. **Cores**: Altere as cores RGB nos métodos de criação de ícones
3. **Novos ícones**: Adicione novos métodos seguindo o padrão existente

## Exemplo de Código

```python
from icons import ADASIcons

# Criar instância dos ícones
icons = ADASIcons(icon_size=32)

# Acessar um ícone específico
warning_icon = icons.icons['warning']
car_icon = icons.icons['car']
```

## Características Técnicas

- **Formato**: Imagens OpenCV (numpy arrays)
- **Canais**: RGB (3 canais de cor)
- **Tamanho padrão**: 32x32 pixels
- **Transparência**: Usa alpha blending para integração suave

## Benefícios

1. **Visual mais intuitivo**: Ícones facilitam a identificação rápida dos tipos de alerta
2. **Consistência visual**: Design uniforme em todos os alertas
3. **Personalizável**: Fácil de modificar e expandir
4. **Performance**: Ícones pré-renderizados para mínimo impacto no FPS

## Troubleshooting

### Problema: Ícones não aparecem
**Solução**: Verifique se o arquivo `icons.py` está no mesmo diretório que `adas.py`

### Problema: Ícones muito grandes/pequenos
**Solução**: Ajuste o parâmetro `icon_size` no construtor `ADASIcons`

### Problema: Cores incorretas
**Solução**: Verifique os valores RGB nos métodos de criação de ícones
