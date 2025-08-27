"""
Sistema de ícones para o ADAS - Placas de sinalização e ícones de atenção
"""

import cv2
import numpy as np
import os

class ADASIcons:
    """Classe para gerenciar ícones do sistema ADAS"""
    
    def __init__(self, icon_size=32):
        self.icon_size = icon_size
        self.icons = self._create_icons()
    
    def _create_icons(self):
        """Cria os ícones como imagens OpenCV"""
        icons_dict = {}
        
        # Ícone de atenção (triângulo amarelo)
        icons_dict['warning'] = self._create_warning_icon()
        
        # Ícone de perigo (triângulo vermelho)
        icons_dict['danger'] = self._create_danger_icon()
        
        # Ícone de informação (círculo azul)
        icons_dict['info'] = self._create_info_icon()
        
        # Ícone de carro
        icons_dict['car'] = self._create_car_icon()
        
        # Ícone de pedestre
        icons_dict['person'] = self._create_person_icon()
        
        # Ícone de moto
        icons_dict['motorcycle'] = self._create_motorcycle_icon()
        
        # Ícone de caminhão
        icons_dict['truck'] = self._create_truck_icon()
        
        # Ícone de ônibus
        icons_dict['bus'] = self._create_bus_icon()
        
        # Ícone de placa de pare
        icons_dict['stop_sign'] = self._create_stop_sign_icon()
        
        # Ícone de semáforo
        icons_dict['traffic_light'] = self._create_traffic_light_icon()
        
        return icons_dict
    
    def _create_warning_icon(self):
        """Cria ícone de atenção (triângulo amarelo com ponto de exclamação)"""
        size = self.icon_size
        icon = np.zeros((size, size, 3), dtype=np.uint8)
        icon[:] = (0, 255, 255)  # Fundo amarelo
        
        # Desenha triângulo
        points = np.array([[size//2, 0], [0, size], [size, size]], np.int32)
        cv2.fillPoly(icon, [points], (0, 255, 255))
        
        # Ponto de exclamação
        cv2.circle(icon, (size//2, size//2 + 10), 3, (0, 0, 0), -1)
        cv2.rectangle(icon, (size//2 - 1, size//2 - 10), (size//2 + 1, size//2), (0, 0, 0), -1)
        
        return icon
    
    def _create_danger_icon(self):
        """Cria ícone de perigo (triângulo vermelho com X)"""
        size = self.icon_size
        icon = np.zeros((size, size, 3), dtype=np.uint8)
        icon[:] = (0, 0, 255)  # Fundo vermelho
        
        # Desenha triângulo
        points = np.array([[size//2, 0], [0, size], [size, size]], np.int32)
        cv2.fillPoly(icon, [points], (0, 0, 255))
        
        # Desenha X
        cv2.line(icon, (5, 5), (size-5, size-5), (255, 255, 255), 2)
        cv2.line(icon, (size-5, 5), (5, size-5), (255, 255, 255), 2)
        
        return icon
    
    def _create_info_icon(self):
        """Cria ícone de informação (círculo azul com i)"""
        size = self.icon_size
        icon = np.zeros((size, size, 3), dtype=np.uint8)
        icon[:] = (255, 165, 0)  # Fundo laranja
        
        # Desenha círculo
        center = (size // 2, size // 2)
        cv2.circle(icon, center, size//2-2, (255, 255, 255), 2)
        
        # Letra i
        cv2.line(icon, (center[0], center[1]-4), (center[0], center[1]+4), (255, 255, 255), 2)
        cv2.circle(icon, (center[0], center[1]+6), 2, (255, 255, 255), -1)
        
        return icon
    
    def _create_car_icon(self):
        """Cria ícone de carro"""
        size = self.icon_size
        icon = np.zeros((size, size, 3), dtype=np.uint8)
        icon[:] = (0, 0, 0)  # Fundo preto
        
        # Corpo do carro
        cv2.rectangle(icon, (5, 10), (size-5, size-5), (0, 255, 0), -1)
        # Janelas
        cv2.rectangle(icon, (8, 5), (size-8, 12), (200, 200, 200), -1)
        # Rodas
        cv2.circle(icon, (10, size-3), 3, (50, 50, 50), -1)
        cv2.circle(icon, (size-10, size-3), 3, (50, 50, 50), -1)
        
        return icon
    
    def _create_person_icon(self):
        """Cria ícone de pedestre"""
        size = self.icon_size
        icon = np.zeros((size, size, 3), dtype=np.uint8)
        icon[:] = (0, 0, 0)  # Fundo preto
        
        # Cabeça
        cv2.circle(icon, (size//2, 8), 5, (255, 255, 255), -1)
        # Corpo
        cv2.line(icon, (size//2, 13), (size//2, size-10), (255, 255, 255), 2)
        # Braços
        cv2.line(icon, (size//2, 20), (size//2-8, 25), (255, 255, 255), 2)
        cv2.line(icon, (size//2, 20), (size//2+8, 25), (255, 255, 255), 2)
        # Pernas
        cv2.line(icon, (size//2, size-10), (size//2-6, size-5), (255, 255, 255), 2)
        cv2.line(icon, (size//2, size-10), (size//2+6, size-5), (255, 255, 255), 2)
        
        return icon
    
    def _create_motorcycle_icon(self):
        """Cria ícone de moto"""
        size = self.icon_size
        icon = np.zeros((size, size, 3), dtype=np.uint8)
        icon[:] = (0, 0, 0)  # Fundo preto
        
        # Quadro da moto
        cv2.line(icon, (5, size-5), (size-5, size-5), (0, 255, 255), 2)
        cv2.line(icon, (size-10, size-5), (size-5, 10), (0, 255, 255), 2)
        # Rodas
        cv2.circle(icon, (8, size-5), 4, (100, 100, 100), -1)
        cv2.circle(icon, (size-8, size-5), 4, (100, 100, 100), -1)
        # Guidão
        cv2.line(icon, (size-5, 10), (size-15, 5), (0, 255, 255), 2)
        
        return icon
    
    def _create_truck_icon(self):
        """Cria ícone de caminhão"""
        size = self.icon_size
        icon = np.zeros((size, size, 3), dtype=np.uint8)
        icon[:] = (0, 0, 0)  # Fundo preto
        
        # Cabine
        cv2.rectangle(icon, (5, 8), (size//2, size-5), (255, 165, 0), -1)
        # Carroceria
        cv2.rectangle(icon, (size//2, 8), (size-5, size-5), (0, 0, 255), -1)
        # Rodas
        cv2.circle(icon, (12, size-3), 3, (50, 50, 50), -1)
        cv2.circle(icon, (size-12, size-3), 3, (50, 50, 50), -1)
        
        return icon
    
    def _create_bus_icon(self):
        """Cria ícone de ônibus"""
        size = self.icon_size
        icon = np.zeros((size, size, 3), dtype=np.uint8)
        icon[:] = (0, 0, 0)  # Fundo preto
        
        # Corpo do ônibus
        cv2.rectangle(icon, (5, 5), (size-5, size-5), (0, 255, 0), -1)
        # Janelas
        cv2.rectangle(icon, (8, 8), (size-8, 15), (200, 200, 200), -1)
        # Rodas
        cv2.circle(icon, (10, size-3), 3, (50, 50, 50), -1)
        cv2.circle(icon, (size-10, size-3), 3, (50, 50, 50), -1)
        
        return icon
    
    def _create_stop_sign_icon(self):
        """Cria ícone de placa de pare"""
        size = self.icon_size
        icon = np.zeros((size, size, 3), dtype=np.uint8)
        icon[:] = (0, 0, 255)  # Fundo vermelho
        
        # Desenha octógono
        points = np.array([[size//2, 0], [0, size//2], [size//2, size], [size, size//2]], np.int32)
        cv2.fillPoly(icon, [points], (0, 0, 255))
        
        # Texto "PARE"
        cv2.putText(icon, 'PARE', (size//4, size//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return icon
    
    def _create_traffic_light_icon(self):
        """Cria ícone de semáforo"""
        size = self.icon_size
        icon = np.zeros((size, size, 3), dtype=np.uint8)
        icon[:] = (0, 0, 0)  # Fundo preto
        
        # Desenha semáforo
        cv2.rectangle(icon, (size//2 - 5, 5), (size//2 + 5, size-5), (255, 255, 255), -1)
        cv2.circle(icon, (size//2, 15), 5, (0, 255, 0), -1)  # Verde
        cv2.circle(icon, (size//2, 30), 5, (0, 255, 255), -1)  # Amarelo
        cv2.circle(icon, (size//2, 45), 5, (0, 0, 255), -1)  # Vermelho
        
        return icon
