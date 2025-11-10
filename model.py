"""
Model - Lógica de negócio para análise de risco de tornado
"""

import cv2
import numpy as np
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os

# Importação condicional de YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class TornadoRiskAnalyzer:
    """Analisador de risco de tornado baseado em processamento de imagens"""

    def __init__(self, image_path=None):
        """Inicializa o analisador com o caminho da imagem"""
        if image_path is None:
            image_path = self.select_image_dialog()
            if image_path is None:
                raise ValueError("Nenhuma imagem foi selecionada")

        self.image_path = image_path
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")

        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Métricas de risco
        self.risk_metrics = {}
        
        # Detecta condições de luminosidade
        self.avg_brightness = np.mean(self.gray)
        self.is_night = self.avg_brightness < 80  # Limiar: abaixo de 80 é noite
        self.period = "Noite" if self.is_night else "Dia"

    def select_image_dialog(self):
        """Abre caixa de diálogo para selecionar imagem"""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            return None

        root = tk.Tk()
        root.withdraw()  # Oculta a janela principal

        # Tipos de arquivo suportados
        filetypes = [
            ('Imagens', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'),
            ('JPEG', '*.jpg *.jpeg'),
            ('PNG', '*.png'),
            ('BMP', '*.bmp'),
            ('TIFF', '*.tiff *.tif'),
            ('Todos os arquivos', '*.*')
        ]

        image_path = filedialog.askopenfilename(
            title="Selecione uma imagem para análise de tornado",
            filetypes=filetypes,
            initialdir=os.getcwd()
        )

        root.destroy()

        return image_path if image_path else None

    def analyze_cloud_patterns(self):
        """Analisa padrões de nuvens que indicam rotação - detecção rigorosa para satélite"""

        # Detecção de bordas otimizada
        edges1 = cv2.Canny(self.gray, 20, 80)
        edges2 = cv2.Canny(self.gray, 40, 120)
        edges3 = cv2.Canny(self.gray, 60, 180)
        edges_combined = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # Detecta contornos
        contours, _ = cv2.findContours(edges_combined, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        circular_patterns = 0
        spiral_patterns = 0
        funnel_shapes = 0
        rotation_indicators = []
        
        area_threshold = 150  # Menor mas com validação rigorosa
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_threshold:
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                # === DETECÇÃO DE PADRÕES CIRCULARES ===
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                rotation_indicators.append(circularity)
                
                # Critério rigoroso de circularidade
                # Tornado visto de cima é muito circular
                if circularity > 0.65:  # Apenas formações MUITO circulares
                    # Validação adicional: verifica se é realmente circular
                    moments = cv2.moments(contour)
                    if moments['m00'] > 0:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                        
                        # Calcula raio médio e variância
                        distances = []
                        for point in contour:
                            x, y = point[0]
                            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                            distances.append(dist)
                        
                        if len(distances) > 20:
                            dist_mean = np.mean(distances)
                            dist_std = np.std(distances)
                            dist_cv = dist_std / dist_mean if dist_mean > 0 else 1.0
                            
                            # Círculo perfeito tem CV baixo (< 0.15)
                            if dist_cv < 0.2:
                                circular_patterns += 1
                
                # === DETECÇÃO DE PADRÕES ESPIRAIS ===
                moments = cv2.moments(contour)
                if moments['m00'] > 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    
                    distances = []
                    angles = []
                    for point in contour:
                        x, y = point[0]
                        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                        angle = np.arctan2(y - cy, x - cx)
                        distances.append(dist)
                        angles.append(angle)
                    
                    # Espiral rigorosa: distribuição angular consistente
                    if len(angles) > 25:
                        # Ordena pontos por ângulo
                        angle_sorted_idx = np.argsort(angles)
                        dist_sorted = np.array(distances)[angle_sorted_idx]
                        
                        # Calcula correlação entre ângulo e distância
                        angle_range = np.max(angles) - np.min(angles)
                        dist_range = np.max(dist_sorted) - np.min(dist_sorted)
                        
                        if angle_range > 0 and dist_range > 0:
                            # Coeficiente de correlação entre ângulo e distância
                            angles_normalized = (np.array(angles) - np.min(angles)) / angle_range
                            dist_normalized = (dist_sorted - np.min(dist_sorted)) / dist_range
                            correlation = np.abs(np.corrcoef(angles_normalized, dist_normalized)[0, 1])
                            
                            # Espiral real tem alta correlação (> 0.5)
                            if correlation > 0.55:
                                # Validação: deve ter rotação consistente
                                angle_diffs = np.abs(np.diff(angles))
                                angle_diffs = angle_diffs[angle_diffs < np.pi]
                                if len(angle_diffs) > 10:
                                    mean_diff = np.mean(angle_diffs)
                                    if 0.08 < mean_diff < 0.5:
                                        spiral_patterns += 1
                
                # === DETECÇÃO DE FORMAS DE FUNIL ===
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    solidity = area / hull_area  # Quanto mais baixo, mais cônico
                    
                    # Critério rigoroso: forma deve ser alongada e cônica
                    if 0.3 < solidity < 0.65:
                        # Validação adicional: verifica aspect ratio
                        rect = cv2.minAreaRect(contour)
                        if rect[1][0] > 0 and rect[1][1] > 0:
                            aspect_ratio = max(rect[1][0], rect[1][1]) / min(rect[1][0], rect[1][1])
                            
                            # Funil é alongado (aspect ratio > 1.5)
                            if aspect_ratio > 1.5:
                                # Verifica se é cônico (afunila para um ponto)
                                tip_density = self._check_funnel_tip(contour)
                                if tip_density > 0.6:
                                    funnel_shapes += 1
        
        self.risk_metrics['circular_patterns'] = circular_patterns
        self.risk_metrics['spiral_patterns'] = spiral_patterns
        self.risk_metrics['funnel_shapes'] = funnel_shapes
        self.risk_metrics['rotation_indicators'] = rotation_indicators
        
        return edges_combined, contours

    def _check_funnel_tip(self, contour):
        """Verifica se o contorno tem característica de funil (afunila para um ponto)"""
        if len(contour) < 10:
            return 0
        
        # Divide contorno em segmentos e verifica redução de área
        segments = []
        segment_size = len(contour) // 4
        
        for i in range(4):
            start = i * segment_size
            end = (i + 1) * segment_size if i < 3 else len(contour)
            segment = contour[start:end]
            if len(segment) > 0:
                segments.append(cv2.contourArea(segment))
        
        # Contorno que afunila deve ter redução consistente
        if len(segments) >= 2:
            diffs = []
            for i in range(len(segments) - 1):
                if segments[i] > 0:
                    diff = (segments[i] - segments[i + 1]) / segments[i]
                    diffs.append(diff)
            
            # Se há redução consistente, é provável um funil
            if len(diffs) >= 2:
                consistency = 1 - np.std(diffs)  # Quanto mais consistente, melhor
                return max(0, consistency)
        
        return 0

    def analyze_color_intensity(self):
        """Analisa intensidade de cores - otimizado para satélite"""

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Para satélite, padrões diferentes
        # Áreas muito escuras indicam nuvens densas (potencial tornado)
        very_dark_threshold = 50
        dark_threshold = 100

        very_dark_areas = np.sum(self.gray < very_dark_threshold)
        dark_areas = np.sum(self.gray < dark_threshold)

        very_dark_percentage = (very_dark_areas / self.gray.size) * 100
        dark_percentage = (dark_areas / self.gray.size) * 100

        # Contraste extremo
        contrast_areas = 0
        kernel_size = 5
        contrast_threshold = 25  # Mais sensível para satélite
        
        for i in range(0, self.gray.shape[0] - kernel_size, kernel_size):
            for j in range(0, self.gray.shape[1] - kernel_size, kernel_size):
                patch = self.gray[i:i+kernel_size, j:j+kernel_size]
                if patch.size > 0:
                    patch_std = np.std(patch)
                    if patch_std > contrast_threshold:
                        contrast_areas += 1

        contrast_percentage = (contrast_areas * kernel_size * kernel_size / self.gray.size) * 100

        # Em satélite, verde pode indicar áreas com nuvens de tempestade
        lower_green = np.array([20, 20, 20])
        upper_green = np.array([100, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = (np.sum(green_mask > 0) / green_mask.size) * 100

        # Vermelho/Laranja em satélite pode indicar calor ou áreas críticas
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([20, 255, 255])
        lower_red2 = np.array([150, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        lower_orange = np.array([10, 50, 50])
        upper_orange = np.array([30, 255, 255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        red_mask = cv2.bitwise_or(red_mask1, cv2.bitwise_or(red_mask2, orange_mask))
        red_percentage = (np.sum(red_mask > 0) / red_mask.size) * 100

        # Cinza (nuvens)
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 30, 200])
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        gray_percentage = (np.sum(gray_mask > 0) / gray_mask.size) * 100

        self.risk_metrics['very_dark_areas_percent'] = very_dark_percentage
        self.risk_metrics['dark_areas_percent'] = dark_percentage
        self.risk_metrics['contrast_percentage'] = contrast_percentage
        self.risk_metrics['green_areas_percent'] = green_percentage
        self.risk_metrics['red_areas_percent'] = red_percentage
        self.risk_metrics['gray_storm_percent'] = gray_percentage

        return hsv, green_mask, red_mask

    def analyze_texture_patterns(self):
        """Analisa texturas que podem indicar turbulência"""

        # Calcula gradientes (mudanças bruscas indicam turbulência)
        grad_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normaliza
        gradient_normalized = cv2.normalize(gradient_magnitude, None, 0, 255,
                                           cv2.NORM_MINMAX, cv2.CV_8U)

        # Calcula entropia (medida de aleatoriedade/caos)
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))

        # Alto gradiente indica mudanças bruscas
        high_gradient_areas = np.sum(gradient_normalized > 100)
        turbulence_index = (high_gradient_areas / gradient_normalized.size) * 100

        self.risk_metrics['turbulence_index'] = turbulence_index
        self.risk_metrics['entropy'] = entropy

        return gradient_normalized

    def detect_vortex_patterns(self):
        """Detecta padrões de vórtice com rigor - específico para satélite"""
        
        detected_vortices = 0
        spiral_confidence = 0
        
        # === DETECÇÃO 1: Hough Circles com múltiplas estratégias ===
        # Primeira passada - círculos bem definidos
        circles1 = cv2.HoughCircles(self.gray, cv2.HOUGH_GRADIENT,
                                    dp=1, minDist=80,
                                    param1=40, param2=35,
                                    minRadius=30, maxRadius=250)
        
        valid_circles = []
        if circles1 is not None:
            circles1 = np.uint16(np.around(circles1))
            for circle in circles1[0]:
                x, y, r = circle
                # Validação: verifica se o círculo tem padrão de tornado
                if self._validate_vortex_circle(x, y, r):
                    valid_circles.append(circle)
                    detected_vortices += 1
        
        # Segunda passada - círculos menores e mais intensos
        circles2 = cv2.HoughCircles(self.gray, cv2.HOUGH_GRADIENT,
                                    dp=1.5, minDist=50,
                                    param1=60, param2=25,
                                    minRadius=15, maxRadius=120)
        
        if circles2 is not None:
            circles2 = np.uint16(np.around(circles2))
            for circle in circles2[0]:
                x, y, r = circle
                # Evita duplicação
                is_duplicate = False
                for vc in valid_circles:
                    if np.sqrt((x - vc[0])**2 + (y - vc[1])**2) < r + vc[2]:
                        is_duplicate = True
                        break
                
                if not is_duplicate and self._validate_vortex_circle(x, y, r):
                    valid_circles.append(circle)
                    detected_vortices += 1
        
        # === DETECÇÃO 2: Análise de Rotação Angular ===
        # Procura por estruturas que giram consistentemente
        edges = cv2.Canny(self.gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                # Extrai pontos e analisa rotação
                moments = cv2.moments(contour)
                if moments['m00'] > 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    
                    # Calcula ângulos dos pontos
                    angles = []
                    distances = []
                    for point in contour:
                        x, y = point[0]
                        angle = np.arctan2(y - cy, x - cx)
                        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                        angles.append(angle)
                        distances.append(dist)
                    
                    if len(angles) > 30:
                        # Ordena por ângulo e verifica padrão
                        angle_array = np.array(angles)
                        dist_array = np.array(distances)
                        angle_idx = np.argsort(angle_array)
                        dist_sorted = dist_array[angle_idx]
                        
                        # Calcula suavidade da rotação
                        angle_diffs = np.abs(np.diff(angle_array[angle_idx]))
                        angle_diffs = angle_diffs[angle_diffs < np.pi]
                        
                        if len(angle_diffs) > 15:
                            mean_angle_diff = np.mean(angle_diffs)
                            angle_consistency = 1 - np.std(angle_diffs) / (mean_angle_diff + 0.01)
                            
                            # Rotação consistente = espiral ou vórtice
                            if 0.1 < mean_angle_diff < 0.3 and angle_consistency > 0.5:
                                spiral_confidence += 1
        
        self.risk_metrics['detected_vortices'] = detected_vortices
        self.risk_metrics['spiral_confidence'] = spiral_confidence
        self.risk_metrics['valid_vortex_circles'] = valid_circles

    def detect_tornado_patterns_yolo(self):
        """Detecta padrões de tornado usando YOLO (Deep Learning)"""
        if not YOLO_AVAILABLE:
            return 0, 0, 0
        
        try:
            
            # Carrega modelo YOLO pré-treinado (detecção genérica)
            # Usaremos YOLOv8n (nano) para ser rápido
            model = YOLO('yolo12n.pt')
            
            # Executa detecção
            results = model(self.image_rgb, verbose=False)
            
            yolo_detections = 0
            yolo_confidence = 0
            yolo_circular_objects = 0
            
            # Processa resultados
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i, box in enumerate(boxes):
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        # Coordenadas
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Calcula aspect ratio (circular = ratio próximo a 1)
                        aspect_ratio = max(w, h) / (min(w, h) + 1)
                        
                        # Objetos circulares detectados têm confiança alta
                        if conf > 0.5 and 0.7 < aspect_ratio < 1.4:
                            # Pode ser formação circular de tornado
                            yolo_circular_objects += 1
                            yolo_confidence += conf
                        
                        if conf > 0.6:
                            yolo_detections += 1
            
            # Calcula média de confiança
            if yolo_detections > 0:
                yolo_confidence = yolo_confidence / yolo_detections
            
            self.risk_metrics['yolo_detections'] = yolo_detections
            self.risk_metrics['yolo_circular'] = yolo_circular_objects
            self.risk_metrics['yolo_confidence'] = yolo_confidence
            
            return yolo_detections, yolo_circular_objects, yolo_confidence
            
        except Exception as e:
            return 0, 0, 0

    def _validate_vortex_circle(self, cx, cy, r):
        """Valida se um círculo detectado é realmente um vórtice tornado"""
        if r < 15 or r > 250:
            return False
        
        # Extrai região circular
        y1 = max(0, int(cy - r))
        y2 = min(self.gray.shape[0], int(cy + r))
        x1 = max(0, int(cx - r))
        x2 = min(self.gray.shape[1], int(cx + r))
        
        region = self.gray[y1:y2, x1:x2]
        if region.size == 0:
            return False
        
        # Vórtices tornado têm características específicas:
        # 1. Núcleo mais escuro que a borda
        center_mask = cv2.circle(np.zeros(region.shape, dtype=np.uint8), 
                                (r, r), int(r * 0.3), 255, -1)
        edge_mask = cv2.circle(np.zeros(region.shape, dtype=np.uint8), 
                              (r, r), int(r * 0.8), 255, -1) - center_mask
        
        if np.sum(center_mask) == 0 or np.sum(edge_mask) == 0:
            return False
        
        center_brightness = np.mean(region[center_mask > 0])
        edge_brightness = np.mean(region[edge_mask > 0])
        
        # Centro deve ser mais escuro
        if center_brightness >= edge_brightness:
            return False
        
        # 2. Estrutura radial/circular consistente
        gradient = cv2.Canny(region, 20, 60)
        if np.sum(gradient) < 100:
            return False
        
        # 3. Razão entre intensidade interna e externa
        intensity_ratio = center_brightness / (edge_brightness + 1)
        if intensity_ratio > 0.95:  # Muito semelhante = não é vórtice
            return False
        
        return True

    def calculate_risk_score(self):
        """Calcula pontuação de risco de tornado (0-100) - Otimizado para satélite"""

        score = 0
        details = []

        # Recupera métricas
        circular_patterns = self.risk_metrics.get('circular_patterns', 0)
        spiral_patterns = self.risk_metrics.get('spiral_patterns', 0)
        funnel_shapes = self.risk_metrics.get('funnel_shapes', 0)
        detected_vortices = self.risk_metrics.get('detected_vortices', 0)
        spiral_confidence = self.risk_metrics.get('spiral_confidence', 0)
        very_dark = self.risk_metrics.get('very_dark_areas_percent', 0)
        dark = self.risk_metrics.get('dark_areas_percent', 0)
        contrast = self.risk_metrics.get('contrast_percentage', 0)
        green = self.risk_metrics.get('green_areas_percent', 0)
        red = self.risk_metrics.get('red_areas_percent', 0)
        gray_storm = self.risk_metrics.get('gray_storm_percent', 0)
        turbulence = self.risk_metrics.get('turbulence_index', 0)
        entropy = self.risk_metrics.get('entropy', 0)

        # === SCORES OTIMIZADOS PARA SATÉLITE ===
        
        # 1. Padrões de rotação/circulares (0-50 pontos) - FATOR CRÍTICO
        rotation_score = 0
        
        # Circular: apenas padrões muito bem validados
        if circular_patterns >= 1:
            rotation_score += min(circular_patterns * 15, 25)
            details.append(f"  Circulares validados: {circular_patterns} x 15")
        
        # Vórtices por Hough: detecção rigorosa
        if detected_vortices >= 1:
            rotation_score += min(detected_vortices * 12, 25)
            details.append(f"  Vórtices rigorosos: {detected_vortices} x 12")
        
        # Espirais: apenas correlação forte
        if spiral_patterns >= 1:
            rotation_score += min(spiral_patterns * 10, 20)
            details.append(f"  Espirais confirmadas: {spiral_patterns} x 10")
        
        # Confiança de rotação angular
        if spiral_confidence >= 1:
            rotation_score += min(spiral_confidence * 8, 15)
            details.append(f"  Rotações validadas: {spiral_confidence} x 8")
        
        # Funis: formas muito específicas
        if funnel_shapes >= 1:
            rotation_score += min(funnel_shapes * 12, 20)
            details.append(f"  Funis detectados: {funnel_shapes} x 12")
        
        rotation_score = min(rotation_score, 50)
        score += rotation_score
        details.append(f"\n→ Subtotal Rotação: {rotation_score:.1f}/50")

        # 2. Áreas escuras (0-25 pontos)
        darkness_score = 0
        if very_dark > 10:
            darkness_score = min((very_dark - 10) * 1.2, 25)
        score += darkness_score
        details.append(f"Densidade das nuvens: {darkness_score:.1f}/25 (muito escuro:{very_dark:.1f}%)")

        # 3. Contraste extremo (0-15 pontos)
        contrast_score = 0
        if contrast > 8:
            contrast_score = min((contrast - 8) * 1.5, 15)
        score += contrast_score
        details.append(f"Contraste/Turbulência: {contrast_score:.1f}/15 (contraste:{contrast:.1f}%)")

        # 4. Cores de tempestade (0-10 pontos)
        color_score = 0
        if green > 5:
            color_score += min(green * 1.0, 5)
        if red > 3:
            color_score += min(red * 1.5, 5)
        color_score = min(color_score, 10)
        score += color_score
        details.append(f"Cores de tempestade: {color_score:.1f}/10 (verde:{green:.1f}%, verm:{red:.1f}%)")

        # 5. Detecção YOLO (0-15 pontos)
        yolo_score = 0
        if YOLO_AVAILABLE and 'yolo_circular' in self.risk_metrics:
            yolo_circular = self.risk_metrics.get('yolo_circular', 0)
            if yolo_circular > 0:
                yolo_score = min(yolo_circular * 5, 15)
                score += yolo_score
                details.append(f"Detecção YOLO: {yolo_score:.1f}/15 (objetos circulares:{yolo_circular})")

        # === BÔNUS: Combinações perigosas ===
        bonus_score = 0

        # Bônus 1: Vórtices + escuridão = TORNADO PROVÁVEL
        if detected_vortices >= 1 and very_dark > 15:
            bonus_score += 15
            details.append("🔴 BÔNUS: Vórtice + nuvens escuras (+15)")

        # Bônus 2: Múltiplos padrões de rotação
        total_rotation_patterns = circular_patterns + detected_vortices + spiral_patterns
        if total_rotation_patterns >= 3:
            bonus_score += 10
            details.append(f"� BÔNUS: {total_rotation_patterns} padrões de rotação (+10)")

        # Bônus 3: Alto contraste + escuridão + rotação
        if contrast > 15 and very_dark > 20 and (detected_vortices + circular_patterns) >= 1:
            bonus_score += 20
            details.append("⚠️  ALERTA: Tripla ameaça detectada (+20)")

        # Bônus 4: YOLO confirmando padrões estruturais
        if YOLO_AVAILABLE and 'yolo_circular' in self.risk_metrics:
            yolo_circular = self.risk_metrics.get('yolo_circular', 0)
            if yolo_circular > 0 and (detected_vortices + circular_patterns) >= 1:
                bonus_score += 10
                details.append(f"🤖 BÔNUS: YOLO confirmou padrões estruturais (+10)")

        score += bonus_score

        # Score final
        final_score = min(score, 100)

        self.risk_metrics['risk_score'] = final_score
        self.risk_metrics['score_details'] = details
        self.risk_metrics['period'] = self.period

        # Classificação
        if final_score < 15:
            risk_level = "BAIXO"
        elif final_score < 30:
            risk_level = "MODERADO"
        elif final_score < 50:
            risk_level = "ELEVADO"
        elif final_score < 70:
            risk_level = "ALTO"
        elif final_score < 85:
            risk_level = "MUITO ALTO"
        else:
            risk_level = "CRÍTICO - POSSÍVEL TORNADO"

        self.risk_metrics['risk_level'] = risk_level

        return final_score, risk_level

    def generate_pdf_report(self, output_path="relatorio_tornado.pdf"):
        """Gera relatório em PDF com histogramas e análises"""

        # Realiza todas as análises
        edges, contours = self.analyze_cloud_patterns()
        hsv, green_mask, red_mask = self.analyze_color_intensity()
        gradient = self.analyze_texture_patterns()
        self.detect_vortex_patterns()  # Nova detecção

        # Debug das métricas detectadas
        self.debug_metrics()

        # Calcula score de risco
        risk_score, risk_level = self.calculate_risk_score()

        # Cria PDF
        with PdfPages(output_path) as pdf:
            # PÁGINA 1: Capa com Resumo Executivo (Design Profissional)
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor('white')
            ax_main = fig.add_subplot(111)
            ax_main.set_xlim(0, 10)
            ax_main.set_ylim(0, 10)
            ax_main.axis('off')
            
            # Define cor baseado no risco
            if risk_level == "BAIXO":
                risk_color = '#2ecc71'  # Verde
                risk_emoji = '🟢'
            elif risk_level == "MODERADO":
                risk_color = '#f39c12'  # Laranja
                risk_emoji = '🟡'
            elif risk_level == "ALTO":
                risk_color = '#e74c3c'  # Vermelho
                risk_emoji = '🟠'
            else:
                risk_color = '#c0392b'  # Vermelho escuro
                risk_emoji = '🔴'
            
            # Título principal
            ax_main.text(5, 9.2, '🌪️ ANÁLISE DE RISCO DE TORNADO 🌪️',
                        fontsize=24, fontweight='bold', ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
            
            # Linha divisória
            ax_main.plot([0.5, 9.5], [8.8, 8.8], 'k-', linewidth=2)
            
            # Seção 1: SCORE DE RISCO (Bem destacado)
            ax_main.text(5, 8.3, 'ÍNDICE DE RISCO',
                        fontsize=14, fontweight='bold', ha='center')
            
            # Caixa grande com o score
            risk_box_text = f'{risk_score:.1f}/100'
            ax_main.add_patch(plt.Rectangle((3, 6.8), 4, 1.2, 
                                           facecolor=risk_color, alpha=0.2, edgecolor=risk_color, linewidth=3))
            ax_main.text(5, 7.4, risk_box_text,
                        fontsize=32, fontweight='bold', ha='center', va='center',
                        color=risk_color)
            
            ax_main.text(5, 6.4, f'{risk_emoji} {risk_level}',
                        fontsize=12, fontweight='bold', ha='center', style='italic')
            
            # Linha divisória
            ax_main.plot([0.5, 9.5], [6, 6], 'k-', linewidth=1, linestyle='--', alpha=0.5)
            
            # Seção 2: INFORMAÇÕES DA IMAGEM
            info_left = f"""INFORMAÇÕES DA ANÁLISE

📅 Data: {datetime.now().strftime('%d/%m/%Y')}
🕐 Hora: {datetime.now().strftime('%H:%M:%S')}
📄 Arquivo: {os.path.basename(self.image_path)[:30]}
🌙 Período: {self.period}
"""
            ax_main.text(0.7, 5.3, info_left, fontsize=9, family='monospace',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.3))
            
            # Seção 3: MÉTRICAS PRINCIPAIS
            metrics_text = f"""MÉTRICAS DETECTADAS

🔵 Padrões Circulares: {self.risk_metrics['circular_patterns']}
⚫ Vórtices: {self.risk_metrics['detected_vortices']}
🌀 Espirais: {self.risk_metrics['spiral_patterns']}
🔻 Funis: {self.risk_metrics['funnel_shapes']}
"""
            ax_main.text(5.5, 5.3, metrics_text, fontsize=9, family='monospace',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.3))
            
            # Seção 4: ANÁLISE DETALHADA
            analysis_text = f"""ANÁLISE DETALHADA

📊 Áreas Escuras: {self.risk_metrics['dark_areas_percent']:.1f}%
🟢 Áreas Verdes: {self.risk_metrics['green_areas_percent']:.1f}%
🔴 Áreas Vermelhas: {self.risk_metrics['red_areas_percent']:.1f}%
❄️ Turbulência: {self.risk_metrics['turbulence_index']:.1f}%
📈 Entropia: {self.risk_metrics['entropy']:.2f}
🤖 YOLO Detect: {self.risk_metrics.get('yolo_detections', 0)}
"""
            ax_main.text(0.7, 2.8, analysis_text, fontsize=9, family='monospace',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.2))
            
            # Seção 5: RECOMENDAÇÕES
            if risk_score < 20:
                rec = "✓ Risco mínimo - Monitorar"
            elif risk_score < 50:
                rec = "⚠ Risco moderado - Alertar"
            elif risk_score < 80:
                rec = "⚠⚠ Risco elevado - Agir"
            else:
                rec = "🚨 RISCO CRÍTICO - AÇÃO IMEDIATA"
                
            ax_main.text(5.5, 2.8, f"""RECOMENDAÇÃO

{rec}

Status: {'🔴 ATIVO' if risk_score >= 50 else '🟡 OBSERVAÇÃO' if risk_score >= 20 else '🟢 NORMAL'}
""", fontsize=9, family='monospace',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='mistyrose', alpha=0.3))
            
            # Linha divisória final
            ax_main.plot([0.5, 9.5], [0.8, 0.8], 'k-', linewidth=2)
            
            # Rodapé
            ax_main.text(5, 0.3, 
                        f'Relatório Gerado por: 🌪️ Analisador de Risco de Tornado v1.0 | Método: Clássico + YOLO Deep Learning',
                        fontsize=7, ha='center', style='italic', color='gray')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # PÁGINA 2: Imagens Técnicas
            fig = plt.figure(figsize=(11, 8.5))

            # Imagem original
            ax1 = plt.subplot(2, 2, 1)
            ax1.imshow(self.image_rgb)
            ax1.set_title('Imagem Original (Satélite)', fontsize=12, fontweight='bold')
            ax1.axis('off')

            # Detecção de bordas
            ax2 = plt.subplot(2, 2, 2)
            ax2.imshow(edges, cmap='gray')
            ax2.set_title('Detecção de Bordas (Canny)', fontsize=12, fontweight='bold')
            ax2.axis('off')

            # Análise de gradiente
            ax3 = plt.subplot(2, 2, 3)
            ax3.imshow(gradient, cmap='hot')
            ax3.set_title('Índice de Turbulência', fontsize=12, fontweight='bold')
            ax3.axis('off')

            # HSV Analysis
            ax4 = plt.subplot(2, 2, 4)
            ax4.imshow(hsv)
            ax4.set_title('Análise HSV (Cores)', fontsize=12, fontweight='bold')
            ax4.axis('off')

            plt.suptitle('ANÁLISE TÉCNICA - PROCESSAMENTO DE IMAGEM',
                        fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # PÁGINA 3: Detecção de Padrões (Circular, Espiral, Funil)
            fig = plt.figure(figsize=(11, 8.5))
            
            # Prepara a imagem original com contornos destacados
            image_with_patterns = self.image_rgb.copy()
            
            # Detecta contornos novamente para desenhar
            edges_temp = cv2.Canny(self.gray, 50, 150)
            edges_temp = cv2.bitwise_or(edges_temp, cv2.Canny(self.gray, 80, 200))
            edges_temp = cv2.bitwise_or(edges_temp, cv2.Canny(self.gray, 120, 250))
            contours_temp, _ = cv2.findContours(edges_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Prepara informações sobre contornos
            area_threshold = 300 if self.is_night else 500
            circularity_threshold = 0.6 if self.is_night else 0.75
            
            circular_contours = []
            spiral_contours = []
            funnel_contours = []
            
            for contour in contours_temp:
                area = cv2.contourArea(contour)
                if area > area_threshold:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # Classifica padrão
                        if circularity > circularity_threshold:
                            circular_contours.append(contour)
                        
                        # Funil
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            solidity = area / hull_area
                            if solidity < 0.5:
                                funnel_contours.append(contour)
                        
                        # Espiral
                        moments = cv2.moments(contour)
                        if moments['m00'] > 0:
                            cx = int(moments['m10'] / moments['m00'])
                            cy = int(moments['m01'] / moments['m00'])
                            
                            distances = []
                            for point in contour:
                                x, y = point[0]
                                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                                distances.append(dist)
                            
                            if len(distances) > 15:
                                dist_std = np.std(distances)
                                dist_mean = np.mean(distances)
                                if dist_mean > 0 and (dist_std / dist_mean) > 0.5:
                                    spiral_contours.append(contour)
            
            # Subplot 1: Padrões Circulares
            ax1 = plt.subplot(2, 2, 1)
            img_circular = self.image_rgb.copy()
            cv2.drawContours(img_circular, circular_contours, -1, (0, 255, 0), 2)
            ax1.imshow(img_circular)
            ax1.set_title(f'Padrões Circulares\n({len(circular_contours)} detectados)', 
                         fontsize=12, fontweight='bold', color='green')
            ax1.axis('off')
            
            # Subplot 2: Padrões Espirais
            ax2 = plt.subplot(2, 2, 2)
            img_spiral = self.image_rgb.copy()
            cv2.drawContours(img_spiral, spiral_contours, -1, (255, 165, 0), 2)
            ax2.imshow(img_spiral)
            ax2.set_title(f'Padrões Espirais\n({len(spiral_contours)} detectados)', 
                         fontsize=12, fontweight='bold', color='orange')
            ax2.axis('off')
            
            # Subplot 3: Formas de Funil
            ax3 = plt.subplot(2, 2, 3)
            img_funnel = self.image_rgb.copy()
            cv2.drawContours(img_funnel, funnel_contours, -1, (255, 0, 0), 2)
            ax3.imshow(img_funnel)
            ax3.set_title(f'Formas de Funil\n({len(funnel_contours)} detectados)', 
                         fontsize=12, fontweight='bold', color='red')
            ax3.axis('off')
            
            # Subplot 4: Todos os padrões combinados
            ax4 = plt.subplot(2, 2, 4)
            img_combined = self.image_rgb.copy()
            cv2.drawContours(img_combined, circular_contours, -1, (0, 255, 0), 2)   # Verde
            cv2.drawContours(img_combined, spiral_contours, -1, (255, 165, 0), 2)   # Laranja
            cv2.drawContours(img_combined, funnel_contours, -1, (255, 0, 0), 2)     # Vermelho
            ax4.imshow(img_combined)
            ax4.set_title('Todos os Padrões Detectados\n(Verde: Circular | Laranja: Espiral | Vermelho: Funil)', 
                         fontsize=11, fontweight='bold')
            ax4.axis('off')
            
            plt.suptitle('DETECÇÃO DE PADRÕES - CIRCULAR, ESPIRAL E FUNIL',
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # PÁGINA OPCIONAL: Detecção YOLO (Deep Learning)
            if YOLO_AVAILABLE and 'yolo_circular' in self.risk_metrics:
                fig = plt.figure(figsize=(11, 8.5))
                
                yolo_detections = self.risk_metrics.get('yolo_detections', 0)
                yolo_circular = self.risk_metrics.get('yolo_circular', 0)
                yolo_confidence = self.risk_metrics.get('yolo_confidence', 0)
                
                # Se há detecções YOLO, visualiza
                if yolo_detections > 0:
                    # Subplot 1: Imagem com bounding boxes YOLO
                    ax1 = plt.subplot(2, 2, 1)
                    img_yolo = self.image_rgb.copy()
                    
                    # Se temos modelo YOLO, roda novamente para pegar boxes
                    try:
                        model = YOLO('yolov8n.pt')
                        results = model(self.image_rgb, verbose=False)
                        
                        for result in results:
                            if result.boxes is not None:
                                boxes = result.boxes
                                for box in boxes:
                                    x1, y1, x2, y2 = box.xyxy[0]
                                    conf = float(box.conf[0])
                                    if conf > 0.5:
                                        # Desenha bounding box em verde
                                        cv2.rectangle(img_yolo, 
                                                    (int(x1), int(y1)), 
                                                    (int(x2), int(y2)), 
                                                    (0, 255, 0), 2)
                                        # Escreve confiança
                                        cv2.putText(img_yolo, f'{conf:.2f}', 
                                                  (int(x1), int(y1)-10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.5, (0, 255, 0), 1)
                    except:
                        pass
                    
                    ax1.imshow(img_yolo)
                    ax1.set_title(f'Detecção YOLO\n({yolo_detections} objetos detectados)', 
                                 fontsize=12, fontweight='bold')
                    ax1.axis('off')
                    
                    # Subplot 2: Objetos circulares detectados
                    ax2 = plt.subplot(2, 2, 2)
                    yolo_info = f"""ANÁLISE YOLO (Deep Learning)

Total de Objetos: {yolo_detections}
Objetos Circulares: {yolo_circular}
Confiança Média: {yolo_confidence:.2f}

INTERPRETAÇÃO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A rede neural YOLO detectou {yolo_detections} 
objetos distintos na imagem de satélite.

{yolo_circular} deles apresentam forma 
aproximadamente circular, o que pode 
indicar formações de tornado em 
desenvolvimento.

Confiança de detecção: {yolo_confidence*100:.1f}%
"""
                    ax2.axis('off')
                    ax2.text(0.1, 0.5, yolo_info, fontsize=10, family='monospace',
                            verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.2))
                    
                    # Subplot 3: Comparação com métodos clássicos
                    ax3 = plt.subplot(2, 2, 3)
                    methods = ['YOLO', 'Circular', 'Vórtices', 'Espiral']
                    detections = [
                        yolo_circular,
                        self.risk_metrics.get('circular_patterns', 0),
                        self.risk_metrics.get('detected_vortices', 0),
                        self.risk_metrics.get('spiral_patterns', 0)
                    ]
                    colors_bars = ['#00BFFF', '#00FF00', '#FF8C00', '#FF0000']
                    ax3.bar(methods, detections, color=colors_bars, alpha=0.7, edgecolor='black')
                    ax3.set_title('Comparação de Métodos de Detecção', fontweight='bold')
                    ax3.set_ylabel('Contagem')
                    ax3.grid(True, alpha=0.3, axis='y')
                    
                    # Subplot 4: Resumo de confiança
                    ax4 = plt.subplot(2, 2, 4)
                    ax4.axis('off')
                    confidence_text = f"""MÉTRICAS YOLO

Confiança Média: {yolo_confidence:.3f}
Taxa de Detecção: {(yolo_circular/max(yolo_detections,1)*100):.1f}%

VALIDAÇÃO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Detectado por YOLO
✓ Confiança > 0.5
✓ Forma aproximadamente circular

Recomendação:
CORRELACIONAR COM PADRÕES
CLÁSSICOS para aumentar
confiabilidade da previsão.
"""
                    ax4.text(0.1, 0.5, confidence_text, fontsize=10, family='monospace',
                            verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2))
                    
                    plt.suptitle('DETECÇÃO POR DEEP LEARNING (YOLO)',
                                fontsize=16, fontweight='bold', y=0.98)
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                else:
                    # Se YOLO disponível mas sem detecções
                    fig = plt.figure(figsize=(11, 8.5))
                    ax = fig.add_subplot(111)
                    ax.axis('off')
                    no_yolo_msg = """DETECÇÃO YOLO

A rede neural YOLO foi executada, mas nenhum 
objeto foi detectado que corresponda aos 
critérios de interesse para formações de tornado.

Isso pode significar:
• Imagem muito clara (sem tempestade)
• Padrões muito sutis para a detecção
• Formação ainda em estágios iniciais

Recomendação:
Use os métodos clássicos (Circular, Vórtice, Espiral)
como guia principal de análise.
"""
                    ax.text(0.5, 0.5, no_yolo_msg, fontsize=12, family='monospace',
                           verticalalignment='center', horizontalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                    plt.suptitle('DETECÇÃO POR DEEP LEARNING (YOLO)',
                                fontsize=16, fontweight='bold')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

            # Página 2: Histogramas
            fig = plt.figure(figsize=(11, 8.5))

            # Histograma de intensidade
            ax1 = plt.subplot(2, 2, 1)
            hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
            ax1.plot(hist, color='blue')
            ax1.fill_between(range(256), hist.ravel(), alpha=0.3)
            ax1.set_title('Histograma de Intensidade', fontweight='bold')
            ax1.set_xlabel('Intensidade')
            ax1.set_ylabel('Frequência')
            ax1.grid(True, alpha=0.3)

            # Histograma RGB
            ax2 = plt.subplot(2, 2, 2)
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                ax2.plot(hist, color=color, alpha=0.7, label=color.upper())
            ax2.set_title('Histograma RGB', fontweight='bold')
            ax2.set_xlabel('Intensidade')
            ax2.set_ylabel('Frequência')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Distribuição de circularidade
            ax3 = plt.subplot(2, 2, 3)
            if self.risk_metrics['rotation_indicators']:
                ax3.hist(self.risk_metrics['rotation_indicators'],
                        bins=20, color='purple', alpha=0.7, edgecolor='black')
                ax3.axvline(x=0.7, color='red', linestyle='--',
                           label='Limiar de risco (0.7)')
                ax3.legend()
            ax3.set_title('Distribuição de Circularidade', fontweight='bold')
            ax3.set_xlabel('Índice de Circularidade')
            ax3.set_ylabel('Frequência')
            ax3.grid(True, alpha=0.3)

            # Gráfico de pizza com métricas
            ax4 = plt.subplot(2, 2, 4)
            metrics_values = [
                self.risk_metrics['dark_areas_percent'],
                self.risk_metrics['green_areas_percent'],
                self.risk_metrics['red_areas_percent'],
                100 - (self.risk_metrics['dark_areas_percent'] +
                       self.risk_metrics['green_areas_percent'] +
                       self.risk_metrics['red_areas_percent'])
            ]
            labels = ['Áreas Escuras', 'Áreas Verdes', 'Áreas Vermelhas', 'Outras']
            colors_pie = ['#333333', '#00ff00', '#ff0000', '#cccccc']
            ax4.pie(metrics_values, labels=labels, colors=colors_pie,
                   autopct='%1.1f%%', startangle=90)
            ax4.set_title('Distribuição de Áreas de Risco', fontweight='bold')

            plt.suptitle('HISTOGRAMAS E DISTRIBUIÇÕES',
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()