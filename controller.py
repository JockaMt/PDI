"""
Controller - Lógica de controle para o analisador de tornado
"""

import threading
import os
from model import TornadoRiskAnalyzer


class TornadoController:
    """Controlador principal da aplicação"""

    def __init__(self, view):
        self.view = view
        self.analyzer = None

    def select_image(self, image_path):
        """Seleciona uma imagem para análise"""
        try:
            self.view.current_image_path = image_path
            self.view.load_and_display_image(image_path)
            self.view.update_ui_state(image_loaded=True)
            self.view.status_label.config(text=f"Imagem carregada: {image_path.split('/')[-1].split('\\')[-1]}")
        except Exception as e:
            self.view.show_error("Erro", f"Erro ao carregar imagem: {e}")

    def analyze_image(self, image_path):
        """Executa análise da imagem em thread separada"""
        if not image_path:
            return

        # Desabilita botão durante análise
        self.view.update_ui_state(analyzing=True)

        # Executa em thread separada para não travar a UI
        thread = threading.Thread(target=self.run_analysis, args=(image_path,))
        thread.daemon = True
        thread.start()

    def run_analysis(self, image_path):
        """Executa a análise da imagem"""
        try:
            # Cria analisador
            self.analyzer = TornadoRiskAnalyzer(image_path)

            # Realiza análises
            edges, contours = self.analyzer.analyze_cloud_patterns()
            hsv, green_mask, red_mask = self.analyzer.analyze_color_intensity()
            gradient = self.analyzer.analyze_texture_patterns()
            
            # Executa detecção YOLO (se disponível)
            self.analyzer.detect_tornado_patterns_yolo()
            
            # Calcula score incluindo YOLO
            risk_score, risk_level = self.analyzer.calculate_risk_score()

            # Atualiza UI na thread principal
            self.view.root.after(0, self.display_results, risk_score, risk_level, self.analyzer.risk_metrics)

        except Exception as e:
            self.view.root.after(0, lambda: self.view.show_error("Erro", f"Erro na análise: {e}"))
            self.view.root.after(0, lambda: self.view.update_ui_state())

    def display_results(self, risk_score, risk_level, metrics):
        """Exibe os resultados na view"""
        self.view.display_results(risk_score, risk_level, metrics)
        self.view.update_ui_state(analysis_complete=True)

    def generate_pdf_report(self, output_path):
        """Gera relatório PDF em thread separada"""
        if not self.analyzer:
            return

        # Desabilita botão durante geração
        self.view.update_ui_state(generating_pdf=True)
        
        # Executa em thread separada para não travar a UI
        thread = threading.Thread(target=self.run_pdf_generation, args=(output_path,))
        thread.daemon = True
        thread.start()

    def run_pdf_generation(self, output_path):
        """Executa a geração do PDF em thread separada"""
        try:
            self.view.root.after(0, lambda: self.view.status_label.config(text="Gerando PDF..."))
            
            # Usa caminho absoluto se não for relativo
            if not os.path.isabs(output_path):
                output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_path)
            
            # Gera PDF
            result = self.analyzer.generate_pdf_report(output_path)
            
            # Atualiza UI na thread principal
            self.view.root.after(0, lambda: self.view.show_success(
                "Sucesso", 
                f"Relatório PDF gerado com sucesso!\n\nLocal: {output_path}"
            ))
            self.view.root.after(0, lambda: self.view.status_label.config(text="PDF gerado com sucesso!"))
            
        except Exception as e:
            self.view.root.after(0, lambda: self.view.show_error("Erro", f"Erro ao gerar PDF: {e}"))
            self.view.root.after(0, lambda: self.view.status_label.config(text=f"Erro: {str(e)[:50]}"))
        
        finally:
            # Re-habilita botão
            self.view.root.after(0, lambda: self.view.update_ui_state())