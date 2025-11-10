"""
View - Interface gráfica para o analisador de tornado
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import os
import sv_ttk


class TornadoAnalyzerGUI:
    """Interface gráfica para análise de risco de tornado"""

    def __init__(self, controller):
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("Analisador de Risco de Tornado")
        self.root.geometry("1200x800")

        # Aplica o tema Sun Valley
        sv_ttk.set_theme("dark")

        # Variáveis
        self.current_image_path = None

        self.setup_ui()

    def setup_ui(self):
        """Configura a interface gráfica"""
        # Configuração da grade do frame principal
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Frame principal
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky='nsew')
        main_frame.rowconfigure(2, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Título
        title_label = ttk.Label(main_frame,
                               text="🌪️ Analisador de Risco de Tornado",
                               font=('Segoe UI', 20, 'bold'),
                               anchor='center')
        title_label.grid(row=0, column=0, pady=(0, 10), sticky='ew')

        # Frame superior - controles
        control_frame = ttk.Frame(main_frame, style='Card.TFrame', padding=10)
        control_frame.grid(row=1, column=0, pady=(0, 10), sticky='ew')

        # Botão para selecionar imagem
        self.select_btn = ttk.Button(control_frame,
                                    text="📁 Selecionar Imagem",
                                    command=self.select_image,
                                    style='Accent.TButton')
        self.select_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Botão para analisar
        self.analyze_btn = ttk.Button(control_frame,
                                     text="🔍 Analisar Imagem",
                                     command=self.analyze_image,
                                     state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Botão para gerar PDF
        self.pdf_btn = ttk.Button(control_frame,
                                 text="📄 Gerar Relatório PDF",
                                 command=self.generate_pdf,
                                 state=tk.DISABLED)
        self.pdf_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Label de status
        self.status_label = ttk.Label(control_frame,
                                     text="Selecione uma imagem para começar")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Painel com divisor para conteúdo
        content_pane = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        content_pane.grid(row=2, column=0, sticky='nsew')

        # Frame esquerdo - imagem
        left_frame = ttk.Frame(content_pane, style='Card.TFrame', padding=10)
        content_pane.add(left_frame, weight=1)

        ttk.Label(left_frame, text="Imagem Analisada",
                 font=('Segoe UI', 12, 'bold')).pack(pady=5)

        self.image_canvas = tk.Canvas(left_frame, bg='#1c1c1c', highlightthickness=0, width=400, height=300)
        self.image_canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Frame direito - resultados
        right_frame = ttk.Frame(content_pane, style='Card.TFrame', padding=10)
        content_pane.add(right_frame, weight=2)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        ttk.Label(right_frame, text="Resultados da Análise",
                 font=('Segoe UI', 12, 'bold')).grid(row=0, column=0, pady=5, sticky='ew')

        # Frame scrollable para conter texto e gráficos
        self.scrollable_frame = ttk.Frame(right_frame)
        self.scrollable_frame.grid(row=1, column=0, pady=(5, 0), sticky='nsew')
        self.scrollable_frame.rowconfigure(0, weight=1)
        self.scrollable_frame.columnconfigure(0, weight=1)

        # Canvas e scrollbar para o frame scrollable
        canvas = tk.Canvas(self.scrollable_frame, highlightthickness=0, bg='#1c1c1c')
        canvas.grid(row=0, column=0, sticky='nsew')

        scrollbar = ttk.Scrollbar(self.scrollable_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')

        scrollable_frame_inner = ttk.Frame(canvas)
        scrollable_frame_inner.columnconfigure(0, weight=1)

        canvas.create_window((0, 0), window=scrollable_frame_inner, anchor="nw", tags="inner_frame")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig("inner_frame", width=event.width)

        scrollable_frame_inner.bind("<Configure>", _on_frame_configure)

        # Área de texto para resultados dentro do frame scrollable
        self.results_text = tk.Text(scrollable_frame_inner, font=('Courier', 10),
                                   bg='#2d2d2d', fg='#f0f0f0', wrap=tk.WORD, height=15,
                                   relief='flat', borderwidth=0)
        self.results_text.grid(row=0, column=0, sticky='ew', pady=(0, 10))

        # Frame inferior - gráficos dentro do frame scrollable
        self.graphs_frame = ttk.Frame(scrollable_frame_inner, style='Card.TFrame', padding=10)
        # Inicialmente não posicionado com grid

        ttk.Label(self.graphs_frame, text="Histogramas de Análise",
                 font=('Segoe UI', 12, 'bold')).pack(pady=5)

    def select_image(self):
        """Abre caixa de diálogo para seleção de imagem"""
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
            filetypes=filetypes
        )

        if image_path:
            self.controller.select_image(image_path)

    def load_and_display_image(self, image_path):
        """Carrega e exibe a imagem no canvas"""
        try:
            # Carrega imagem
            image = Image.open(image_path)

            # Redimensiona mantendo proporção
            canvas_width = self.image_canvas.winfo_width() or 400
            canvas_height = self.image_canvas.winfo_height() or 300

            image.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)

            # Converte para tkinter
            self.photo = ImageTk.PhotoImage(image)

            # Exibe no canvas
            self.image_canvas.delete("all")
            x = (canvas_width - image.width) // 2
            y = (canvas_height - image.height) // 2
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=self.photo)

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar imagem: {e}")

    def analyze_image(self):
        """Executa análise em thread separada"""
        if not self.current_image_path:
            return

        self.controller.analyze_image(self.current_image_path)

    def display_results(self, risk_score, risk_level, metrics):
        """Exibe os resultados da análise"""
        # Limpa área de resultados
        self.results_text.delete(1.0, tk.END)

        # Cabeçalho
        self.results_text.insert(tk.END, "="*50 + "\n")
        self.results_text.insert(tk.END, "RELATÓRIO DE ANÁLISE DE TORNADO\n")
        self.results_text.insert(tk.END, "="*50 + "\n\n")

        # Informações da imagem
        self.results_text.insert(tk.END, f"Imagem: {os.path.basename(self.current_image_path)}\n")
        self.results_text.insert(tk.END, f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        self.results_text.insert(tk.END, f"Período: {metrics.get('period', 'Desconhecido')}\n\n")

        # Resultado principal
        self.results_text.insert(tk.END, "🎯 RESULTADO PRINCIPAL\n")
        self.results_text.insert(tk.END, "-"*30 + "\n")
        self.results_text.insert(tk.END, f"Índice de Risco: {risk_score:.1f}/100\n")
        self.results_text.insert(tk.END, f"Nível de Risco: {risk_level}\n\n")

        # Detalhes das métricas
        self.results_text.insert(tk.END, "📊 MÉTRICAS DETECTADAS\n")
        self.results_text.insert(tk.END, "-"*30 + "\n")

        self.results_text.insert(tk.END, f"Padrões Circulares: {metrics.get('circular_patterns', 0)}\n")
        self.results_text.insert(tk.END, f"Padrões Espirais: {metrics.get('spiral_patterns', 0)}\n")
        self.results_text.insert(tk.END, f"Formas de Funil: {metrics.get('funnel_shapes', 0)}\n")
        self.results_text.insert(tk.END, f"Áreas Muito Escuras: {metrics.get('very_dark_areas_percent', 0):.1f}%\n")
        self.results_text.insert(tk.END, f"Áreas Escuras: {metrics.get('dark_areas_percent', 0):.1f}%\n")
        self.results_text.insert(tk.END, f"Contraste: {metrics.get('contrast_percentage', 0):.1f}%\n")
        self.results_text.insert(tk.END, f"Áreas Verdes: {metrics.get('green_areas_percent', 0):.1f}%\n")
        self.results_text.insert(tk.END, f"Áreas Vermelhas: {metrics.get('red_areas_percent', 0):.1f}%\n\n")

        # Detalhes do cálculo
        if 'score_details' in metrics:
            self.results_text.insert(tk.END, "🔍 DETALHES DO CÁLCULO\n")
            self.results_text.insert(tk.END, "-"*30 + "\n")
            for detail in metrics['score_details']:
                self.results_text.insert(tk.END, f"• {detail}\n")
            self.results_text.insert(tk.END, "\n")

        # Recomendações
        self.results_text.insert(tk.END, "⚠️ RECOMENDAÇÕES\n")
        self.results_text.insert(tk.END, "-"*30 + "\n")

        if risk_score >= 85:
            self.results_text.insert(tk.END, "ALERTA MÁXIMO! Características de tornado detectadas!\n")
            self.results_text.insert(tk.END, "• Procure abrigo imediatamente\n")
            self.results_text.insert(tk.END, "• Monitore alertas meteorológicos\n")
        elif risk_score >= 65:
            self.results_text.insert(tk.END, "ATENÇÃO! Alto risco de tornado detectado!\n")
            self.results_text.insert(tk.END, "• Monitore condições meteorológicas\n")
            self.results_text.insert(tk.END, "• Tenha plano de evacuação pronto\n")
        elif risk_score >= 40:
            self.results_text.insert(tk.END, "Risco moderado. Monitoramento recomendado.\n")
        else:
            self.results_text.insert(tk.END, "Risco baixo baseado na análise visual.\n")

        # Reabilita botões
        self.analyze_btn.config(state=tk.NORMAL)
        self.pdf_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Análise concluída com sucesso!")

        # Exibe gráficos
        self.show_histograms(metrics)

    def show_histograms(self, metrics):
        """Exibe os histogramas da análise"""
        # Limpa gráficos anteriores
        for widget in self.graphs_frame.winfo_children():
            # Mantém o label do título
            if isinstance(widget, ttk.Label) and "Histogramas" in widget.cget("text"):
                continue
            widget.destroy()

        # Mostra frame de gráficos
        self.graphs_frame.grid(row=1, column=0, sticky='ew', pady=(10, 0))

        # Cria figura matplotlib
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))
        fig.patch.set_facecolor('#2d2d2d')

        # Gráfico 1: Padrões detectados
        patterns = [
            metrics.get('circular_patterns', 0),
            metrics.get('spiral_patterns', 0),
            metrics.get('funnel_shapes', 0)
        ]
        ax1.bar(['Circular', 'Espiral', 'Funil'], patterns,
               color=['#0078d4', '#e81123', '#ffb900'])
        ax1.set_title('Padrões de Rotação', color='white', fontweight='bold')
        ax1.set_ylabel('Quantidade', color='white')
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#1c1c1c')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color('white')
        ax1.spines['left'].set_color('white')

        # Gráfico 2: Distribuição de cores
        colors_values = [
            metrics.get('very_dark_areas_percent', 0),
            metrics.get('green_areas_percent', 0),
            metrics.get('red_areas_percent', 0),
            metrics.get('gray_storm_percent', 0)
        ]
        ax2.pie(colors_values, labels=['Muito Escuro', 'Verde', 'Vermelho', 'Cinza'],
               colors=['#1c1c1c', '#107c10', '#e81123', '#8e8e8e'], autopct='%1.1f%%',
               textprops={'color': 'white'})
        ax2.set_title('Distribuição de Cores', color='white', fontweight='bold')

        # Gráfico 3: Métricas de risco
        risk_components = []
        if 'score_details' in metrics:
            for detail in metrics['score_details']:
                if '/' in detail:
                    score = float(detail.split(':')[1].split('/')[0].strip())
                    risk_components.append(score)

        if risk_components:
            ax3.bar(range(len(risk_components)), risk_components,
                   color='#e81123', alpha=0.7)
            ax3.set_title('Componentes do Score', color='white', fontweight='bold')
            ax3.set_ylabel('Pontos', color='white')
            ax3.tick_params(colors='white')
            ax3.set_facecolor('#1c1c1c')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['bottom'].set_color('white')
            ax3.spines['left'].set_color('white')

        # Gráfico 4: Score final
        final_score = metrics.get('risk_score', 0)
        colors_score = ['#107c10' if final_score < 40 else '#ffb900' if final_score < 75 else '#e81123']
        ax4.bar(['Score de Risco'], [final_score], color=colors_score)
        ax4.set_ylim(0, 100)
        ax4.set_title('Índice Final de Risco', color='white', fontweight='bold')
        ax4.set_ylabel('Score (0-100)', color='white')
        ax4.tick_params(colors='white')
        ax4.set_facecolor('#1c1c1c')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['bottom'].set_color('white')
        ax4.spines['left'].set_color('white')

        # Ajusta layout
        plt.tight_layout()

        # Adiciona ao tkinter
        canvas = FigureCanvasTkAgg(fig, self.graphs_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Atualiza a região de scroll após adicionar conteúdo
        self.root.after(100, self.update_scroll_region)

    def update_scroll_region(self):
        """Atualiza a região de scroll do canvas"""
        try:
            # Encontra o canvas no scrollable_frame
            for widget in self.scrollable_frame.winfo_children():
                if isinstance(widget, tk.Canvas):
                    canvas = widget
                    break
            else:
                return

            # Atualiza a região de scroll
            canvas.configure(scrollregion=canvas.bbox("all"))
        except:
            pass  # Ignora erros se o canvas não estiver pronto

    def generate_pdf(self):
        """Gera relatório PDF"""
        if not hasattr(self.controller, 'analyzer') or not self.controller.analyzer:
            return

        output_name = f"relatorio_tornado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        try:
            self.controller.generate_pdf_report(output_name)
            messagebox.showinfo("Sucesso", f"Relatório PDF gerado: {output_name}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar PDF: {e}")

    def update_ui_state(self, image_loaded=False, analyzing=False, analysis_complete=False):
        """Atualiza o estado da interface"""
        if image_loaded:
            self.analyze_btn.config(state=tk.NORMAL)
            self.pdf_btn.config(state=tk.DISABLED)
            self.results_text.delete(1.0, tk.END)
            # Esconde e limpa gráficos
            self.graphs_frame.grid_remove()
            for widget in self.graphs_frame.winfo_children():
                # Mantém o label do título
                if isinstance(widget, ttk.Label) and "Histogramas" in widget.cget("text"):
                    continue
                widget.destroy()

        elif analyzing:
            self.analyze_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Analisando imagem... Aguarde...")
        elif analysis_complete:
            self.analyze_btn.config(state=tk.NORMAL)
            self.pdf_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Análise concluída com sucesso!")

    def show_error(self, title, message):
        """Exibe mensagem de erro"""
        messagebox.showerror(title, message)

    def run(self):
        """Inicia a interface gráfica"""
        self.root.mainloop()