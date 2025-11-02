import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage

from src.application.use_cases.classify_image_use_case import ClassifyImageUseCase
from src.infrastructure.data_loader.image_data_loader import ImageDataLoader
from src.infrastructure.models.cnn_model import CNNModel


class ClassifyImageApp:
    """Interface gráfica para classificação de imagens."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Classificador de Imagens - CNN")
        self.root.geometry("1200x800")
        
        # Componentes
        self.data_loader = ImageDataLoader()
        self.model_repository = CNNModel()
        self.classify_use_case = ClassifyImageUseCase(
            self.model_repository,
            self.data_loader
        )
        
        # Variáveis
        self.model_path = "models/cnn_model.h5"
        self.test_path = "data/test"
        self.image_size = (224, 224)
        self.classes = []
        self.model_loaded = False
        
        # Interface
        self.create_widgets()
        
        # Tenta carregar modelo e mostrar resumo
        self.load_model_and_summary()
    
    def create_widgets(self):
        """Cria os widgets da interface."""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Título
        title_label = ttk.Label(
            main_frame,
            text="🔍 Classificador de Imagens CNN",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Frame de informações do modelo
        model_info_frame = ttk.LabelFrame(main_frame, text="Informações do Modelo", padding="10")
        model_info_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.model_summary_text = tk.Text(model_info_frame, height=12, width=80, wrap=tk.WORD)
        self.model_summary_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        scrollbar = ttk.Scrollbar(model_info_frame, orient=tk.VERTICAL, command=self.model_summary_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.model_summary_text.configure(yscrollcommand=scrollbar.set)
        
        # Frame de seleção de arquivo
        file_frame = ttk.LabelFrame(main_frame, text="Selecionar Imagem", padding="10")
        file_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.file_path_var = tk.StringVar(value="Nenhum arquivo selecionado")
        file_label = ttk.Label(file_frame, textvariable=self.file_path_var)
        file_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        
        browse_button = ttk.Button(
            file_frame,
            text="📁 Procurar Arquivo...",
            command=self.browse_file
        )
        browse_button.grid(row=0, column=1, padx=5)
        
        classify_button = ttk.Button(
            file_frame,
            text="🔍 Classificar Imagem",
            command=self.classify_image,
            state=tk.DISABLED
        )
        classify_button.grid(row=0, column=2, padx=5)
        self.classify_button = classify_button
        
        # Frame de resultado
        result_frame = ttk.LabelFrame(main_frame, text="Resultado da Classificação", padding="10")
        result_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        main_frame.rowconfigure(3, weight=1)
        
        self.result_text = tk.Text(result_frame, height=4, width=80, wrap=tk.WORD)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Frame de visualização
        viz_frame = ttk.LabelFrame(main_frame, text="Visualização", padding="10")
        viz_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        main_frame.rowconfigure(4, weight=1)
        
        self.fig = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Frame de botões
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        refresh_button = ttk.Button(
            button_frame,
            text="🔄 Recarregar Modelo",
            command=self.load_model_and_summary
        )
        refresh_button.grid(row=0, column=0, padx=5)
        
        quit_button = ttk.Button(
            button_frame,
            text="❌ Sair",
            command=self.root.quit
        )
        quit_button.grid(row=0, column=1, padx=5)
    
    def load_model_and_summary(self):
        """Carrega o modelo e mostra o resumo."""
        try:
            # Verifica se o modelo existe
            if not os.path.exists(self.model_path):
                self.model_summary_text.delete(1.0, tk.END)
                self.model_summary_text.insert(tk.END, f"❌ Modelo não encontrado em: {self.model_path}\n")
                self.model_summary_text.insert(tk.END, "Por favor, treine o modelo primeiro usando: python -m src.presentation.main\n")
                self.model_loaded = False
                self.classify_button.config(state=tk.DISABLED)
                return
            
            # Carrega o modelo
            if not self.classify_use_case.load_model(self.model_path):
                self.model_summary_text.delete(1.0, tk.END)
                self.model_summary_text.insert(tk.END, "❌ Erro ao carregar modelo\n")
                self.model_loaded = False
                self.classify_button.config(state=tk.DISABLED)
                return
            
            # Obtém classes
            train_path = "data/train"
            if os.path.exists(train_path):
                self.classes = self.data_loader.get_class_names(train_path)
            else:
                # Tenta obter do dataset de teste
                if os.path.exists(self.test_path):
                    self.classes = sorted([
                        folder for folder in os.listdir(self.test_path)
                        if os.path.isdir(os.path.join(self.test_path, folder))
                        and not folder.startswith('.')
                    ])
                else:
                    self.classes = []
            
            # Obtém resumo do modelo
            train_path = "data/train"
            summary = self.classify_use_case.get_model_summary(
                self.test_path,
                self.image_size,
                train_path=train_path if os.path.exists(train_path) else None
            )
            
            # Atualiza interface
            self.model_summary_text.delete(1.0, tk.END)
            
            if summary:
                self.display_model_summary(summary)
            else:
                self.model_summary_text.insert(tk.END, "✅ Modelo carregado com sucesso!\n")
                self.model_summary_text.insert(tk.END, f"📁 Caminho: {self.model_path}\n")
                self.model_summary_text.insert(tk.END, f"📊 Classes: {', '.join(self.classes) if self.classes else 'Não encontradas'}\n")
            
            self.model_loaded = True
            self.classify_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.model_summary_text.delete(1.0, tk.END)
            self.model_summary_text.insert(tk.END, f"❌ Erro: {e}\n")
            self.model_loaded = False
            self.classify_button.config(state=tk.DISABLED)
    
    def display_model_summary(self, summary: dict):
        """Exibe o resumo do modelo."""
        text = "=" * 70 + "\n"
        text += "📊 RESUMO DO MODELO\n"
        text += "=" * 70 + "\n\n"
        
        text += f"🎯 ACURÁCIA GERAL: {summary['accuracy'] * 100:.2f}%\n\n"
        text += "📈 MÉTRICAS POR CLASSE:\n\n"
        
        classes = summary.get('classes', [])
        for i, class_name in enumerate(classes):
            if i < len(summary['precision']):
                text += f"  {class_name}:\n"
                text += f"    • Taxa de Acerto: {summary['recall'][i] * 100:.2f}% "
                text += f"({int(summary['recall'][i] * summary['support'][i])}/{int(summary['support'][i])} corretas)\n"
                text += f"    • Precisão:       {summary['precision'][i] * 100:.2f}%\n"
                text += f"    • F1-Score:       {summary['f1'][i] * 100:.2f}%\n\n"
        
        text += "=" * 70 + "\n"
        
        self.model_summary_text.insert(tk.END, text)
    
    def browse_file(self):
        """Abre diálogo para seleção de arquivo."""
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem",
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("GIF", "*.gif"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.selected_file_path = file_path
    
    def classify_image(self):
        """Classifica a imagem selecionada."""
        if not self.model_loaded:
            messagebox.showerror("Erro", "Modelo não carregado!")
            return
        
        if not hasattr(self, 'selected_file_path') or not self.selected_file_path:
            messagebox.showerror("Erro", "Por favor, selecione uma imagem primeiro!")
            return
        
        try:
            # Classifica a imagem
            predicted_class, confidence, image_array, heatmap = self.classify_use_case.classify_image_with_classes(
                self.selected_file_path,
                self.classes,
                self.image_size
            )
            
            if predicted_class is None:
                messagebox.showerror("Erro", "Não foi possível classificar a imagem!")
                return
            
            # Atualiza texto de resultado
            self.result_text.delete(1.0, tk.END)
            result_text = f"✅ Classe Predita: {predicted_class}\n"
            result_text += f"📊 Confiança: {confidence * 100:.2f}%\n"
            self.result_text.insert(tk.END, result_text)
            
            # Visualiza imagem com Grad-CAM
            self.visualize_image_with_gradcam(image_array, heatmap, predicted_class, confidence)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao classificar imagem: {e}")
    
    def visualize_image_with_gradcam(
        self,
        image: np.ndarray,
        heatmap: Optional[np.ndarray],
        predicted_class: str,
        confidence: float
    ):
        """Visualiza a imagem com Grad-CAM e círculos destacando áreas importantes."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        if heatmap is not None:
            # Normaliza o heatmap
            heatmap_norm = heatmap.astype(np.float32) / 255.0
            
            # Aplica threshold para destacar apenas as regiões mais importantes
            threshold = 0.4
            heatmap_mask = heatmap_norm > threshold
            
            # Encontra regiões importantes para desenhar círculos
            if np.any(heatmap_mask):
                labeled, num_features = ndimage.label(heatmap_mask)
                if num_features > 0:
                    centers = []
                    for feature_id in range(1, num_features + 1):
                        coords = np.argwhere(labeled == feature_id)
                        if len(coords) > 0:
                            center = coords.mean(axis=0).astype(int)
                            centers.append((center[1], center[0]))  # (x, y)
                    
                    # Ordena por intensidade e pega as 3 maiores regiões
                    centers_sorted = sorted(
                        centers,
                        key=lambda c: heatmap_norm[c[1], c[0]],
                        reverse=True
                    )[:3]
                else:
                    centers_sorted = []
            else:
                centers_sorted = []
            
            # Cria overlay colorido
            overlay = np.zeros_like(image)
            overlay[:, :, 0] = np.where(heatmap_mask, heatmap_norm * 1.5, 0)  # Vermelho
            overlay[:, :, 1] = np.where(heatmap_mask, heatmap_norm * 0.2, 0)  # Pouco verde
            overlay[:, :, 2] = np.where(heatmap_mask, heatmap_norm * 0.1, 0)  # Muito pouco azul
            
            # Combina imagem original com overlay
            overlayed = image * 0.5 + np.clip(overlay, 0, 1) * 0.5
            overlayed = np.clip(overlayed, 0, 1)
            
            # Plota imagem
            ax.imshow(overlayed)
            
            # Desenha círculos nas regiões importantes
            img_height, img_width = image.shape[:2]
            for center_x, center_y in centers_sorted:
                radius = max(20, min(40, int(img_width * 0.12)))
                circle = plt.Circle(
                    (center_x, center_y),
                    radius,
                    fill=False,
                    edgecolor='red',
                    linewidth=4,
                    linestyle='--',
                    alpha=0.9
                )
                ax.add_patch(circle)
        else:
            # Se não há heatmap, mostra apenas a imagem original
            ax.imshow(image)
        
        ax.axis('off')
        title = f"{predicted_class} ({confidence*100:.2f}%)"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def run(self):
        """Inicia a aplicação."""
        self.root.mainloop()


def main():
    """Função principal."""
    app = ClassifyImageApp()
    app.run()


if __name__ == '__main__':
    main()

