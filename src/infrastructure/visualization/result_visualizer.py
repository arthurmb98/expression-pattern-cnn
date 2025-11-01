from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2

from src.domain.repositories.visualization_repository import VisualizationRepository


class ResultVisualizer(VisualizationRepository):
    """Implementação concreta do visualizador de resultados."""
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: str):
        """Plota gráficos do histórico de treinamento."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfico de Loss
        axes[0].plot(history['loss'], label='Treino', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validação', linewidth=2)
        axes[0].set_title('Loss do Modelo', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Época', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico de Accuracy
        axes[1].plot(history['accuracy'], label='Treino', linewidth=2)
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Validação', linewidth=2)
        axes[1].set_title('Acurácia do Modelo', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('Acurácia', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de histórico salvo em: {save_path}")
    
    def plot_test_images_with_gradcam(
        self,
        images: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        class_names: List[str],
        heatmaps: np.ndarray,
        save_path: str,
        num_images: int = 9
    ):
        """Plota imagens de teste com overlays Grad-CAM."""
        # Calcula grid size
        n_cols = 3
        n_rows = (num_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i in range(num_images):
            if i >= len(images):
                axes[i].axis('off')
                continue
            
            image = images[i]
            heatmap = heatmaps[i]
            true_label = true_labels[i]
            pred_label = pred_labels[i]
            
            # Aplica colormap ao heatmap
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
            
            # Combina imagem original com heatmap
            overlayed = image * 0.6 + heatmap_colored * 0.4
            
            # Plota
            axes[i].imshow(overlayed)
            axes[i].axis('off')
            
            # Adiciona título com informações
            true_class = class_names[true_label] if true_label < len(class_names) else f"Classe {true_label}"
            pred_class = class_names[pred_label] if pred_label < len(class_names) else f"Classe {pred_label}"
            
            color = 'green' if true_label == pred_label else 'red'
            title = f"Verdadeiro: {true_class}\nPredito: {pred_class}"
            axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
        
        # Remove eixos vazios
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Imagens com Grad-CAM salvas em: {save_path}")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: str
    ):
        """Plota matriz de confusão."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Quantidade'}
        )
        plt.title('Matriz de Confusão', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Verdadeiro', fontsize=12)
        plt.xlabel('Predito', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Matriz de confusão salva em: {save_path}")
        
        # Imprime relatório de classificação
        print("\n" + "="*50)
        print("Relatório de Classificação")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=class_names))
        print("="*50)
