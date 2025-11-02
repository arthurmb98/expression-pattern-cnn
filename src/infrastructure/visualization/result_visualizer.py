from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
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
        
        # Exibe o gráfico na tela
        print(f"\n📊 Exibindo gráfico de histórico de treinamento...")
        print(f"   (Feche a janela para continuar)")
        plt.show(block=True)
        plt.close()
        
        print(f"✓ Gráfico de histórico salvo em: {save_path}")
    
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
        
        # Exibe o gráfico na tela
        print(f"\n📊 Exibindo imagens de teste com Grad-CAM...")
        print(f"   (Feche a janela para continuar)")
        plt.show(block=True)
        plt.close()
        
        print(f"✓ Imagens com Grad-CAM salvas em: {save_path}")
    
    def plot_test_images_with_dual_gradcam(
        self,
        images: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        class_names: List[str],
        heatmaps_pred: np.ndarray,
        heatmaps_true: np.ndarray,
        save_path: str,
        num_images: int = 8
    ):
        """Plota imagens de teste com overlays Grad-CAM duplos:
        - Vermelho: regiões que fazem o modelo classificar como classe predita
        - Verde: regiões que fazem a imagem pertencer à classe verdadeira"""
        # Calcula grid size para 8 imagens (2x4 ou 4x2)
        n_cols = 4
        n_rows = 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i in range(num_images):
            if i >= len(images):
                axes[i].axis('off')
                continue
            
            image = images[i]
            heatmap_pred = heatmaps_pred[i]  # Heatmap da classe predita (vermelho)
            heatmap_true = heatmaps_true[i]  # Heatmap da classe verdadeira (verde)
            true_label = true_labels[i]
            pred_label = pred_labels[i]
            
            # Normaliza os heatmaps para valores entre 0 e 1
            heatmap_pred_norm = heatmap_pred.astype(np.float32) / 255.0
            heatmap_true_norm = heatmap_true.astype(np.float32) / 255.0
            
            # Aplica threshold para destacar apenas as regiões mais importantes
            threshold = 0.4
            heatmap_pred_mask = heatmap_pred_norm > threshold
            heatmap_true_mask = heatmap_true_norm > threshold
            
            # Encontra regiões importantes para desenhar círculos
            from scipy import ndimage
            
            # Para heatmap predito (vermelho) - encontra centro das regiões importantes
            if np.any(heatmap_pred_mask):
                labeled_pred, num_features_pred = ndimage.label(heatmap_pred_mask)
                if num_features_pred > 0:
                    centers_pred = []
                    for feature_id in range(1, num_features_pred + 1):
                        coords = np.argwhere(labeled_pred == feature_id)
                        if len(coords) > 0:
                            center = coords.mean(axis=0).astype(int)
                            centers_pred.append((center[1], center[0]))  # (x, y)
            else:
                centers_pred = []
            
            # Para heatmap verdadeiro (verde) - encontra centro das regiões importantes
            if np.any(heatmap_true_mask):
                labeled_true, num_features_true = ndimage.label(heatmap_true_mask)
                if num_features_true > 0:
                    centers_true = []
                    for feature_id in range(1, num_features_true + 1):
                        coords = np.argwhere(labeled_true == feature_id)
                        if len(coords) > 0:
                            center = coords.mean(axis=0).astype(int)
                            centers_true.append((center[1], center[0]))  # (x, y)
            else:
                centers_true = []
            
            # Cria overlays coloridos mais vibrantes
            # Vermelho para classe predita
            overlay_pred = np.zeros_like(image)
            overlay_pred[:, :, 0] = np.where(heatmap_pred_mask, heatmap_pred_norm * 1.5, 0)  # Canal vermelho intensificado
            overlay_pred[:, :, 1] = np.where(heatmap_pred_mask, heatmap_pred_norm * 0.2, 0)  # Pouco verde
            overlay_pred[:, :, 2] = np.where(heatmap_pred_mask, heatmap_pred_norm * 0.1, 0)  # Muito pouco azul
            
            # Verde para classe verdadeira
            overlay_true = np.zeros_like(image)
            overlay_true[:, :, 0] = np.where(heatmap_true_mask, heatmap_true_norm * 0.2, 0)  # Pouco vermelho
            overlay_true[:, :, 1] = np.where(heatmap_true_mask, heatmap_true_norm * 1.5, 0)  # Canal verde intensificado
            overlay_true[:, :, 2] = np.where(heatmap_true_mask, heatmap_true_norm * 0.1, 0)  # Muito pouco azul
            
            # Combina imagem original com ambos os overlays
            overlayed = image * 0.5 + np.clip(overlay_pred, 0, 1) * 0.5 + np.clip(overlay_true, 0, 1) * 0.5
            overlayed = np.clip(overlayed, 0, 1)  # Garante valores entre 0 e 1
            
            # Plota
            axes[i].imshow(overlayed)
            axes[i].axis('off')
            
            # Desenha círculos nas regiões importantes
            img_height, img_width = image.shape[:2]
            
            # Círculos vermelhos para classe predita (apenas as 2 maiores regiões)
            if len(centers_pred) > 0:
                # Ordena por tamanho da região (aproximado)
                centers_sorted = sorted(centers_pred, key=lambda c: heatmap_pred_norm[c[1], c[0]], reverse=True)[:2]
                for center_x, center_y in centers_sorted:
                    # Calcula raio baseado no tamanho da região
                    radius = max(15, min(30, int(img_width * 0.1)))
                    circle = plt.Circle((center_x, center_y), radius, fill=False, 
                                      edgecolor='red', linewidth=3, linestyle='--', alpha=0.8)
                    axes[i].add_patch(circle)
            
            # Círculos verdes para classe verdadeira (apenas as 2 maiores regiões)
            if len(centers_true) > 0:
                centers_sorted = sorted(centers_true, key=lambda c: heatmap_true_norm[c[1], c[0]], reverse=True)[:2]
                for center_x, center_y in centers_sorted:
                    radius = max(15, min(30, int(img_width * 0.1)))
                    circle = plt.Circle((center_x, center_y), radius, fill=False, 
                                      edgecolor='green', linewidth=3, linestyle='-', alpha=0.8)
                    axes[i].add_patch(circle)
            
            # Adiciona título com informações
            true_class = class_names[true_label] if true_label < len(class_names) else f"Classe {true_label}"
            pred_class = class_names[pred_label] if pred_label < len(class_names) else f"Classe {pred_label}"
            
            color = 'green' if true_label == pred_label else 'red'
            title = f"Verdadeiro: {true_class}\nPredito: {pred_class}\n🔴Vermelho=Predita | 🟢Verde=Verdadeira"
            axes[i].set_title(title, fontsize=9, color=color, fontweight='bold')
        
        # Remove eixos vazios
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Exibe o gráfico na tela
        print(f"\n📊 Exibindo imagens de teste com Grad-CAM duplo...")
        print(f"   🔴 Vermelho: Regiões da classe PREDITA")
        print(f"   🟢 Verde: Regiões da classe VERDADEIRA")
        print(f"   (Feche a janela para continuar)")
        plt.show(block=True)
        plt.close()
        
        print(f"✓ Imagens com Grad-CAM duplo salvas em: {save_path}")
    
    def plot_test_images_without_gradcam(
        self,
        images: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        class_names: List[str],
        save_path: str,
        num_images: int = 9
    ):
        """Plota imagens de teste sem overlays Grad-CAM."""
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
            true_label = true_labels[i]
            pred_label = pred_labels[i]
            
            # Plota imagem original
            axes[i].imshow(image)
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
        
        # Exibe o gráfico na tela
        print(f"\n📊 Exibindo imagens de teste...")
        print(f"   (Feche a janela para continuar)")
        plt.show(block=True)
        plt.close()
        
        print(f"✓ Imagens de teste salvas em: {save_path}")
    
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
        
        # Exibe o gráfico na tela
        print(f"\n📊 Exibindo matriz de confusão...")
        print(f"   (Feche a janela para continuar)")
        plt.show(block=True)
        plt.close()
        
        print(f"✓ Matriz de confusão salva em: {save_path}")
    
    def print_class_accuracy_summary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str]
    ):
        """Imprime resumo de acerto por classe e geral."""
        # Calcula métricas
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )
        
        # Calcula acerto por classe (taxa de acerto = recall para cada classe)
        cm = confusion_matrix(y_true, y_pred)
        
        print("\n" + "="*70)
        print("RESUMO DE ACERTO POR CLASSE")
        print("="*70)
        
        # Calcula taxa de acerto por classe (quantos corretos / total de cada classe)
        for i, class_name in enumerate(class_names):
            if i < len(cm):
                correct = cm[i, i] if i < len(cm[i]) else 0
                total = sum(cm[i]) if i < len(cm) else 0
                class_accuracy = (correct / total * 100) if total > 0 else 0.0
                
                precision_val = precision[i] * 100 if i < len(precision) else 0.0
                recall_val = recall[i] * 100 if i < len(recall) else 0.0
                f1_val = f1[i] * 100 if i < len(f1) else 0.0
                support_val = int(support[i]) if i < len(support) else 0
                
                print(f"\n📌 {class_name}:")
                print(f"   ✓ Taxa de Acerto:    {recall_val:.2f}%  ({correct}/{total} corretas)")
                print(f"   ✓ Precisão:           {precision_val:.2f}%")
                print(f"   ✓ F1-Score:           {f1_val:.2f}%")
                print(f"   ✓ Total de amostras:  {support_val}")
        
        print("\n" + "="*70)
        print(f"🎯 ACURÁCIA GERAL DO MODELO: {accuracy * 100:.2f}%")
        print("="*70)
        
        # Imprime relatório completo
        print("\n" + "="*70)
        print("RELATÓRIO DETALHADO DE CLASSIFICAÇÃO")
        print("="*70)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        print("="*70 + "\n")
