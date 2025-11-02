from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class VisualizationRepository(ABC):
    """Interface para visualização de resultados."""
    
    @abstractmethod
    def plot_training_history(self, history: Dict[str, List[float]], save_path: str):
        """Plota gráficos do histórico de treinamento."""
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def plot_test_images_with_dual_gradcam(
        self,
        images: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        class_names: List[str],
        heatmaps_pred: np.ndarray,
        heatmaps_true: np.ndarray,
        save_path: str,
        num_images: int = 9
    ):
        """Plota imagens de teste com overlays Grad-CAM duplos:
        - Vermelho: regiões que fazem o modelo classificar como classe predita
        - Verde: regiões que fazem a imagem pertencer à classe verdadeira"""
        pass
    
    @abstractmethod
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: str
    ):
        """Plota matriz de confusão."""
        pass
    
    @abstractmethod
    def print_class_accuracy_summary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str]
    ):
        """Imprime resumo de acerto por classe e geral."""
        pass
