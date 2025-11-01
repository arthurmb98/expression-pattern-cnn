from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class ModelRepository(ABC):
    """Interface para o modelo de CNN."""
    
    @abstractmethod
    def create_model(
        self, 
        num_classes: int, 
        input_shape: Tuple[int, int, int],
        learning_rate: float = 0.001
    ):
        """Cria um modelo CNN."""
        pass
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """Treina o modelo."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, float]:
        """Avalia o modelo. Retorna (loss, accuracy)."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições."""
        pass
    
    @abstractmethod
    def save_model(self, path: str):
        """Salva o modelo."""
        pass
    
    @abstractmethod
    def load_model(self, path: str):
        """Carrega o modelo."""
        pass
    
    @abstractmethod
    def get_gradcam_heatmap(
        self,
        image: np.ndarray,
        class_idx: int,
        layer_name: str = None
    ) -> np.ndarray:
        """Gera heatmap Grad-CAM para visualização."""
        pass
