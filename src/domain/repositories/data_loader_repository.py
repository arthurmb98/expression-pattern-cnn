from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class DataLoaderRepository(ABC):
    """Interface para carregamento de dados."""
    
    @abstractmethod
    def load_dataset(
        self, 
        train_path: str, 
        test_path: str, 
        image_size: Tuple[int, int]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], List[str]]:
        """
        Carrega o dataset de treino e teste.
        
        Returns:
            Tuple contendo:
            - (X_train, y_train): dados de treino
            - (X_test, y_test): dados de teste
            - classes: lista de nomes das classes
        """
        pass
    
    @abstractmethod
    def get_class_names(self, train_path: str) -> List[str]:
        """Retorna os nomes das classes baseado nas pastas de treinamento."""
        pass
