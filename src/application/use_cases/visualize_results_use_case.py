from typing import List

import numpy as np

from src.domain.entities.training_result import TrainingResult
from src.domain.repositories.data_loader_repository import DataLoaderRepository
from src.domain.repositories.model_repository import ModelRepository
from src.domain.repositories.visualization_repository import VisualizationRepository


class VisualizeResultsUseCase:
    """Caso de uso para visualizar resultados."""
    
    def __init__(
        self,
        data_loader: DataLoaderRepository,
        model_repository: ModelRepository,
        visualization_repository: VisualizationRepository
    ):
        self.data_loader = data_loader
        self.model_repository = model_repository
        self.visualization_repository = visualization_repository
    
    def execute(
        self,
        training_result: TrainingResult,
        test_path: str,
        image_size: tuple,
        num_images_to_plot: int = 9
    ):
        """Executa a visualização dos resultados."""
        # Carrega dados de teste
        (X_train, y_train), (X_test, y_test), classes = self.data_loader.load_dataset(
            "",  # Não precisamos dos dados de treino aqui
            test_path,
            image_size
        )
        
        # Carrega o modelo salvo
        self.model_repository.load_model(training_result.model_path)
        
        # Faz predições
        predictions = self.model_repository.predict(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        
        # Plota histórico de treinamento
        self.visualization_repository.plot_training_history(
            training_result.history,
            save_path="results/training_history.png"
        )
        
        # Plota matriz de confusão
        self.visualization_repository.plot_confusion_matrix(
            true_classes,
            pred_classes,
            classes,
            save_path="results/confusion_matrix.png"
        )
        
        # Gera heatmaps Grad-CAM para algumas imagens
        selected_indices = np.random.choice(
            len(X_test),
            size=min(num_images_to_plot, len(X_test)),
            replace=False
        )
        
        selected_images = X_test[selected_indices]
        selected_true = true_classes[selected_indices]
        selected_pred = pred_classes[selected_indices]
        
        heatmaps = []
        for i, image in enumerate(selected_images):
            class_idx = selected_pred[i]
            heatmap = self.model_repository.get_gradcam_heatmap(
                image,
                class_idx
            )
            heatmaps.append(heatmap)
        
        heatmaps = np.array(heatmaps)
        
        # Plota imagens com Grad-CAM
        self.visualization_repository.plot_test_images_with_gradcam(
            selected_images,
            selected_true,
            selected_pred,
            classes,
            heatmaps,
            save_path="results/test_images_gradcam.png",
            num_images=len(selected_images)
        )
