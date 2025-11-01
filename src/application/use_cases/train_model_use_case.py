from typing import Tuple

from src.domain.entities.dataset import Dataset
from src.domain.entities.training_result import TrainingResult
from src.domain.repositories.data_loader_repository import DataLoaderRepository
from src.domain.repositories.model_repository import ModelRepository


class TrainModelUseCase:
    """Caso de uso para treinar o modelo."""
    
    def __init__(
        self,
        data_loader: DataLoaderRepository,
        model_repository: ModelRepository
    ):
        self.data_loader = data_loader
        self.model_repository = model_repository
    
    def execute(
        self,
        dataset: Dataset,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        learning_rate: float = 0.001
    ) -> TrainingResult:
        """Executa o treinamento do modelo."""
        # Carrega os dados
        (X_train, y_train), (X_test, y_test), classes = self.data_loader.load_dataset(
            dataset.train_path,
            dataset.test_path,
            dataset.image_size
        )
        
        # Cria o modelo
        input_shape = (*dataset.image_size, 3)
        self.model_repository.create_model(
            num_classes=dataset.num_classes,
            input_shape=input_shape,
            learning_rate=learning_rate
        )
        
        # Divide dados de validação
        val_size = int(len(X_train) * validation_split)
        X_val = X_train[:val_size]
        y_val = y_train[:val_size]
        X_train_split = X_train[val_size:]
        y_train_split = y_train[val_size:]
        
        # Treina o modelo
        history = self.model_repository.train(
            X_train_split,
            y_train_split,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.0  # Já fizemos a divisão manualmente
        )
        
        # Avalia o modelo
        test_loss, test_accuracy = self.model_repository.evaluate(X_test, y_test)
        
        # Salva o modelo
        model_path = "models/cnn_model.h5"
        self.model_repository.save_model(model_path)
        
        # Extrai métricas finais do histórico
        final_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1] if 'val_loss' in history else test_loss
        final_accuracy = history['accuracy'][-1]
        final_val_accuracy = history['val_accuracy'][-1] if 'val_accuracy' in history else test_accuracy
        
        return TrainingResult(
            history=history,
            model_path=model_path,
            accuracy=final_accuracy,
            loss=final_loss,
            validation_accuracy=test_accuracy,
            validation_loss=test_loss,
            num_classes=dataset.num_classes,
            classes=classes
        )
