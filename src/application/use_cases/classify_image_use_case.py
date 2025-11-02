from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageFile

# Habilita carregamento de imagens truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.domain.repositories.model_repository import ModelRepository
from src.domain.repositories.data_loader_repository import DataLoaderRepository


class ClassifyImageUseCase:
    """Caso de uso para classificar uma imagem individual."""
    
    def __init__(
        self,
        model_repository: ModelRepository,
        data_loader: DataLoaderRepository
    ):
        self.model_repository = model_repository
        self.data_loader = data_loader
    
    def load_model(self, model_path: str) -> bool:
        """Carrega o modelo salvo."""
        try:
            self.model_repository.load_model(model_path)
            return True
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return False
    
    def get_model_summary(
        self,
        test_path: str,
        image_size: tuple,
        train_path: str = None
    ) -> Optional[dict]:
        """Obtém resumo do modelo usando dataset de teste."""
        try:
            # Tenta obter classes do diretório de treino se não fornecido
            if train_path is None:
                train_path = "data/train"
            
            # Carrega dataset de teste
            try:
                _, (X_test, y_test), classes = self.data_loader.load_dataset(
                    train_path=train_path,
                    test_path=test_path,
                    image_size=image_size
                )
            except ValueError as e:
                # Se falhar, tenta obter classes diretamente do diretório de teste
                import os
                if not os.path.exists(test_path):
                    return None
                
                classes = sorted([
                    folder for folder in os.listdir(test_path)
                    if os.path.isdir(os.path.join(test_path, folder))
                    and not folder.startswith('.')
                ])
                
                # Carrega manualmente apenas o teste
                X_test_list = []
                y_test_list = []
                
                for class_idx, class_name in enumerate(classes):
                    test_class_path = os.path.join(test_path, class_name)
                    if not os.path.exists(test_class_path):
                        continue
                    
                    X_class, y_class = self.data_loader._load_images_from_folder(
                        test_class_path,
                        image_size,
                        class_idx,
                        len(classes)
                    )
                    
                    if len(X_class) > 0:
                        X_test_list.append(X_class)
                        y_test_list.append(y_class)
                
                if len(X_test_list) == 0:
                    return None
                
                X_test = np.concatenate(X_test_list, axis=0)
                y_test = np.concatenate(y_test_list, axis=0)
            
            # Faz predições
            predictions = self.model_repository.predict(X_test)
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
            
            # Calcula métricas
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
            
            accuracy = accuracy_score(true_classes, pred_classes)
            precision, recall, f1, support = precision_recall_fscore_support(
                true_classes, pred_classes, zero_division=0
            )
            cm = confusion_matrix(true_classes, pred_classes)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'confusion_matrix': cm,
                'classes': classes
            }
        except Exception as e:
            print(f"Erro ao obter resumo do modelo: {e}")
            return None
    
    def load_and_preprocess_image(
        self,
        image_path: str,
        image_size: tuple
    ) -> Optional[np.ndarray]:
        """Carrega e preprocessa uma imagem."""
        try:
            # Verifica se o arquivo existe e não está vazio
            import os
            if not os.path.isfile(image_path) or os.path.getsize(image_path) == 0:
                return None
            
            # Carrega imagem
            img = Image.open(image_path)
            img.load()  # Força o carregamento completo
            img = img.convert('RGB')
            
            # Verifica se a imagem tem tamanho válido
            if img.size[0] == 0 or img.size[1] == 0:
                return None
            
            # Redimensiona
            img = img.resize(image_size, Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0
            
            # Verifica se o array tem as dimensões corretas
            if img_array.shape != (*image_size, 3):
                return None
            
            return img_array
        except Exception as e:
            print(f"Erro ao carregar imagem: {e}")
            return None
    
    def classify_image(
        self,
        image_path: str,
        image_size: tuple = (224, 224)
    ) -> Tuple[Optional[str], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Classifica uma imagem.
        Retorna: (classe_predita, confianca, imagem_preprocessada, heatmap)
        """
        # Carrega e preprocessa a imagem
        image_array = self.load_and_preprocess_image(image_path, image_size)
        if image_array is None:
            return None, None, None, None
        
        # Faz predição
        predictions = self.model_repository.predict(np.expand_dims(image_array, axis=0))
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        # Obtém nomes das classes (assumindo que estão no modelo ou precisamos passar)
        # Por enquanto, vamos usar índices genéricos
        predicted_class = f"Classe_{class_idx + 1}"
        
        # Gera heatmap Grad-CAM
        heatmap = None
        try:
            heatmap = self.model_repository.get_gradcam_heatmap(image_array, class_idx)
        except Exception as e:
            print(f"Aviso: Não foi possível gerar heatmap Grad-CAM: {e}")
        
        return predicted_class, confidence, image_array, heatmap
    
    def classify_image_with_classes(
        self,
        image_path: str,
        classes: list,
        image_size: tuple = (224, 224)
    ) -> Tuple[Optional[str], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Classifica uma imagem com nomes de classes específicos.
        Retorna: (classe_predita, confianca, imagem_preprocessada, heatmap)
        """
        # Carrega e preprocessa a imagem
        image_array = self.load_and_preprocess_image(image_path, image_size)
        if image_array is None:
            return None, None, None, None
        
        # Faz predição
        predictions = self.model_repository.predict(np.expand_dims(image_array, axis=0))
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        # Obtém nome da classe
        if class_idx < len(classes):
            predicted_class = classes[class_idx]
        else:
            predicted_class = f"Classe_{class_idx + 1}"
        
        # Gera heatmap Grad-CAM
        heatmap = None
        try:
            heatmap = self.model_repository.get_gradcam_heatmap(image_array, class_idx)
        except Exception as e:
            print(f"Aviso: Não foi possível gerar heatmap Grad-CAM: {e}")
        
        return predicted_class, confidence, image_array, heatmap

