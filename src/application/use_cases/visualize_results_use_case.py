import os
from typing import List

import numpy as np
from PIL import Image, ImageFile

# Habilita carregamento de imagens truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        # Usa as classes do training_result
        classes = training_result.classes
        
        # Carrega dados de teste diretamente do test_path
        # Como já temos as classes do training_result, podemos carregar apenas as imagens de teste
        
        X_test_list = []
        y_test_list = []
        
        num_classes = training_result.num_classes
        
        for class_idx, class_name in enumerate(classes):
            test_class_path = os.path.join(test_path, class_name)
            
            if not os.path.exists(test_class_path):
                continue
            
            images = []
            labels = []
            
            for filename in os.listdir(test_class_path):
                filename_lower = filename.lower()
                if not filename_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    continue
                
                img_path = os.path.join(test_class_path, filename)
                
                # Verifica se é um arquivo válido e não está vazio
                try:
                    if not os.path.isfile(img_path) or os.path.getsize(img_path) == 0:
                        continue
                except:
                    continue
                
                try:
                    # Carrega imagem com tratamento para imagens truncadas
                    img = Image.open(img_path)
                    img.load()  # Força o carregamento completo
                    img = img.convert('RGB')
                    
                    # Verifica se a imagem tem tamanho válido
                    if img.size[0] == 0 or img.size[1] == 0:
                        continue
                    
                    img = img.resize(image_size, Image.Resampling.LANCZOS)
                    img_array = np.array(img) / 255.0
                    
                    # Verifica se o array tem as dimensões corretas
                    if img_array.shape != (*image_size, 3):
                        continue
                    
                    images.append(img_array)
                    
                    label = np.zeros(num_classes)
                    label[class_idx] = 1
                    labels.append(label)
                except Exception as e:
                    print(f"Aviso: Erro ao carregar imagem {filename}: {str(e)[:100]}")
                    continue
            
            if len(images) > 0:
                X_test_list.append(np.array(images))
                y_test_list.append(np.array(labels))
        
        X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.array([])
        y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.array([])
        
        if len(X_test) > 0:
            indices = np.random.permutation(len(X_test))
            X_test = X_test[indices]
            y_test = y_test[indices]
        
        # Carrega o modelo salvo
        self.model_repository.load_model(training_result.model_path)
        
        # Faz predições
        predictions = self.model_repository.predict(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        
        # Exibe resumo de acerto por classe e geral ANTES de plotar os gráficos
        print("\n" + "="*70)
        print("GERANDO GRÁFICOS E CALCULANDO MÉTRICAS DE ACERTO")
        print("="*70)
        
        self.visualization_repository.print_class_accuracy_summary(
            true_classes,
            pred_classes,
            classes
        )
        
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
        gradcam_success = True
        for i, image in enumerate(selected_images):
            try:
                class_idx = selected_pred[i]
                heatmap = self.model_repository.get_gradcam_heatmap(
                    image,
                    class_idx
                )
                heatmaps.append(heatmap)
            except Exception as e:
                print(f"Aviso: Erro ao gerar Grad-CAM para imagem {i}: {e}")
                gradcam_success = False
                # Cria um heatmap vazio para manter o tamanho consistente
                heatmaps.append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))
        
        if not gradcam_success:
            print("Aviso: Alguns heatmaps Grad-CAM falharam. Gerando visualização sem overlay.")
            heatmaps = None
        else:
            heatmaps = np.array(heatmaps)
        
        # Plota imagens com ou sem Grad-CAM
        if heatmaps is not None:
            self.visualization_repository.plot_test_images_with_gradcam(
                selected_images,
                selected_true,
                selected_pred,
                classes,
                heatmaps,
                save_path="results/test_images_gradcam.png",
                num_images=len(selected_images)
            )
        else:
            # Plota apenas as imagens sem Grad-CAM
            self.visualization_repository.plot_test_images_without_gradcam(
                selected_images,
                selected_true,
                selected_pred,
                classes,
                save_path="results/test_images_gradcam.png",
                num_images=len(selected_images)
            )
