import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFile

# Habilita carregamento de imagens truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.domain.repositories.data_loader_repository import DataLoaderRepository


class ImageDataLoader(DataLoaderRepository):
    """Implementação concreta do carregador de dados."""
    
    def get_class_names(self, train_path: str) -> List[str]:
        """Retorna os nomes das classes baseado nas pastas de treinamento."""
        if not os.path.exists(train_path):
            raise ValueError(f"Diretório de treinamento não encontrado: {train_path}")
        
        classes = [
            folder for folder in os.listdir(train_path)
            if os.path.isdir(os.path.join(train_path, folder))
            and not folder.startswith('.')
        ]
        
        return sorted(classes)
    
    def _load_images_from_folder(
        self,
        folder_path: str,
        image_size: Tuple[int, int],
        class_idx: int,
        num_classes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Carrega imagens de uma pasta específica."""
        images = []
        labels = []
        
        if not os.path.exists(folder_path):
            return np.array([]), np.array([])
        
        for filename in os.listdir(folder_path):
            # Verifica extensões JPEG e PNG (case-insensitive)
            filename_lower = filename.lower()
            if not filename_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
                
            img_path = os.path.join(folder_path, filename)
            
            # Verifica se é um arquivo válido e não está vazio
            try:
                if not os.path.isfile(img_path) or os.path.getsize(img_path) == 0:
                    continue
            except:
                continue
            
            try:
                # Carrega imagem com tratamento para imagens truncadas
                # O ImageFile.LOAD_TRUNCATED_IMAGES já foi definido no topo do módulo
                img = Image.open(img_path)
                
                # Força o carregamento completo da imagem
                img.load()
                
                # Converte para RGB (necessário para JPEG com canais diferentes)
                img = img.convert('RGB')
                
                # Verifica se a imagem tem tamanho válido
                if img.size[0] == 0 or img.size[1] == 0:
                    print(f"Aviso: Imagem com tamanho inválido ignorada: {img_path}")
                    continue
                
                # Redimensiona a imagem
                img = img.resize(image_size, Image.Resampling.LANCZOS)
                img_array = np.array(img) / 255.0  # Normaliza entre 0 e 1
                
                # Verifica se o array tem as dimensões corretas
                if img_array.shape != (*image_size, 3):
                    print(f"Aviso: Imagem com dimensões incorretas ignorada: {img_path}")
                    continue
                
                images.append(img_array)
                
                # Cria label one-hot
                label = np.zeros(num_classes)
                label[class_idx] = 1
                labels.append(label)
                
            except Exception as e:
                print(f"Aviso: Erro ao carregar imagem {filename}: {str(e)[:100]}")
                continue
        
        return np.array(images), np.array(labels)
    
    def load_dataset(
        self,
        train_path: str,
        test_path: str,
        image_size: Tuple[int, int]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], List[str]]:
        """Carrega o dataset de treino e teste."""
        # Obtém classes do diretório de treino
        classes = self.get_class_names(train_path)
        num_classes = len(classes)
        
        if num_classes == 0:
            raise ValueError("Nenhuma classe encontrada no diretório de treinamento")
        
        print(f"Classes encontradas: {classes}")
        print(f"Número de classes: {num_classes}")
        
        # Carrega imagens de treino
        X_train_list = []
        y_train_list = []
        
        for class_idx, class_name in enumerate(classes):
            train_class_path = os.path.join(train_path, class_name)
            X_class, y_class = self._load_images_from_folder(
                train_class_path,
                image_size,
                class_idx,
                num_classes
            )
            
            if len(X_class) > 0:
                X_train_list.append(X_class)
                y_train_list.append(y_class)
                print(f"Treino - {class_name}: {len(X_class)} imagens")
            else:
                print(f"Aviso: Nenhuma imagem encontrada em {train_class_path}")
        
        # Carrega imagens de teste
        X_test_list = []
        y_test_list = []
        
        for class_idx, class_name in enumerate(classes):
            test_class_path = os.path.join(test_path, class_name)
            
            if not os.path.exists(test_class_path):
                print(f"Aviso: Diretório de teste não encontrado: {test_class_path}")
                continue
            
            X_class, y_class = self._load_images_from_folder(
                test_class_path,
                image_size,
                class_idx,
                num_classes
            )
            
            if len(X_class) > 0:
                X_test_list.append(X_class)
                y_test_list.append(y_class)
                print(f"Teste - {class_name}: {len(X_class)} imagens")
            else:
                print(f"Aviso: Nenhuma imagem encontrada em {test_class_path}")
        
        # Concatena todos os dados
        X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.array([])
        y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.array([])
        
        X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.array([])
        y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.array([])
        
        # Embaralha os dados
        if len(X_train) > 0:
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]
        
        if len(X_test) > 0:
            indices = np.random.permutation(len(X_test))
            X_test = X_test[indices]
            y_test = y_test[indices]
        
        print(f"\nTotal de imagens de treino: {len(X_train)}")
        print(f"Total de imagens de teste: {len(X_test)}")
        
        return (X_train, y_train), (X_test, y_test), classes
