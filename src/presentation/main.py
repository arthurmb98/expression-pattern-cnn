import argparse
import os
import sys

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.application.use_cases.train_model_use_case import TrainModelUseCase
from src.application.use_cases.visualize_results_use_case import VisualizeResultsUseCase
from src.domain.entities.dataset import Dataset
from src.infrastructure.data_loader.image_data_loader import ImageDataLoader
from src.infrastructure.models.cnn_model import CNNModel
from src.infrastructure.visualization.result_visualizer import ResultVisualizer


def validate_paths(train_path: str, test_path: str):
    """Valida os caminhos fornecidos."""
    if not os.path.exists(train_path):
        raise ValueError(f"Diretório de treinamento não encontrado: {train_path}")
    
    if not os.path.exists(test_path):
        raise ValueError(f"Diretório de teste não encontrado: {test_path}")
    
    # Verifica se há pelo menos uma pasta em treinamento
    train_folders = [
        f for f in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, f)) and not f.startswith('.')
    ]
    
    if len(train_folders) == 0:
        raise ValueError(f"Nenhuma classe encontrada em: {train_path}")
    
    # Verifica se todas as pastas de treino têm correspondente em teste
    missing_folders = []
    for folder in train_folders:
        test_folder_path = os.path.join(test_path, folder)
        if not os.path.exists(test_folder_path):
            missing_folders.append(folder)
    
    if missing_folders:
        raise ValueError(
            f"As seguintes classes de treino não têm pasta correspondente em teste: {missing_folders}"
        )
    
    print(f"✓ Validação concluída: {len(train_folders)} classes encontradas")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Treina uma CNN para classificação de imagens usando clean architecture'
    )
    parser.add_argument(
        '--train_path',
        type=str,
        default='data/train',
        help='Caminho para o diretório de treinamento (deve conter subpastas com nomes das classes) (default: data/train)'
    )
    parser.add_argument(
        '--test_path',
        type=str,
        default='data/test',
        help='Caminho para o diretório de teste (deve conter subpastas com mesmos nomes das classes) (default: data/test)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número de épocas de treinamento (default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Tamanho do batch (default: 32)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[224, 224],
        help='Tamanho das imagens (altura largura) (default: 224 224)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Taxa de aprendizado (default: 0.001)'
    )
    parser.add_argument(
        '--num_images_to_plot',
        type=int,
        default=9,
        help='Número de imagens de teste para plotar com Grad-CAM (default: 9)'
    )
    
    args = parser.parse_args()
    
    try:
        # Valida caminhos
        validate_paths(args.train_path, args.test_path)
        
        # Cria diretórios necessários
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/checkpoints', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Inicializa componentes
        data_loader = ImageDataLoader()
        model_repository = CNNModel()
        visualization_repository = ResultVisualizer()
        
        # Obtém classes
        classes = data_loader.get_class_names(args.train_path)
        num_classes = len(classes)
        
        # Cria entidade Dataset
        dataset = Dataset(
            train_path=args.train_path,
            test_path=args.test_path,
            classes=classes,
            num_classes=num_classes,
            image_size=tuple(args.image_size)
        )
        
        print(f"\n{'='*60}")
        print("CONFIGURAÇÃO DO TREINAMENTO")
        print(f"{'='*60}")
        print(f"Classes: {classes}")
        print(f"Número de classes: {num_classes}")
        print(f"Tamanho da imagem: {dataset.image_size}")
        print(f"Épocas: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"{'='*60}\n")
        
        # Caso de uso: Treinar modelo
        train_use_case = TrainModelUseCase(data_loader, model_repository)
        training_result = train_use_case.execute(
            dataset=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        print(f"\n{'='*60}")
        print("RESULTADOS DO TREINAMENTO")
        print(f"{'='*60}")
        print(f"✓ Acurácia de treino:   {training_result.accuracy:.4f} ({training_result.accuracy*100:.2f}%)")
        print(f"✓ Loss de treino:       {training_result.loss:.4f}")
        print(f"✓ Acurácia de teste:     {training_result.validation_accuracy:.4f} ({training_result.validation_accuracy*100:.2f}%)")
        print(f"✓ Loss de teste:        {training_result.validation_loss:.4f}")
        print(f"✓ Modelo salvo em:      {training_result.model_path}")
        print(f"{'='*60}\n")
        
        # Caso de uso: Visualizar resultados
        visualize_use_case = VisualizeResultsUseCase(
            data_loader,
            model_repository,
            visualization_repository
        )
        
        visualize_use_case.execute(
            training_result=training_result,
            test_path=args.test_path,
            image_size=dataset.image_size,
            num_images_to_plot=args.num_images_to_plot
        )
        
        print(f"\n{'='*60}")
        print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
        print(f"{'='*60}")
        print(f"\n📁 Arquivos gerados:")
        print(f"  ✓ Histórico de treinamento:  results/training_history.png")
        print(f"  ✓ Matriz de confusão:       results/confusion_matrix.png")
        print(f"  ✓ Imagens de teste:         results/test_images_gradcam.png")
        print(f"  ✓ Modelo treinado:          models/cnn_model.h5")
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
