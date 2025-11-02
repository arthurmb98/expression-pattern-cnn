# CNN para Classificação de Imagens - Clean Architecture

Projeto de CNN (Convolutional Neural Network) em Python para classificação de imagens, implementado seguindo os princípios de Clean Architecture.

## Características

- ✅ Arquitetura limpa com separação de camadas (Domain, Application, Infrastructure, Presentation)
- ✅ Suporta N classes dinamicamente baseado nas pastas fornecidas
- ✅ Aceita diferentes tamanhos de massa de treinamento e teste
- ✅ Gera gráficos de treinamento, matriz de confusão e visualizações com Grad-CAM
- ✅ Visualização de áreas importantes nas imagens que influenciam a classificação

## Estrutura do Projeto

```
expression-pattern-cnn/
├── src/
│   ├── domain/                  # Camada de domínio
│   │   ├── entities/           # Entidades do domínio
│   │   └── repositories/       # Interfaces (contratos)
│   ├── application/            # Camada de aplicação
│   │   └── use_cases/          # Casos de uso
│   ├── infrastructure/         # Camada de infraestrutura
│   │   ├── data_loader/        # Implementação do carregador de dados
│   │   ├── models/             # Implementação do modelo CNN
│   │   └── visualization/      # Implementação do visualizador
│   └── presentation/           # Camada de apresentação
│       └── main.py             # CLI principal
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.8+
- TensorFlow 2.13+
- NumPy, Matplotlib, Scikit-learn, Pillow, OpenCV

## Instalação

1. Clone o repositório:
```bash
git clone <repo-url>
cd expression-pattern-cnn
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Organização dos Dados

O projeto espera que os dados estejam organizados da seguinte forma:

```
data/
├── train/
│   ├── classe1/
│   │   ├── imagem1.jpg
│   │   ├── imagem2.jpg
│   │   └── ...
│   ├── classe2/
│   │   ├── imagem1.jpg
│   │   └── ...
│   └── classeN/
│       └── ...
└── test/
    ├── classe1/
    │   ├── imagem1.jpg
    │   └── ...
    ├── classe2/
    │   └── ...
    └── classeN/
        └── ...
```

**Importante:**
- Cada pasta em `train/` deve ter uma pasta correspondente em `test/` com o mesmo nome
- As massas de treinamento e teste não precisam ter o mesmo tamanho
- O número de classes é determinado automaticamente pela quantidade de pastas

## Uso

### Comando Principal (Recomendado)

O comando mais simples para rodar a aplicação com todos os valores padrão:

```bash
python -m src.presentation.main
```

ou no macOS/Linux:

```bash
python3 -m src.presentation.main
```

**Valores padrão usados:**
- Diretório de treinamento: `data/train`
- Diretório de teste: `data/test`
- Épocas: 50
- Batch size: 32
- Tamanho da imagem: 224x224
- Learning rate: 0.001
- Número de imagens para visualização: 9

### Personalizando os Parâmetros

Se você quiser personalizar algum parâmetro, pode especificá-los:

```bash
python -m src.presentation.main \
    --train_path data/train \
    --test_path data/test \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0001
```

### Exemplos Adicionais

**Exemplo com mais épocas:**
```bash
python -m src.presentation.main --epochs 100
```

**Exemplo com batch size maior:**
```bash
python -m src.presentation.main --batch_size 64 --epochs 50
```

**Exemplo com caminhos customizados:**
```bash
python -m src.presentation.main \
    --train_path meu_diretorio/train \
    --test_path meu_diretorio/test
```

### Parâmetros

- `--train_path`: Caminho para o diretório de treinamento (default: `data/train`)
- `--test_path`: Caminho para o diretório de teste (default: `data/test`)
- `--epochs`: Número de épocas de treinamento (default: 50)
- `--batch_size`: Tamanho do batch (default: 32)
- `--image_size`: Tamanho das imagens altura largura (default: 224 224)
- `--learning_rate`: Taxa de aprendizado (default: 0.001)
- `--num_images_to_plot`: Número de imagens para plotar com Grad-CAM (default: 9)

## Saídas Geradas

Após o treinamento, os seguintes arquivos são gerados:

1. **`models/cnn_model.h5`**: Modelo treinado salvo
2. **`models/checkpoints/best_model.h5`**: Melhor modelo durante treinamento
3. **`results/training_history.png`**: Gráficos de loss e acurácia ao longo do treinamento
4. **`results/confusion_matrix.png`**: Matriz de confusão
5. **`results/test_images_gradcam.png`**: Imagens de teste com overlays Grad-CAM mostrando as áreas que influenciam a classificação

## Arquitetura da CNN

O modelo utiliza:
- **Base Model**: MobileNetV2 pré-treinado no ImageNet (transfer learning)
- **Data Augmentation**: Random flip, rotation e zoom
- **Camadas de Classificação**: Global Average Pooling + Dense layers com dropout
- **Otimizador**: Adam com learning rate configurável
- **Callbacks**: Early stopping, learning rate reduction e model checkpointing

## Visualização Grad-CAM

O projeto implementa Grad-CAM (Gradient-weighted Class Activation Mapping) para visualizar as regiões da imagem que mais influenciam a predição do modelo. As áreas destacadas em vermelho/amarelo são as mais importantes para a classificação.

## Clean Architecture

O projeto segue os princípios de Clean Architecture:

- **Domain**: Entidades e interfaces (independente de frameworks)
- **Application**: Casos de uso (lógica de negócio)
- **Infrastructure**: Implementações concretas (TensorFlow, PIL, etc.)
- **Presentation**: Interface com o usuário (CLI)

Isso permite:
- Fácil substituição de implementações
- Testabilidade
- Manutenibilidade
- Baixo acoplamento

## Licença

Este projeto é fornecido como está, para fins educacionais e de pesquisa.

## Contribuindo

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request.

