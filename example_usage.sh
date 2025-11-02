#!/bin/bash
# Exemplo de uso do projeto CNN para classificação de imagens

# Comando mais simples - usa todos os valores padrão
python -m src.presentation.main

# Exemplo com valores padrão explícitos
# python -m src.presentation.main \
#     --train_path data/train \
#     --test_path data/test \
#     --epochs 50 \
#     --batch_size 32

# Exemplo com opções customizadas
# python -m src.presentation.main \
#     --train_path data/train \
#     --test_path data/test \
#     --epochs 100 \
#     --batch_size 64 \
#     --image_size 224 224 \
#     --learning_rate 0.0001 \
#     --num_images_to_plot 12

