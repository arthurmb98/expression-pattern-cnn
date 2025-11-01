from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Dataset:
    """Entidade que representa um dataset de imagens."""
    train_path: str
    test_path: str
    classes: List[str]
    num_classes: int
    image_size: Tuple[int, int] = (224, 224)
