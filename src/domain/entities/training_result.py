from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TrainingResult:
    """Entidade que representa o resultado do treinamento."""
    history: Dict[str, List[float]]
    model_path: str
    accuracy: float
    loss: float
    validation_accuracy: float
    validation_loss: float
    num_classes: int
    classes: List[str]
