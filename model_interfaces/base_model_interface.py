from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def predict(self, frame: np.ndarray):
        """Run inference on a frame and return detection results."""
        pass

    @abstractmethod
    def get_label(self, class_id: int) -> str:
        """Map class_id to human-readable label."""
        pass
