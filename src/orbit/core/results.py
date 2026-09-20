from typing import List, Optional

class Results:
    def __init__(
        self,
        name: str,
        final_loss: float,
        loss_history: List[float],
        hyperparams: Optional[dict] = None,
        duration_seconds: Optional[float] = None,
        gradient_norm_history: Optional[List[float]] = None,
        accuracy_history: Optional[List[float]] = None,
        test_loss: Optional[float] = None,
        test_accuracy: Optional[float] = None,
    ):
        self.name = name
        self.final_loss = final_loss
        self.loss_history = loss_history
        self.hyperparams = hyperparams
        if self.hyperparams is None:
            self.hyperparams = {}
        self.duration_seconds = duration_seconds
        self.gradient_norm_history = gradient_norm_history
        self.accuracy_history = accuracy_history
        self.test_loss = test_loss
        self.test_accuracy = test_accuracy

    def __repr__(self):
        return f"Results(name={self.name!r}, final_loss={self.final_loss:.4f}, epochs={len(self.loss_history)})"

    def to_dict(self) -> dict:
        """
        Plain-Python representation, safe to json.dump directly. final_loss
        and loss_history entries are cast to float because Trainer computes
        them via numpy arithmetic (np.float64, not float), which json can't
        serialize on its own.
        """
        return {
            "name": self.name,
            "final_loss": float(self.final_loss),
            "loss_history": [float(loss) for loss in self.loss_history],
            "hyperparams": self.hyperparams,
            "duration_seconds": float(self.duration_seconds) if self.duration_seconds is not None else None,
            "gradient_norm_history": (
                [float(g) for g in self.gradient_norm_history]
                if self.gradient_norm_history is not None
                else None
            ),
            "accuracy_history": (
                [float(a) for a in self.accuracy_history]
                if self.accuracy_history is not None
                else None
            ),
            "test_loss": float(self.test_loss) if self.test_loss is not None else None,
            "test_accuracy": float(self.test_accuracy) if self.test_accuracy is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Results":
        return cls(**data)
