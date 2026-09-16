from typing import List, Optional

class Results:
    def __init__(self, name: str, final_loss: float, loss_history: List[float], hyperparams: Optional[dict] = None):
        self.name = name
        self.final_loss = final_loss
        self.loss_history = loss_history
        self.hyperparams = hyperparams
        if self.hyperparams is None:
            self.hyperparams = {}

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
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Results":
        return cls(**data)
