from typing import Dict, List, Optional


def _floats(values: Optional[List[float]]) -> Optional[List[float]]:
    return [float(v) for v in values] if values is not None else None

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
        layer_gradient_norm_history: Optional[Dict[str, List[float]]] = None,
        diverged_at_epoch: Optional[int] = None,
        val_loss_history: Optional[List[float]] = None,
        val_accuracy_history: Optional[List[float]] = None,
        best_epoch: Optional[int] = None,
        stopped_early_at_epoch: Optional[int] = None,
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
        # {weight parameter name, e.g. "0.weight": per-epoch gradient norm};
        # None for results saved before it was tracked.
        self.layer_gradient_norm_history = layer_gradient_norm_history
        # The epoch where the loss or gradients became inf/NaN and training
        # stopped; None for a run that finished normally (or an old result).
        # When set, final_loss is NaN and the histories end just before it.
        self.diverged_at_epoch = diverged_at_epoch
        # Per-epoch loss/accuracy on the validation set ("validation_split");
        # None without one (or for an old result).
        self.val_loss_history = val_loss_history
        self.val_accuracy_history = val_accuracy_history
        # Early stopping ("patience"): the epoch with the lowest validation
        # loss, whose weights the trained model ends with, and the epoch
        # training stopped at (None if it ran all its epochs). loss_history
        # still covers every epoch trained, up to the stop.
        self.best_epoch = best_epoch
        self.stopped_early_at_epoch = stopped_early_at_epoch

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
            "layer_gradient_norm_history": (
                {name: [float(g) for g in norms] for name, norms in self.layer_gradient_norm_history.items()}
                if self.layer_gradient_norm_history is not None
                else None
            ),
            "diverged_at_epoch": int(self.diverged_at_epoch) if self.diverged_at_epoch is not None else None,
            "val_loss_history": _floats(self.val_loss_history),
            "val_accuracy_history": _floats(self.val_accuracy_history),
            "best_epoch": int(self.best_epoch) if self.best_epoch is not None else None,
            "stopped_early_at_epoch": (
                int(self.stopped_early_at_epoch) if self.stopped_early_at_epoch is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Results":
        return cls(**data)
