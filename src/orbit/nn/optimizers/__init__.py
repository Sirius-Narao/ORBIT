try:
    from .optimizer import Optimizer
    from .sgd import SGD
except ImportError:
    from orbit.nn.optimizers.optimizer import Optimizer
    from orbit.nn.optimizers.sgd import SGD

__all__ = ["Optimizer", "SGD"]