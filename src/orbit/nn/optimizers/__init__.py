try:
    from .optimizer import Optimizer
    from .sgd import SGD
    from .adam import Adam
except ImportError:
    from orbit.nn.optimizers.optimizer import Optimizer
    from orbit.nn.optimizers.sgd import SGD
    from orbit.nn.optimizers.adam import Adam

__all__ = ["Optimizer", "SGD", "Adam"]
