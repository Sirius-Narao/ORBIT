import numpy as np
from typing import Dict, List, Optional, Tuple, Iterator
from orbit.core.tensor import Tensor
from orbit.nn.parameter import Parameter


class Module:
    """
    Base class for all neural network modules in ORBIT.

    Modules can contain Parameters (learnable weights/biases) and child Modules.
    Modules can be nested in trees (e.g. MLP -> Linear -> Parameter).
    """

    def __init__(self) -> None:
        self._params: Dict[str, Parameter] = {}
        self._modules: Dict[str, Module] = {}
        self.training: bool = True

    def register_parameter(self, name: str, param: Optional[Parameter]) -> Optional[Parameter]:
        """
        Register a Parameter with this module under `name`.
        """
        if param is not None:
            if not isinstance(param, Parameter):
                raise TypeError(f"Cannot register parameter '{name}' of type {type(param)}. Expected Parameter.")
            self._params[name] = param
        return param

    def register_module(self, name: str, module: Optional["Module"]) -> Optional["Module"]:
        """
        Register a child Module under `name`.
        """
        if module is not None:
            if not isinstance(module, Module):
                raise TypeError(f"Cannot register child module '{name}' of type {type(module)}. Expected Module.")
            self._modules[name] = module
        return module

    def parameters(self) -> List[Parameter]:
        """
        Return a list of all parameters in this module and all child modules recursively.
        """
        params: List[Parameter] = list(self._params.values())
        for submodule in self._modules.values():
            params.extend(submodule.parameters())
        return params

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        """
        Yield (name, parameter) pairs for all parameters in this module
        and child modules recursively.
        """
        for name, param in self._params.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param

        for child_name, submodule in self._modules.items():
            sub_prefix = f"{prefix}.{child_name}" if prefix else child_name
            yield from submodule.named_parameters(prefix=sub_prefix)

    def children(self) -> Iterator["Module"]:
        """
        Yield immediate child modules.
        """
        for submodule in self._modules.values():
            yield submodule

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        """
        Yield (name, child_module) pairs for immediate child modules.
        """
        for name, submodule in self._modules.items():
            yield name, submodule

    def modules(self) -> Iterator["Module"]:
        """
        Yield self and all descendant submodules recursively.
        """
        yield self
        for submodule in self._modules.values():
            yield from submodule.modules()

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        """
        Yield (name, module) pairs for self and all descendant submodules recursively.
        """
        yield prefix, self
        for child_name, submodule in self._modules.items():
            sub_prefix = f"{prefix}.{child_name}" if prefix else child_name
            yield from submodule.named_modules(prefix=sub_prefix)

    def forward(self, *args, **kwargs):
        """
        Define the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        Invoke forward computation.
        """
        return self.forward(*args, **kwargs)

    def zero_grad(self) -> None:
        """
        Reset gradients of all parameters in this module and submodules to None.
        """
        for param in self.parameters():
            param.grad = None

    def train(self, mode: bool = True) -> "Module":
        """
        Set training mode to `mode` for this module and all child submodules.
        """
        self.training = mode
        for submodule in self._modules.values():
            submodule.train(mode)
        return self

    def eval(self) -> "Module":
        """
        Set evaluation mode (training=False) for this module and all child submodules.
        """
        return self.train(False)


if __name__ == "__main__":
    # Example demonstrating Module usage
    class DummyLinear(Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.weight = self.register_parameter(
                "weight", Parameter(np.random.randn(in_features, out_features))
            )
            self.bias = self.register_parameter(
                "bias", Parameter(np.zeros(out_features))
            )

        def forward(self, x: Tensor) -> Tensor:
            return x @ self.weight + self.bias

    class DummyMLP(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = self.register_module("Linear1", DummyLinear(10, 5))
            self.linear2 = self.register_module("Linear2", DummyLinear(5, 2))

        def forward(self, x: Tensor) -> Tensor:
            return self.linear2(self.linear1(x))

    model = DummyMLP()
    print("Named parameters in DummyMLP:")
    for name, param in model.named_parameters():
        print(f"  {name}: shape {param.shape}")

    print("\nAll modules in hierarchy:")
    for name, mod in model.named_modules():
        print(f"  '{name}': {type(mod).__name__}")