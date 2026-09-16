import numpy as np
import pytest
from orbit.core import Tensor
from orbit.nn import Parameter, Module


class LinearDummy(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = self.register_parameter(
            "weight", Parameter(np.ones((in_features, out_features)))
        )
        self.bias = self.register_parameter(
            "bias", Parameter(np.zeros(out_features))
        )

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias


class MLPDummy(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = self.register_module("layer1", LinearDummy(4, 3))
        self.layer2 = self.register_module("layer2", LinearDummy(3, 2))

    def forward(self, x: Tensor) -> Tensor:
        return self.layer2(self.layer1(x))


def test_empty_module():
    m = Module()
    assert len(m.parameters()) == 0
    assert len(list(m.named_parameters())) == 0
    assert len(list(m.children())) == 0
    assert len(list(m.modules())) == 1
    assert m.training is True


def test_register_parameter():
    m = Module()
    p = Parameter(np.array([1.0, 2.0]))
    registered = m.register_parameter("weight", p)

    assert registered is p
    assert "weight" in m._params
    assert m._params["weight"] is p

    # Invalid registration type
    with pytest.raises(TypeError):
        m.register_parameter("invalid", [1, 2, 3])  # type: ignore


def test_register_module():
    m = Module()
    child = Module()
    registered = m.register_module("child", child)

    assert registered is child
    assert "child" in m._modules
    assert m._modules["child"] is child

    # Invalid registration type
    with pytest.raises(TypeError):
        m.register_module("invalid", "not_a_module")  # type: ignore


def test_parameters_and_named_parameters():
    model = MLPDummy()

    # Total parameters across layers: 4 (layer1.weight, layer1.bias, layer2.weight, layer2.bias)
    params = model.parameters()
    assert len(params) == 4
    assert isinstance(params[0], Parameter)

    named_params = dict(model.named_parameters())
    expected_keys = {"layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"}
    assert set(named_params.keys()) == expected_keys
    assert named_params["layer1.weight"] is model.layer1.weight
    assert named_params["layer2.bias"] is model.layer2.bias


def test_repeated_parameters_calls():
    model = MLPDummy()
    params1 = model.parameters()
    params2 = model.parameters()

    # Distinct list instances containing identical parameter references
    assert params1 is not params2
    assert params1 == params2
    assert [id(p) for p in params1] == [id(p) for p in params2]


def test_zero_grad():
    model = MLPDummy()
    # Attach dummy gradients
    for p in model.parameters():
        p.grad = np.ones_like(p.data)

    for p in model.parameters():
        assert p.grad is not None

    model.zero_grad()

    for p in model.parameters():
        assert p.grad is None


def test_train_eval_propagation():
    model = MLPDummy()
    assert model.training is True
    assert model.layer1.training is True
    assert model.layer2.training is True

    model.eval()
    assert model.training is False
    assert model.layer1.training is False
    assert model.layer2.training is False

    model.train()
    assert model.training is True
    assert model.layer1.training is True
    assert model.layer2.training is True


def test_children_vs_modules():
    model = MLPDummy()

    direct_children = list(model.children())
    assert len(direct_children) == 2
    assert model.layer1 in direct_children
    assert model.layer2 in direct_children

    all_modules = list(model.modules())
    assert len(all_modules) == 3  # model + layer1 + layer2
    assert all_modules[0] is model
    assert all_modules[1] is model.layer1
    assert all_modules[2] is model.layer2


def test_call_forward():
    model = MLPDummy()
    x = Tensor(np.ones((2, 4)))
    out = model(x)

    assert isinstance(out, Tensor)
    assert out.shape == (2, 2)
