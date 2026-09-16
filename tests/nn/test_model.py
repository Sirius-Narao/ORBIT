import numpy as np
from orbit.core import Tensor
from orbit.nn import Module, Sequential
from orbit.nn.layers import Linear
from orbit.nn.activations import Tanh, Sigmoid


def make_xor_layers():
    """
    Fresh Linear/Tanh/Sigmoid instances with identical weights each call,
    so a Sequential model and a hand-built Module can be compared fairly.
    """
    fc1 = Linear(2, 8)
    fc2 = Linear(8, 1)
    fc1.weight.data = np.full((2, 8), 0.1)
    fc1.bias.data = np.zeros(8)
    fc2.weight.data = np.full((8, 1), 0.2)
    fc2.bias.data = np.zeros(1)
    return fc1, fc2, Tanh(), Sigmoid()


class XORModel(Module):
    def __init__(self, fc1, tanh, fc2, sigmoid):
        super().__init__()
        self.fc1 = self.register_module("fc1", fc1)
        self.tanh = self.register_module("tanh", tanh)
        self.fc2 = self.register_module("fc2", fc2)
        self.sigmoid = self.register_module("sigmoid", sigmoid)

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(self.fc2(self.tanh(self.fc1(x))))


def test_sequential_forward_shape():
    model = Sequential(Linear(2, 8), Tanh(), Linear(8, 1), Sigmoid())
    x = Tensor(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))

    out = model(x)

    assert out.shape == (4, 1)


def test_sequential_registers_all_parameters():
    model = Sequential(Linear(2, 8), Tanh(), Linear(8, 1), Sigmoid())

    # 2 Linear layers -> weight + bias each = 4 parameters total.
    # (Tanh/Sigmoid have none.) This fails if Sequential stored layers in
    # a plain list instead of using register_module.
    assert len(model.parameters()) == 4


def test_sequential_matches_hand_built_module():
    fc1, fc2, tanh, sigmoid = make_xor_layers()
    hand_built = XORModel(fc1, tanh, fc2, sigmoid)

    fc1_seq, fc2_seq, tanh_seq, sigmoid_seq = make_xor_layers()
    sequential = Sequential(fc1_seq, tanh_seq, fc2_seq, sigmoid_seq)

    x = Tensor(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))

    assert np.allclose(hand_built(x).data, sequential(x).data)
