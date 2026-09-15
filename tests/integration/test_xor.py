from orbit.nn.module import Module
from orbit.nn.layers.linear import Linear
from orbit.nn.activations import Tanh, Sigmoid
from orbit.nn.losses.mse import MSE
from orbit.nn.optimizers import SGD
from orbit.core import Tensor
from orbit.core.dataset import TensorDataset
from orbit.core.dataloader import DataLoader
from orbit.nn.training.trainer import Trainer

class XORModel(Module):
    def __init__(self):
        super().__init__()

        # register internal modules
        self.fc1 = self.register_module(
            "fc1", Linear(2, 8)
        )
        self.fc2 = self.register_module(
            "fc2", Linear(8, 1)
        )
        self.tanh = self.register_module(
            "tanh", Tanh()
        )
        self.sigmoid = self.register_module(
            "sigmoid", Sigmoid()
        )

    def forward(self, x: Tensor):
        return self.sigmoid(self.fc2(self.tanh(self.fc1(x))))


def test_xor_converges():
    """
    End-to-end proof that Tensor -> Module -> Linear -> Activation -> Loss
    -> autograd -> Optimizer all correctly compose.

    XOR is not linearly separable, so a model that actually learns it
    (rather than, say, always predicting 0.5) demonstrates that gradients
    are flowing correctly through a hidden layer and a nonlinearity.

    Trains on the full batch of 4 rows at once. This relies on
    backward_add correctly un-broadcasting the gradient for Linear's bias
    term (see test_add_broadcast_bias_backward in test_autograd.py).
    """
    X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = Tensor([[0.0], [1.0], [1.0], [0.0]])

    dataset = TensorDataset(X.data, Y.data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = XORModel()
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=2)

    loss = Trainer().fit(model, loss_fn, optimizer, dataloader, epochs=3000, verbose=True, log_every=10)

    assert loss < 0.05

    predictions = (model(X).data > 0.5).astype(float)
    assert (predictions == Y.data).all()