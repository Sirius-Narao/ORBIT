from orbit.nn.losses.loss import Loss
from orbit.core import Tensor
from orbit.core.autograd import backward_cross_entropy
import numpy as np

class CrossEntropy(Loss):

    def __init__(self):
        super().__init__("cross_entropy")
    
    def forward(self, y_pred: Tensor, y_true: Tensor):
        """
        Supports logits/probabilities for y_pred, and class index for y_true only.

        CCE formula (for a batch of N samples, C classes):
            probs  = softmax(y_pred)            # shape (N, C) or (C,)
            loss_i = -log(probs[i, y_true[i]])  # pick the true-class probability
            loss   = mean(loss_i)  over i in N
        """
        # --- softmax over the last axis (converts logits → probabilities) ---
        logits  = y_pred.data                                              # (N, C) or (C,)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)         # numerical stability
        exp     = np.exp(shifted)
        probs   = exp / np.sum(exp, axis=-1, keepdims=True)               # (N, C) or (C,)

        # --- clamp to avoid log(0) ---
        eps   = 1e-12
        probs = np.clip(probs, eps, 1.0)

        # --- gather the probability assigned to the true class ---
        indices = y_true.data.astype(int)                                  # 1-D integer array
        if probs.ndim == 1:
            # single sample: probs shape (C,), indices is a scalar
            true_probs = probs[indices]
        else:
            # batch: probs shape (N, C), pick probs[i, indices[i]] for each i
            true_probs = probs[np.arange(len(indices)), indices]           # (N,)

        # --- mean negative log-likelihood ---
        loss = np.mean(-np.log(true_probs))

        # --- build the output Tensor and wire the computation graph ---
        out = Tensor(
            loss,
            requires_grad=y_pred.requires_grad,
            operation="cross_entropy",
            parents=[y_pred],                                              # y_true has no gradient
        )
        # Capture indices in the closure so the backward rule can reconstruct
        # the gradient dL/dlogits = (probs - one_hot) / N
        out._backward = lambda res=out, parent=y_pred, idx=indices: \
            backward_cross_entropy(res, parent, idx)

        # return tensor
        return out
    
    