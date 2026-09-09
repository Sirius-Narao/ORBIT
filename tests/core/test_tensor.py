import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.orbit.core.tensor import Tensor


def run_tests():
    print("=" * 60)
    print("   RUNNING TENSOR SUITE TESTS   ")
    print("=" * 60 + "\n")

    # 1. Initialization and Attributes
    print("[1/8] Testing Initialization & Metadata Properties...")
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    assert t.shape == (2, 3), f"Expected shape (2, 3), got {t.shape}"
    assert t.ndim == 2, f"Expected ndim 2, got {t.ndim}"
    assert t.size == 6, f"Expected size 6, got {t.size}"
    assert t.dtype == np.float64, f"Expected float64, got {t.dtype}"
    assert t.requires_grad is True
    print(f"  [OK] Properties: shape={t.shape}, ndim={t.ndim}, size={t.size}, dtype={t.dtype}")

    # 2. String Representations
    print("\n[2/8] Testing Representation (__repr__ & __str__)...")
    print(f"  repr: {repr(t)}")
    print("  str snippet:\n" + str(t))

    # 3. Standard Arithmetic Operators
    print("[3/8] Testing Binary Math Operators (+, -, *, /, **, @)...")
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[2.0, 0.5], [1.0, 2.0]])

    add_res = a + b
    sub_res = a - b
    mul_res = a * b
    div_res = a / b
    pow_res = a ** b
    matmul_res = a @ b

    assert np.allclose(add_res.data, [[3.0, 2.5], [4.0, 6.0]])
    assert np.allclose(sub_res.data, [[-1.0, 1.5], [2.0, 2.0]])
    assert np.allclose(mul_res.data, [[2.0, 1.0], [3.0, 8.0]])
    assert np.allclose(div_res.data, [[0.5, 4.0], [3.0, 2.0]])
    assert np.allclose(pow_res.data, [[1.0, 1.41421356], [3.0, 16.0]])
    assert np.allclose(matmul_res.data, [[4.0, 4.5], [10.0, 9.5]])
    print("  [OK] All binary math operations computed correctly.")

    # 4. Reverse Operators (with scalars)
    print("\n[4/8] Testing Reverse Math Operators (scalar op tensor)...")
    radd = 5 + a
    rsub = 5 - a
    rmul = 3 * a
    rdiv = 12 / a
    rpow = 2 ** a
    rmat = Tensor([[1.0, 0.0], [0.0, 1.0]]) @ a

    assert np.allclose(radd.data, [[6.0, 7.0], [8.0, 9.0]])
    assert np.allclose(rsub.data, [[4.0, 3.0], [2.0, 1.0]])
    assert np.allclose(rmul.data, [[3.0, 6.0], [9.0, 12.0]])
    assert np.allclose(rdiv.data, [[12.0, 6.0], [4.0, 3.0]])
    assert np.allclose(rpow.data, [[2.0, 4.0], [8.0, 16.0]])
    assert np.allclose(rmat.data, a.data)
    print("  [OK] All reverse math operations computed correctly.")

    # 5. Unary Negation
    print("\n[5/8] Testing Negation (-tensor)...")
    neg_a = -a
    assert np.allclose(neg_a.data, [[-1.0, -2.0], [-3.0, -4.0]])
    print(f"  [OK] -a data:\n{neg_a.data}")

    # 6. Aggregation Functions (sum & mean)
    print("\n[6/8] Testing Reductions (sum, mean)...")
    t_red = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert t_red.sum().data == 21.0
    assert np.allclose(t_red.sum(axis=0).data, [5.0, 7.0, 9.0])
    assert t_red.mean().data == 3.5
    assert np.allclose(t_red.mean(axis=1).data, [2.0, 5.0])
    print(f"  [OK] sum()={t_red.sum().data}, mean()={t_red.mean().data}")

    # 7. Reshaping
    print("\n[7/8] Testing Reshape...")
    t_flat = Tensor([1, 2, 3, 4, 5, 6])
    reshaped_args = t_flat.reshape(2, 3)
    reshaped_tuple = t_flat.reshape((2, 3))
    assert reshaped_args.shape == (2, 3)
    assert reshaped_tuple.shape == (2, 3)
    print(f"  [OK] reshaped shape: {reshaped_args.shape}")

    # 8. Slicing & Indexing
    print("\n[8/8] Testing Slicing (__getitem__)...")
    t_slice = Tensor([[10, 20, 30], [40, 50, 60]])
    sliced = t_slice[0, 1:]
    assert np.allclose(sliced.data, [20, 30])
    print(f"  [OK] t_slice[0, 1:] data: {sliced.data}")

    print("   ALL TESTS PASSED SUCCESSFULLY!   ")
    print("=" * 60)


def test_tensor_suite():
    run_tests()


if __name__ == "__main__":
    run_tests()

