"""
Implementation of the autodifferentiation Functions for Tensor.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Implement for Task 2.4.
        a, b = ctx.saved_values

        # use expand function
        return a.expand(b * grad_output), b.expand(a * grad_output)
        # raise NotImplementedError("Need to implement for Task 2.4")


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4.
        t = ctx.saved_values[0]
        return grad_output.f.mul_zip(
            grad_output, t.f.mul_zip(t, t.f.add_zip(t.f.neg_map(t), tensor([1.0])))
        )
        # raise NotImplementedError("Need to implement for Task 2.4")


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4.
        x_prime = ctx.saved_values[0]
        return grad_output.f.relu_back_zip(x_prime, grad_output)
        # raise NotImplementedError("Need to implement for Task 2.4")


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4.
        t: Tensor = ctx.saved_values[0]
        return t.f.log_back_zip(t, grad_output)
        # raise NotImplementedError("Need to implement for Task 2.4")


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        res = t1.f.exp_map(t1)
        ctx.save_for_backward(res)
        return res

        # ctx.save_for_backward(t1)
        # return t1.f.exp_map(t1)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4.
        t: Tensor = ctx.saved_values[0]
        return t.f.mul_zip(grad_output, t)
        # raise NotImplementedError("Need to implement for Task 2.4")


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        return a.f.lt_zip(a, b)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Implement for Task 2.4.
        nt1 = grad_output.zeros()
        nt2 = grad_output.zeros()
        return nt1, nt2
        # raise NotImplementedError("Need to implement for Task 2.4")


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        return a.f.eq_zip(a, b)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Implement for Task 2.4.
        nt1 = grad_output.zeros()
        nt2 = grad_output.zeros()
        return nt1, nt2
        # raise NotImplementedError("Need to implement for Task 2.4")


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        return a.f.is_close_zip(a, b)
        # raise NotImplementedError("Need to implement for Task 2.3")


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.

        # create a tensor and a order
        tensor1 = a._tensor
        order1 = order.to_numpy().astype(np.int32)

        # save shape and strides of the tensor for backward
        shape = tensor1.shape
        strides = tensor1.strides
        ctx.save_for_backward(shape, strides)

        # create new shape and strides
        new_shape = tuple([shape[i] for i in order1])
        new_strides = tuple([strides[i] for i in order1])

        return minitorch.Tensor.make(tensor1._storage, new_shape, new_strides, backend=a.backend)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        # TODO: Implement for Task 2.4.
        # use saved shape and strides
        (shape, strides,) = ctx.saved_values

        # make the original tensor
        result = minitorch.Tensor.make(
            grad_output._tensor._storage,
            shape, strides,
            backend=grad_output.backend)
        return (result, 0.0,)  # float -> 0.0

        # other attempt:
        # (order, ) = ctx.saved_values
        # new_order = minitorch.tensor([0]) * len(order)
        # for i in range(len(order)):
        #     new_order[order[i]] = i
        # # new_tensor_data = grad_output._tensor.permute(*order)
        # return a._new(a._tensor.permute(*order))

        # order = ctx.saved_values
        # result = []
        # for i in sorted(enumerate(order), key=lambda i: i[1]):
        #     result.append(i[0])
        # order = [a[0] for a in sorted(enumerate(order), key=lambda a: a[1])]
        # return grad_output.a._new(grad_output._tensor.permute(*result))

        # reverse_order = [a.index(i) for i in range(len(order))]
        # return grad_output._new(grad_output._tensor.permute(*reverse_order))

        # raise NotImplementedError("Need to implement for Task 2.4")


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    return minitorch.Tensor.make(
        [0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )