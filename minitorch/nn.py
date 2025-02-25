from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # TODO: Implement for Task 4.3.

    # pooling
    # https://minitorch.github.io/module4/pooling/

    height_now = height // kh
    width_now = width // kw

    x = input.contiguous()
    y = x.view(batch, channel, height, width_now, kw)
    z = y.permute(0, 1, 3, 2, 4)
    temp = z.contiguous()

    # Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width)
    result = temp.view(batch, channel, height_now, width_now, kh * kw)

    return result, height_now, width_now
    # raise NotImplementedError("Need to implement for Task 4.3")


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape

    # TODO: Implement for Task 4.3.
    input, height_now, width_now = tile(input, kernel)
    out = input.mean(dim=4)
    result = out.view(batch, channel, height_now, width_now)

    return result
    # raise NotImplementedError("Need to implement for Task 4.3")


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"

        # TODO: Implement for Task 4.4.
        # similar to argmax
        ctx.save_for_backward(input, dim)
        result = max_reduce(input, int(dim[0]))
        return result
        # raise NotImplementedError("Need to implement for Task 4.4")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"

        # TODO: Implement for Task 4.4.
        input, dim = ctx.saved_values
        result = grad_output * argmax(input, int(dim[0]))
        return result, 0  # 0 is needed!
        # raise NotImplementedError("Need to implement for Task 4.4")


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """

    # TODO: Implement for Task 4.4.
    result = input.exp() / input.exp().sum(dim=dim)
    return result

    # raise NotImplementedError("Need to implement for Task 4.4")


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """

    # TODO: Implement for Task 4.4.
    max_input = max(input, dim)
    temp = (input - max_input).exp().sum(dim=dim).log()
    result = input - temp - max_input
    return result


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape

    # TODO: Implement for Task 4.4.
    input, new_height, new_width = tile(input, kernel)
    result = max(input, 4)

    return result.view(batch, channel, new_height, new_width)
    # raise NotImplementedError("Need to implement for Task 4.4")


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with randoom positions dropped out
    """
    # TODO: Implement for Task 4.4.
    if ignore:
        return input
    else:
        result = input * (rand(input.shape) > rate)
        return result

    # raise NotImplementedError("Need to implement for Task 4.4")