"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    # TODO: Implement for Task 0.1.
    return float(x * y)


def id(x: float) -> float:
    "$f(x) = x$"
    # TODO: Implement for Task 0.1.
    return float(x)


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    # TODO: Implement for Task 0.1.
    return float(x + y)


def neg(x: float) -> float:
    "$f(x) = -x$"
    # TODO: Implement for Task 0.1.
    return float(-1 * x)


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    # TODO: Implement for Task 0.1.
    if x < y:
        return float(1)
    else:
        return float(0)


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    # TODO: Implement for Task 0.1.
    if x == y:
        return float(1)
    else:
        return float(0)


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    # TODO: Implement for Task 0.1.
    if x > y:
        return float(x)
    else:
        return float(y)


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    # TODO: Implement for Task 0.1.
    # 1e-2 -> 1*10^(-2)
    if abs(x - y) < 1e-2:
        return float(1)
    else:
        return float(0)


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    # TODO: Implement for Task 0.1.
    # https://www.cvmart.net/community/detail/5701
    if x >= 0:
        return 1 / (1 + math.exp(-1 * x))
    else:
        return math.exp(x) / (1 + math.exp(1 * x))


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    # TODO: Implement for Task 0.1.
    if x > 0:
        return float(x)
    else:
        return float(0)


# Epsilon
EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    ans = 1 / x
    # alterantive:
    # if x == log(x):
    #     return float(d * (-1 / x**2))
    # alternative:
    # return d / (x + EPS)
    return d * ans


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    # TODO: Implement for Task 0.1.
    return float(1 / x)


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    # if x == 1 / x:
    #     return float(d * (-1 / x**2))
    ans = float(-1 / x**2)
    return d * ans


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    if x < 0:
        ans = 0
    else:
        ans = 1
    return d * ans


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    # TODO: Implement for Task 0.3.
    def fn_new(ls: Iterable[float]) -> Iterable[float]:
        ls_new = []
        for i in ls:
            ls_new.append(fn(i))
        return ls_new

    return fn_new


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    """
    Apply 'neg' to each element in 'ls' with 'map' function.

    Args:
        ls: input list.

    Returns:
        A list after applying 'neg' to each element in input.
    """
    # TODO: Implement for Task 0.3.
    neg_ls = map(neg)
    return neg_ls(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    # TODO: Implement for Task 0.3.
    def new_fun(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ans_ls = []
        for i, j in zip(ls1, ls2):
            ans_ls.append(fn(i, j))
        return ans_ls
        # return [fn(i, j) for i, j in zip(ls1, ls2)]

    return new_fun


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    """
    Add the elements of 'ls1' and 'ls2' with function 'zipWith' and 'add'.

    Args:
        ls1: first input list
        ls1: second input list

    Returns:
        Return a list after adding elements of two input list with function 'zipWith' and 'add'.
    """
    # TODO: Implement for Task 0.3.
    add_ls = zipWith(add)
    return add_ls(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    # TODO: Implement for Task 0.3.
    def fn_new(ls: Iterable[float]) -> float:
        res = start  # x_0  fn = '+'
        for x in ls:
            res = fn(x, res)  # x1 + x0 -> res / x2 + res -> res / x3 + res -> res / res
        return res

    return fn_new


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    """
    To sum up an input list using function `reduce` and `add`.

    Args:
        ls: an input list

    Returns:
        Return a float (sum) after suming up the input list.
    """
    # TODO: Implement for Task 0.3.
    sum = reduce(add, 0)
    return sum(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    """
    To make a product of a list using function 'reduce' and 'mul'.

    Args:
        ls: an input list.

    Returns:
        Return a float (product) after making the product of input list.

    """
    # TODO: Implement for Task 0.3.
    product = reduce(mul, 1)
    return product(ls)