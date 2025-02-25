from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # answer from module 1
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(vals2)
    return delta / (2 * epsilon)

    # vals1 = list(vals)
    # vals1[arg] -= epsilon
    # vals2 = list(vals)
    # vals2[arg] += epsilon
    # return (f(*vals2) - f(*vals1)) / (2 * epsilon)

    # raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # reference: https://en.wikipedia.org/wiki/Topological_sorting

    # list to contain the sorted nodes
    result: List[Variable] = []
    # use set() method to convert any of the iterable to sequence of iterable elements with distinct elements
    visited = set()

    def visit(var: Variable) -> None:

        # if not leaf, then visit
        if var.is_constant():
            return

        # pass if visited before
        if var.unique_id in visited:
            return
        # add current nonde into visited set
        visited.add(var.unique_id)

        # visit parents node
        for parent in var.parents:
            visit(parent)

        # add current node into the front of result list
        result.insert(0, var)

    # call visit function
    visit(variable)

    return result
    # raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # answer from module 1
    top_sorted_nodes = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    for node in top_sorted_nodes:
        node_deriv = derivatives[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(node_deriv)
            continue
        for parent, parent_deriv in node.chain_rule(node_deriv):
            derivatives[parent.unique_id] = derivatives.get(parent.unique_id, 0) + parent_deriv
    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values