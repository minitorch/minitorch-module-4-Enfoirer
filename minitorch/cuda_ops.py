from typing import Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:

        # calculate in and out index, then position (i)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # similar as tensor_ops.py task 2.3

        # position within size
        if i < out_size:

            # to_index and broadcast_index

            # Convert an `ordinal` to an index in the `shape`
            to_index(i, out_shape, out_index)
            # Convert a `big_index` into `big_shape` to a smaller `out_index`
            # into `shape` following broadcasting rules.
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # calculate in and out position
            out_position = index_to_position(out_index, out_strides)
            in_position = index_to_position(in_index, in_strides)

            # get data from the storage using the index of in_position
            data = in_storage[in_position]
            result = fn(data)

            # update out
            # store result back to the out storage for out tensor
            out[out_position] = result

        # raise NotImplementedError("Need to implement for Task 3.3")
    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:

        # calculate in and out index, then position (i)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # similar as tensor_ops.py task 2.3

        # similar as tensor_map
        # within size
        if i < out_size:

            # Convert an `ordinal` to an index in the `shape`
            to_index(i, out_shape, out_index)

            # calculate out position
            out_position = index_to_position(out_index, out_strides)
            # Convert a `big_index` into `big_shape` to a smaller `out_index`
            # into `shape` following broadcasting rules.
            broadcast_index(out_index, out_shape, a_shape, a_index)

            # calculate the position of a
            a_position = index_to_position(a_index, a_strides)
            # Convert a `big_index` into `big_shape` to a smaller `out_index`
            # into `shape` following broadcasting rules.
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # calculate the position of b
            b_position = index_to_position(b_index, b_strides)

            # get data from the storage using the index of in_position
            a_data = a_storage[a_position]
            b_data = b_storage[b_position]
            result = fn(a_data, b_data)

            # update out
            # store result back to the out storage for out tensor
            out[out_position] = result

        # raise NotImplementedError("Need to implement for Task 3.3")

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.

    # within size
    # if i is smaller than size, update cache
    if i < size:
        val = float(a[i])
        cache[pos] = val
        # sync
        cuda.syncthreads()
    # not within size
    else:
        cache[pos] = 0

    # within size
    if i < size:

        # if i is smaller than size, loop over
        for j in range(5):  # j = 1, 2, 4, 8, 16
            if pos % (2**j * 2) == 0:
                cache[pos] += cache[pos + 2**j]
                # sync
                cuda.syncthreads()
            if pos == 0:
                # update out
                out[cuda.blockIdx.x] = cache[0]

    # raise NotImplementedError("Need to implement for Task 3.3")


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.

    Returns:
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        cache[pos] = reduce_value

        # TODO: Implement for Task 3.3.
        # similar as tensor_ops.py task 2.3

        # within size
        if out_pos < out_size:

            # Convert an `ordinal` to an index in the `shape`
            to_index(out_pos, out_shape, out_index)
            i = index_to_position(out_index, out_strides)

            # update out_index
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos
            if out_index[reduce_dim] < a_shape[reduce_dim]:

                # calculate the index of a
                a_index = index_to_position(out_index, a_strides)

                # update cache and sync
                cache[pos] = a_storage[a_index]
                cuda.syncthreads()

                # set a counter
                counter = 0

                while 2 ** counter < BLOCK_DIM:
                    j = 2 ** counter
                    if pos % (j * 2) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + j])
                        # sync
                        cuda.syncthreads()
                    counter += 1
            if pos == 0:
                out[i] = cache[0]
        # raise NotImplementedError("Need to implement for Task 3.3")

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square
    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.

    # reference:
    # https://numba.pydata.org/numba-doc/dev/cuda/examples.html

    #  for i:
    #      for j:
    #           for k:
    #               out[i, j] += a[i, k] * b[k, j]

    # a and b both are shape [size, size] with strides [size, 1].
    # shared memory
    # data must be first moved to shared memory.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # x thread and y thread for cuda
    x_thread = cuda.threadIdx.x
    y_thread = cuda.threadIdx.y

    if x_thread >= size or y_thread >= size:
        return

    # update shared memory and then sync
    a_shared[x_thread, y_thread] = a[size * x_thread + y_thread]
    b_shared[x_thread, y_thread] = b[size * x_thread + y_thread]
    cuda.syncthreads()

    # sum
    temp = 0

    # calcaulte sum
    for i in range(size):
        temp += a_shared[x_thread, i] * b_shared[i, y_thread]

    # update out
    out[size * x_thread + y_thread] = temp

    # raise NotImplementedError("Need to implement for Task 3.3")


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.

    # reference:
    # https://numba.pydata.org/numba-doc/dev/cuda/examples.html

    # total
    temp = 0

    # loop over
    # each thread computes one element in the result matrix.
    # the dot product is chunked into dot products of TPB-long vectors.
    for x_initial in range(0, a_shape[2], BLOCK_DIM):  # 120

        x = x_initial + pj
        if i < a_shape[1] and x < a_shape[2]:
            a_stride_batch = a_batch_stride * batch
            a_stride_total = a_strides[1] * i + a_strides[2] * x
            a_total_stride = a_stride_batch + a_stride_total
            a_shared[pi, pj] = a_storage[a_total_stride]

        x = x_initial + pi
        if j < b_shape[2] and x < b_shape[1]:
            b_stride_batch = b_batch_stride * batch
            b_stride_total = b_strides[1] * x + b_strides[2] * j
            b_total_stride = b_stride_batch + b_stride_total
            b_shared[pi, pj] = b_storage[b_total_stride]

        # wait until all threads finish computing
        # sycn
        cuda.syncthreads()

        # computes partial product on the shared memory
        for x in range(BLOCK_DIM):
            if (x_initial + x) < a_shape[2]:
                temp = temp + a_shared[pi, x] * b_shared[x, pj]

    # update out
    if out_shape[1] > i and out_shape[2] > j:
        out_total_stride = out_strides[0] * batch +\
            out_strides[1] * i + out_strides[2] * j
        out[out_total_stride] = temp

    # raise NotImplementedError("Need to implement for Task 3.4")


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)