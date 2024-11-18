# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


# Parallel Check
```
(.venv) (base) [huangf2@islogin01 mod3-fyng]$ python project/parallel_check.py
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (163)
--------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                         |
        out: Storage,                                                                 |
        out_shape: Shape,                                                             |
        out_strides: Strides,                                                         |
        in_storage: Storage,                                                          |
        in_shape: Shape,                                                              |
        in_strides: Strides,                                                          |
    ) -> None:                                                                        |
        # TODO: Implement for Task 3.1.                                               |
        if (out_strides == in_strides).all() and (out_shape == in_shape).all():-------| #0, 1
            for i in prange(len(out)):------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                            |
        else:                                                                         |
            for i in prange(len(out)):------------------------------------------------| #3
                out_index = np.empty(len(out_shape), dtype=np.int32)                  |
                in_index = np.empty(len(in_shape), dtype=np.int32)                    |
                to_index(i, out_shape, out_index)                                     |
                broadcast_index(out_index, out_shape, in_shape, in_index)             |
                in_pos = index_to_position(in_index, in_strides)                      |
                out_pos = index_to_position(out_index, out_strides)                   |
                out[out_pos] = fn(in_storage[in_pos])                                 |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (177) is hoisted
 out of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (178) is hoisted
 out of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: in_index = np.empty(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (211)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (211)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                                                                                  |
        out: Storage,                                                                                                                                                          |
        out_shape: Shape,                                                                                                                                                      |
        out_strides: Strides,                                                                                                                                                  |
        a_storage: Storage,                                                                                                                                                    |
        a_shape: Shape,                                                                                                                                                        |
        a_strides: Strides,                                                                                                                                                    |
        b_storage: Storage,                                                                                                                                                    |
        b_shape: Shape,                                                                                                                                                        |
        b_strides: Strides,                                                                                                                                                    |
    ) -> None:                                                                                                                                                                 |
        # TODO: Implement for Task 3.1.                                                                                                                                        |
        if np.array_equal(out_strides, a_strides) and np.array_equal(out_strides, b_strides) and np.array_equal(out_shape, a_shape) and np.array_equal(out_shape, b_shape):    |
            for i in prange(len(out)):-----------------------------------------------------------------------------------------------------------------------------------------| #4
                out[i] = fn(a_storage[i], b_storage[i])                                                                                                                        |
        else:                                                                                                                                                                  |
            for i in prange(len(out)):-----------------------------------------------------------------------------------------------------------------------------------------| #5
                out_index = np.empty(len(out_shape), dtype=np.int32)                                                                                                           |
                a_index = np.empty(len(a_shape), dtype=np.int32)                                                                                                               |
                b_index = np.empty(len(b_shape), dtype=np.int32)                                                                                                               |
                to_index(i, out_shape, out_index)                                                                                                                              |
                out_pos = index_to_position(out_index, out_strides)                                                                                                            |
                broadcast_index(out_index, out_shape, a_shape, a_index)                                                                                                        |
                a_pos = index_to_position(a_index, a_strides)                                                                                                                  |
                broadcast_index(out_index, out_shape, b_shape, b_index)                                                                                                        |
                b_pos = index_to_position(b_index, b_strides)                                                                                                                  |
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])                                                                                                          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #4, #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (228) is hoisted
 out of the parallel loop labelled #5 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (229) is hoisted
 out of the parallel loop labelled #5 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_index = np.empty(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (230) is hoisted
 out of the parallel loop labelled #5 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: b_index = np.empty(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (263)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (263)
---------------------------------------------------------------------|loop #ID
    def _reduce(                                                     |
        out: Storage,                                                |
        out_shape: Shape,                                            |
        out_strides: Strides,                                        |
        a_storage: Storage,                                          |
        a_shape: Shape,                                              |
        a_strides: Strides,                                          |
        reduce_dim: int,                                             |
    ) -> None:                                                       |
        # TODO: Implement for Task 3.1.                              |
        reduce_size = a_shape[reduce_dim]                            |
        reduce_stride = a_strides[reduce_dim]                        |
        for i in prange(len(out)):-----------------------------------| #6
            out_index = np.empty(len(out_shape), dtype=np.int32)     |
            to_index(i, out_shape, out_index)                        |
            out_pos = index_to_position(out_index, out_strides)      |
            out_index[reduce_dim] = 0                                |
            a_pos = index_to_position(out_index, a_strides)          |
            for j in range(reduce_size):                             |
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])    |
                a_pos += reduce_stride                               |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #6).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (276) is hoisted
 out of the parallel loop labelled #6 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (288)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /data1/tanseyw/projects/feiyang/mod3-fyng/minitorch/fast_ops.py (288)
----------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                |
    out: Storage,                                                           |
    out_shape: Shape,                                                       |
    out_strides: Strides,                                                   |
    a_storage: Storage,                                                     |
    a_shape: Shape,                                                         |
    a_strides: Strides,                                                     |
    b_storage: Storage,                                                     |
    b_shape: Shape,                                                         |
    b_strides: Strides,                                                     |
) -> None:                                                                  |
    """NUMBA tensor matrix multiply function.                               |
                                                                            |
    Should work for any tensor shapes that broadcast as long as             |
                                                                            |
    ```                                                                     |
    assert a_shape[-1] == b_shape[-2]                                       |
    ```                                                                     |
                                                                            |
    Optimizations:                                                          |
                                                                            |
    * Outer loop in parallel                                                |
    * No index buffers or function calls                                    |
    * Inner loop should have no global writes, 1 multiply.                  |
                                                                            |
                                                                            |
    Args:                                                                   |
    ----                                                                    |
        out (Storage): storage for `out` tensor                             |
        out_shape (Shape): shape for `out` tensor                           |
        out_strides (Strides): strides for `out` tensor                     |
        a_storage (Storage): storage for `a` tensor                         |
        a_shape (Shape): shape for `a` tensor                               |
        a_strides (Strides): strides for `a` tensor                         |
        b_storage (Storage): storage for `b` tensor                         |
        b_shape (Shape): shape for `b` tensor                               |
        b_strides (Strides): strides for `b` tensor                         |
                                                                            |
    Returns:                                                                |
    -------                                                                 |
        None : Fills in `out`                                               |
                                                                            |
    """                                                                     |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                  |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                  |
                                                                            |
    # TODO: Implement for Task 3.2.                                         |
    # only need to consider the 3D case                                     |
    d_shared = a_shape[-1]                                                  |
    for l in prange(len(out)):----------------------------------------------| #7
        j = l % out_shape[2]                                                |
        i = (l // out_shape[2]) % out_shape[1]                              |
        n = l // (out_shape[1] * out_shape[2])                              |
        acc = 0.0                                                           |
        out_pos = n*out_strides[0] + i*out_strides[1] + j*out_strides[2]    |
        for k in range(d_shared):                                           |
            a_pos = n*a_batch_stride + i*a_strides[1] + k*a_strides[2]      |
            b_pos = n*b_batch_stride + k*b_strides[1] + j*b_strides[2]      |
            acc += a_storage[a_pos] * b_storage[b_pos]                      |
        out[out_pos] = acc                                                  |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #7).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```