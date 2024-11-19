import random
from collections import defaultdict
import minitorch
import time
import sys
import numpy as np
import seaborn as sns
import pandas as pd

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend: minitorch.TensorBackend, size: int =16) -> None:
    '''Perform matrix multiplication using the given backend'''
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    x @ y


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 5
    times = []
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu
            
            times.append(
                {"size": size, "backend": "fast", "time (s)": fast_time}
            )
            times.append(
                {"size": size, "backend": "gpu", "time (s)": gpu_time}
            )

    df = pd.DataFrame(times)
    g = sns.lineplot(
        data=df, x="size", y="time (s)", 
        hue="backend", palette="viridis"
    )
    g.set(yscale='log')
    fig = g.get_figure()
    fig.savefig("plot/benchmark.png")