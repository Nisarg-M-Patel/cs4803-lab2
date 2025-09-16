import asyncio
from typing import List

from lib import Dist, Model
from utils import (
    get_model_theoretical_best_time,
    get_training_stats,
    write_chrome_trace,
)


async def part3_training_loop(model: Model, batch_size: int) -> Model:
    """
    Implement distributed training using 1F1B  (1 forward pass, 1 backward pass) scheduling.
    The default 1F1B strategy should be implemented. See paper here : https://arxiv.org/pdf/2104.04473

    We will need to split layers across world_size. Each rank would work on a subset of layers but all the batches.

    It is guaranteed that the number of layers is divisible by the number of world_size. (i.e. num_layers % world_size == 0)
    It is also guaranteed that the number of batches is divisible by the number of world_size. (i.e. num_batches % world_size == 0)
    Coordinate the forward and backward passes across multiple GPUs in a pipelined manner. Verify the pipelined nature of the training using `visualize.ipynb`.

    Use `model.send` and `model.receive` to pass data between different world_size. See `lib.py` for more details on these functions.
    """
    raise NotImplementedError


async def main():
    world_size = 16
    num_layers = 32
    global_batch_size = 2048
    batch_size = 128

    global_microbatches = []
    for i in range(global_batch_size // batch_size):
        microbatch_i = list(range(i * batch_size, (i + 1) * batch_size))
        global_microbatches.append(microbatch_i)

    dist = Dist(world_size)
    models: List[Model] = [
        Model(
            num_layers=num_layers,
            global_batch_size=global_batch_size,
            rank=i,
            dist=dist,
        )
        for i in range(world_size)
    ]

    theoretical_time = get_model_theoretical_best_time(models[0])

    out = await asyncio.gather(
        *[part3_training_loop(models[i], batch_size) for i in range(world_size)]
    )

    execution_time, max_memory, arg_max_memory = get_training_stats(out)

    mfu = (theoretical_time / execution_time) * 100
    print(f"MFU: {mfu}")

    write_chrome_trace(out, "./debug_traces/part3.json")


if __name__ == "__main__":
    asyncio.run(main())
