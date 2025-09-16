import asyncio
from typing import List

from lib import Dist, Model
from utils import (
    get_model_theoretical_best_time,
    get_training_stats,
    write_chrome_trace,
)


async def part2_training_loop(model: Model) -> Model:
    """
    Implement distributed training using Pipeline parallelism.

    Split (shard) the layers and optimizer states equally between GPUs (ranks).
    Pass the full set of batches for activations and gradient activations between layers.
    Additionally, explore how passing only a subset of batches for activations and gradient activations between layers affects performance (in terms of memory usage and e2e time)

    Coordinate the forward and backward passes across multiple GPUs in a pipelined manner. Verify the pipelined nature of the training using `visualize.py`.

    Use `model.send` and `model.receive` to pass data between different ranks. See `lib.py` for more details on these functions.
    """
    raise NotImplementedError


async def main():
    world_size = 4
    num_layers = 16
    global_batch_size = 2048

    dist = Dist(world_size)
    models: List[Model] = [
        Model(rank, dist, num_layers, global_batch_size) for rank in range(world_size)
    ]

    theoretical_time = get_model_theoretical_best_time(models[0])

    out = await asyncio.gather(*(part2_training_loop(model) for model in models))

    time, memory, max_mem_rank = get_training_stats(out)

    mfu = (theoretical_time / time) * 100

    print(f"MFU: {mfu}")

    write_chrome_trace(out, "./debug_traces/part2.json")

    print(time, memory, max_mem_rank)


if __name__ == "__main__":
    asyncio.run(main())
