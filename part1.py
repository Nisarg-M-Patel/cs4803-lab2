import asyncio
from typing import List

from lib import Dist, Model
from utils import (
    get_model_theoretical_best_time,
    get_training_stats,
    write_chrome_trace,
)


async def part1_training_loop(model: Model, batch_size: int) -> Model:
    """
    Implement distributed training using ZeRO (Zero Redundancy Optimizer) optimization.

    Shard model parameters, gradients, and optimizer states across GPUs (ranks).
    Coordinate parameter updates and gradient reductions across GPUs.
    Implement efficient memory management by only creating the full model params when needed
    """
    raise NotImplementedError


async def main():
    world_size = 32
    num_layers = 64
    global_batch_size = 256
    batch_size = 8

    dist = Dist(world_size)
    models: List[Model] = [
        Model(rank, dist, num_layers, global_batch_size) for rank in range(world_size)
    ]

    theoretical_time = get_model_theoretical_best_time(models[0])

    out = await asyncio.gather(
        *(part1_training_loop(model, batch_size) for model in models)
    )

    execution_time, max_memory, max_memory_idx = get_training_stats(out)

    mfu = (theoretical_time / execution_time) * 100

    print(f"MFU: {mfu}")

    write_chrome_trace(out, "./debug_traces/part1.json")


if __name__ == "__main__":
    asyncio.run(main())
