import asyncio
from typing import List

from lib import Dist, Model
from utils import (
    get_model_theoretical_best_time,
    get_training_stats,
    write_chrome_trace,
)


async def part4_training_loop(model: Model, batch_size: int) -> Model:
    """
    There is not currently a one size fits all approach for distributed training.
    The right choice will depend on the constants such as batch size, memory per GPU, communication overhead, implementation complexity, model size, and specifics of architecture.

    Implement a distributed training approach that you think will be most effective. The approach will be evaluated and ranked on the leaderboard based on time.
    You may use any of the techniques from the previous parts or any combination of them. You may also implement a new approach from scratch if you think it will be most effective.

    Things to try possibly try out but not mentioned in the previous parts:
    - Interleaved 1F1B (One Forward One Backward) Scheduling - See paper for more details.
    - Different Data parallelism and Pipeline parallelism degree. We can experiment with how many degrees we want to see which cases reduce time and memory.
    - Gradient Accumulation - Accumulate gradients over multiple batches and update weights only once
    - Gradient Checkpointing - Only store activations for a few layers and recompute them during backward pass
    """
    raise NotImplementedError


async def main():
    world_size = 32
    num_layers = 64
    global_batch_size = 2048
    batch_size = 64

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
        *[part4_training_loop(models[i], batch_size) for i in range(world_size)]
    )

    execution_time, max_memory, arg_max_memory = get_training_stats(out)

    mfu = (theoretical_time / execution_time) * 100
    print(f"MFU: {mfu}")

    write_chrome_trace(out, "./debug_traces/part4.json")


if __name__ == "__main__":
    asyncio.run(main())
