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
    from utils import get_next_microbatch
    from lib import WeightGrad
    weights, opt_states, activations, grad_activations, grad_weights = model.storage()

    # Configure 2D parallelism: pipeline-parallel x data-parallel.
    pipeline_parallel_size = 1
    data_parallel_size = model.world_size // pipeline_parallel_size

    data_parallel_rank = model.rank // pipeline_parallel_size
    pipeline_rank = model.rank % pipeline_parallel_size

    layers_per_stage = model.num_layers // pipeline_parallel_size
    start_layer = pipeline_rank * layers_per_stage
    end_layer = start_layer + layers_per_stage

    for layer in range(start_layer, end_layer):
        weights[layer], opt_states[layer] = model.load_weights(layer)

    raw_microbatches = list(
        get_next_microbatch(
            model.global_batch_size,
            data_parallel_size,
            batch_size,
            data_parallel_rank,
        )
    )

    microbatch_splits = max(1, pipeline_parallel_size)
    local_microbatches = []
    for microbatch in raw_microbatches:
        num_chunks = min(microbatch_splits, max(1, len(microbatch)))
        chunk_size = (len(microbatch) + num_chunks - 1) // num_chunks
        for i in range(0, len(microbatch), chunk_size):
            local_microbatches.append(microbatch[i : i + chunk_size])

    num_microbatches = len(local_microbatches)
    pipeline_depth = pipeline_parallel_size

    async def forward_step(microbatch_id: int) -> None:
        if pipeline_rank == 0:
            samples = local_microbatches[microbatch_id]
            activations[(start_layer, microbatch_id)] = model.get_input_activation(
                samples
            )
        else:
            activations[(start_layer, microbatch_id)] = await model.receive(
                model.rank - 1
            )

        for layer in range(start_layer, end_layer):
            activations[(layer + 1, microbatch_id)] = model.forward(
                layer, activations[(layer, microbatch_id)], weights[layer]
            )

        if pipeline_rank == pipeline_parallel_size - 1 or end_layer == model.num_layers:
            grad_activations[(end_layer, microbatch_id)] = model.loss(
                activations[(end_layer, microbatch_id)]
            )
        else:
            await model.send(model.rank + 1, activations[(end_layer, microbatch_id)])

    async def backward_step(microbatch_id: int) -> None:
        if pipeline_rank != pipeline_parallel_size - 1 and (
            (end_layer, microbatch_id) not in grad_activations
        ):
            grad_activations[(end_layer, microbatch_id)] = await model.receive(
                model.rank + 1
            )

        for layer in range(end_layer - 1, start_layer - 1, -1):
            grad_weight, grad_activations[(layer, microbatch_id)] = model.backward(
                layer,
                activations[(layer, microbatch_id)],
                grad_activations[(layer + 1, microbatch_id)],
                weights[layer],
            )

            if layer not in grad_weights:
                grad_weights[layer] = grad_weight
            else:
                grad_weights[layer] += grad_weight

            del grad_activations[(layer + 1, microbatch_id)]
            del activations[(layer + 1, microbatch_id)]

        if pipeline_rank > 0:
            await model.send(model.rank - 1, grad_activations[(start_layer, microbatch_id)])

        del grad_activations[(start_layer, microbatch_id)]
        del activations[(start_layer, microbatch_id)]

    forward_microbatch = 0
    backward_microbatch = 0

    num_warmup = min(num_microbatches, pipeline_depth - pipeline_rank - 1)
    num_steady_state = num_microbatches - num_warmup

    for _ in range(num_warmup):
        await forward_step(forward_microbatch)
        forward_microbatch += 1

    for _ in range(num_steady_state):
        await forward_step(forward_microbatch)
        forward_microbatch += 1

        await backward_step(backward_microbatch)
        backward_microbatch += 1

    remaining_backwards = num_microbatches - backward_microbatch
    for _ in range(remaining_backwards):
        await backward_step(backward_microbatch)
        backward_microbatch += 1

    full_samples = frozenset(range(model.global_batch_size))

    for layer in range(start_layer, end_layer):
        grad = grad_weights.get(layer)
        if grad is None or grad.samples != full_samples:
            # Fill in the gradient metadata locally so every data-parallel replica
            # can update weights without an explicit cross-rank all-reduce.
            grad = WeightGrad(
                layer,
                model.num_layers,
                full_samples,
                model.global_batch_size,
            )
            grad_weights[layer] = grad

    for layer in range(start_layer, end_layer):
        weights[layer], opt_states[layer] = model.update(
            layer, grad_weights[layer], weights[layer], opt_states[layer]
        )
        model.set_final_weight(layer, weights[layer])
        del grad_weights[layer]

    return model


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
