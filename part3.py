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
    from utils import get_global_batch_list
    weights, opt_states, activations, grad_activations, grad_weights = model.storage()

    layers_per_rank = model.num_layers // model.world_size
    start_layer = model.rank * layers_per_rank
    end_layer = start_layer + layers_per_rank

    for layer in range(start_layer, end_layer):
        weights[layer], opt_states[layer] = model.load_weights(layer)

    microbatches = get_global_batch_list(model.global_batch_size, batch_size)
    num_microbatches = len(microbatches)
    pipeline_size = model.world_size

    num_warmup_microbatches = min(
        num_microbatches, pipeline_size - model.rank - 1
    )
    num_steady_state_iters = num_microbatches - num_warmup_microbatches

    async def forward_step(microbatch_id: int) -> None:
        if model.rank == 0:
            samples = microbatches[microbatch_id]
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

        if model.rank == pipeline_size - 1:
            grad_activations[(end_layer, microbatch_id)] = model.loss(
                activations[(end_layer, microbatch_id)]
            )
        else:
            await model.send(model.rank + 1, activations[(end_layer, microbatch_id)])

    async def backward_step(microbatch_id: int) -> None:
        if model.rank != pipeline_size - 1 and (
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

        if model.rank > 0:
            await model.send(
                model.rank - 1, grad_activations[(start_layer, microbatch_id)]
            )

        del grad_activations[(start_layer, microbatch_id)]
        del activations[(start_layer, microbatch_id)]

    forward_microbatch = 0
    backward_microbatch = 0

    for _ in range(num_warmup_microbatches):
        await forward_step(forward_microbatch)
        forward_microbatch += 1

    for _ in range(num_steady_state_iters):
        await forward_step(forward_microbatch)
        forward_microbatch += 1

        await backward_step(backward_microbatch)
        backward_microbatch += 1

    remaining_backwards = num_microbatches - backward_microbatch
    for _ in range(remaining_backwards):
        await backward_step(backward_microbatch)
        backward_microbatch += 1

    for layer in range(start_layer, end_layer):
        weights[layer], opt_states[layer] = model.update(
            layer, grad_weights[layer], weights[layer], opt_states[layer]
        )
        model.set_final_weight(layer, weights[layer])

    return model

async def main():
    world_size = 4
    num_layers = 4
    global_batch_size = 512
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
        *[part3_training_loop(models[i], batch_size) for i in range(world_size)]
    )

    execution_time, max_memory, arg_max_memory = get_training_stats(out)

    mfu = (theoretical_time / execution_time) * 100
    print(f"MFU: {mfu}")

    write_chrome_trace(out, "./debug_traces/part3.json")


if __name__ == "__main__":
    asyncio.run(main())
