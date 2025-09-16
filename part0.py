import asyncio
from typing import List

from lib import Dist, Model
from utils import (
    get_model_theoretical_best_time,
    get_next_microbatch,
    get_training_stats,
    write_chrome_trace,
)


async def train_ddp(model: Model, batch_size: int) -> Model:
    """
    We will perform distributed data parallel training in the function below.
    """
    # Storage on device.
    weights, opt_states, activations, grad_activations, grad_weights = model.storage()

    # Load in weights for all layers
    for l in range(model.num_layers):
        weights[l], opt_states[l] = model.load_weights(l)

    for microbatch in get_next_microbatch(
        model.global_batch_size, model.world_size, batch_size, model.rank
    ):
        # Load all the activations for this particular rank (GPU)
        activations[0] = model.get_input_activation(microbatch)

        # Forward pass using the previous activations
        for l in range(model.num_layers):
            activations[l + 1] = model.forward(l, activations[l], weights[l])

        # Backward
        grad_activations[model.num_layers] = model.loss(activations[model.num_layers])

        for l in range(model.num_layers - 1, -1, -1):
            grad_weights[l], grad_activations[l] = model.backward(
                l, activations[l], grad_activations[l + 1], weights[l]
            )
            # Remember to delete the activations to save memory
            del grad_activations[l + 1], activations[l]

        for l in range(model.num_layers):
            grad_weights[l] += await model.all_reduce(grad_weights[l], l)

    for l in range(model.num_layers):
        weights[l], opt_states[l] = model.update(
            l, grad_weights[l], weights[l], opt_states[l]
        )
        model.set_final_weight(l, weights[l])

    return model


async def main():
    world_size = 8
    num_layers = 36
    global_batch_size = 32
    batch_size = 4

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

    theoretical_best_time = get_model_theoretical_best_time(models[0])

    out = await asyncio.gather(
        *[
            train_ddp(
                models[i],
                batch_size,
            )
            for i in range(world_size)
        ]
    )

    time, memory, max_mem_rank = get_training_stats(out)

    mfu = (theoretical_best_time / time) * 100

    print(mfu)

    write_chrome_trace(out, "./debug_traces/part0.json")


if __name__ == "__main__":
    asyncio.run(main())
