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
    from utils import get_next_microbatch
    
    weights, opt_states, activations, grad_activations, grad_weights = model.storage()
    
    for l in range(model.num_layers):
        weights[l], opt_states[l] = model.load_weights(
            l, model.rank, model.world_size
        )
    
    for microbatch in get_next_microbatch(
        model.global_batch_size, model.world_size, batch_size, model.rank
    ):
        activations[0] = model.get_input_activation(microbatch)
        
        for l in range(model.num_layers):
            weight_shard = weights[l]
            weights[l] = await model.all_gather(weights[l], l)
            activations[l + 1] = model.forward(l, activations[l], weights[l])
            weights[l] = weight_shard
        
        grad_activations[model.num_layers] = model.loss(activations[model.num_layers])
        
        for l in range(model.num_layers - 1, -1, -1):
            weight_shard = weights[l]
            weights[l] = await model.all_gather(weights[l], l)
            new_grad, grad_activations[l] = model.backward(
                l, activations[l], grad_activations[l + 1], weights[l]
            )
            weights[l] = weight_shard
            del grad_activations[l + 1], activations[l]
            new_grad = await model.reduce_scatter(new_grad, l)
            if l not in grad_weights:
                grad_weights[l] = new_grad
            else:
                grad_weights[l] += new_grad
    
    for l in range(model.num_layers):
        weights[l], opt_states[l] = model.update(
            l, grad_weights[l], weights[l], opt_states[l]
        )
        model.set_final_weight(l, weights[l])
    
    return model


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
