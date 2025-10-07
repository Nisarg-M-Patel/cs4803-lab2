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
    weights, opt_states, activations, grad_activations, grad_weights = model.storage()
    
    #split up layers per rank equally
    layers_per_rank = model.num_layers // model.world_size
    start_layer = model.rank * layers_per_rank
    end_layer = start_layer + layers_per_rank
    
    for l in range(start_layer, end_layer):
        weights[l], opt_states[l] = model.load_weights(l)
    
    all_samples = list(range(model.global_batch_size))
    
    #if the models rank is 0 it has the first set of layers so we can get input activations and forward
    if model.rank == 0:
        activations[start_layer] = model.get_input_activation(all_samples)
    else:
        activations[start_layer] = await model.receive(model.rank - 1)
    
    #run the forward pass for our set of layers and send them to tnext rank
    for l in range(start_layer, end_layer):
        activations[l + 1] = model.forward(l, activations[l], weights[l])
    
    if model.rank == model.world_size - 1:
        grad_activations[end_layer] = model.loss(activations[end_layer])
    else:
        await model.send(model.rank + 1, activations[end_layer])
    #last rank, dont send begin to backprop
    if model.rank == model.world_size - 1:
        pass
    else:
        grad_activations[end_layer] = await model.receive(model.rank + 1)
    #backprop and send to previous rank
    for l in range(end_layer - 1, start_layer - 1, -1):
        grad_weights[l], grad_activations[l] = model.backward(
            l, activations[l], grad_activations[l + 1], weights[l]
        )
        del grad_activations[l + 1]
    
    if model.rank > 0:
        await model.send(model.rank - 1, grad_activations[start_layer])
    #update
    for l in range(start_layer, end_layer):
        weights[l], opt_states[l] = model.update(
            l, grad_weights[l], weights[l], opt_states[l]
        )
        model.set_final_weight(l, weights[l])
    
    return model



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
