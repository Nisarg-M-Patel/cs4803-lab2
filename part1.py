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
    #import the get next microbatch, similiar to part 0
    from utils import get_next_microbatch
    #get the model storage and parse into variables
    weights, optStates, activations, activationGrads, grad_weights = model.storage()

    #shard weights for all layers 1/world_size
    for layer in range(model.num_layers):
        weights[layer], optStates[layer] = model.load_weights(
            layer,
            model.rank,
            model.world_size
        )
    
    #forward pass and backpropogate microbatches
    for microbatch in get_next_microbatch(
        model.global_batch_size,
        model.world_size,
        batch_size,
        model.rank
        ):
        #input activations
        activations[0] = model.get_input_activation(microbatch)

        # Forward pass
        for layer in range(model.num_layers):
            #save the shard
            shard = weights[layer]
            #gather full weights
            weights[layer] = await model.all_gather(weights[layer], layer)
            activations[layer + 1] = model.forward(layer, activations[layer], weights[layer])
            #restore shard
            weights[layer] = shard
        
        #loss
        activationGrads[model.num_layers] = model.loss(activations[model.num_layers])
        
        #backward pass
        for layer in range(model.num_layers - 1, -1, -1):
            #save shard
            shard = weights[layer]
            #gather weights
            weights[layer] = await model.all_gather(weights[layer], layer)
            new_gradient, activationGrads[layer] = model.backward(
                layer, activations[layer], activationGrads[layer + 1], weights[layer]
            )
            #restore shard
            weights[layer] = shard
            
            del activationGrads[layer + 1], activations[layer]
            
            #accumulate gradients
            if layer not in grad_weights:
                grad_weights[layer] = new_gradient
            else:
                grad_weights[layer] += new_gradient
    
    #reduce-scatter gradients
    for layer in range(model.num_layers):
        grad_weights[layer] = await model.reduce_scatter(grad_weights[layer], layer)
    
    #update param
    for layer in range(model.num_layers):
        weights[layer], optStates[layer] = model.update(
            layer, grad_weights[layer], weights[layer], optStates[layer]
        )
        model.set_final_weight(layer, weights[layer])
    
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
