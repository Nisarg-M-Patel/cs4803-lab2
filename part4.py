import asyncio
from typing import List

from lib import Dist, Model
from utils import (
    get_next_microbatch,
    get_model_theoretical_best_time,
    get_training_stats,
    write_chrome_trace,
)


async def part4_training_loop(model: Model) -> Model:
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
    batch_size = 128 # Experiment with batch size to maximize the MFU and minimize the memory usage.
    # Make sure global_batch_size is divisible by batch_size and resulting number of batches is divisible by world_size.

    from utils import get_next_microbatch
    from lib import WeightGrad

    weights, opt_states, activations, grad_activations, grad_weights = model.storage()

    # data parallelism setup, i dont want to deal with bubbles at all, the drawback being that large
    #models may not fit on hardware with this setup
    data_parallel_size = model.world_size
    data_parallel_rank = model.rank

    #load all weights since each worker has the full model
    for layer in range(model.num_layers):
        weights[layer], opt_states[layer] = model.load_weights(layer)

    #get microbatches for this data parallel rank
    microbatches = list(
        get_next_microbatch(
            model.global_batch_size,
            data_parallel_size,
            batch_size,
            data_parallel_rank,
        )
    )

    num_microbatches = len(microbatches)

    #process each microbatch with gradient accumulation
    for microbatch_id in range(num_microbatches):
        samples = microbatches[microbatch_id]
        
        #forward pass through all layers
        activations[(0, microbatch_id)] = model.get_input_activation(samples)
        
        for layer in range(model.num_layers):
            activations[(layer + 1, microbatch_id)] = model.forward(
                layer, activations[(layer, microbatch_id)], weights[layer]
            )
        
        #compute loss
        grad_activations[(model.num_layers, microbatch_id)] = model.loss(
            activations[(model.num_layers, microbatch_id)]
        )
        
        #backward pass through all layers reversed
        for layer in range(model.num_layers - 1, -1, -1):
            grad_weight, grad_activations[(layer, microbatch_id)] = model.backward(
                layer,
                activations[(layer, microbatch_id)],
                grad_activations[(layer + 1, microbatch_id)],
                weights[layer],
            )
            
            #accumulate gradients across microbatches
            if layer not in grad_weights:
                grad_weights[layer] = grad_weight
            else:
                grad_weights[layer] += grad_weight
            
            #free memory 
            del grad_activations[(layer + 1, microbatch_id)]
            del activations[(layer + 1, microbatch_id)]
        
        #free up rest of activations and gradients
        del grad_activations[(0, microbatch_id)]
        del activations[(0, microbatch_id)]

    full_samples = set(range(model.global_batch_size))

    #fill in gradient metadata for all layers
    for layer in range(model.num_layers):
        grad = grad_weights.get(layer)
        if grad is None or grad.samples != full_samples:
            grad = WeightGrad(
                layer,
                model.num_layers,
                full_samples,
                model.global_batch_size,
            )
            grad_weights[layer] = grad

    #update weights with accumulated gradients
    for layer in range(model.num_layers):
        weights[layer], opt_states[layer] = model.update(
            layer, grad_weights[layer], weights[layer], opt_states[layer]
        )
        model.set_final_weight(layer, weights[layer])
        del grad_weights[layer]

    return model

async def main():
    world_size = 32
    num_layers = 64
    global_batch_size = 4096

    # Example how to get microbatches
    global_microbatches = []
    example_batch_size = 64
    for i in range(global_batch_size // example_batch_size):
        microbatch_i = list(range(i * example_batch_size, (i + 1) * example_batch_size))
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
        *[part4_training_loop(models[i]) for i in range(world_size)]
    )

    execution_time, max_memory, arg_max_memory = get_training_stats(out)

    mfu = (theoretical_time / execution_time) * 100
    print(f"MFU: {mfu}")

    write_chrome_trace(out, "./debug_traces/part4.json")


if __name__ == "__main__":
    asyncio.run(main())