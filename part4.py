import asyncio
from typing import List, Dict
import math
from collections import defaultdict


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
    
    import math
    from lib import WeightGrad
    from utils import get_next_microbatch

    # Setup ZeRO style storage similar to part1 with gradient accumulation.
    weights, opt_states, activations, grad_activations, grad_weights = model.storage()

    # Load sharded weights and optimizer states.
    for layer in range(model.num_layers):
        weights[layer], opt_states[layer] = model.load_weights(
            layer, model.rank, model.world_size
        )

    # Decide how many microbatches to accumulate before communicating gradients.
    samples_per_rank = model.global_batch_size // model.world_size
    target_accumulation = max(1, batch_size // 16)

    # Ensure we have at least one microbatch and not more accumulation steps than available.
    micro_batch_size = max(1, batch_size // max(1, target_accumulation))
    total_microbatches = math.ceil(samples_per_rank / micro_batch_size)
    accumulation_steps = max(1, min(target_accumulation, total_microbatches))

    # Gradient checkpointing configuration. Keep boundaries and periodic checkpoints.
    checkpoint_interval = max(1, int(math.sqrt(model.num_layers)))

    def should_checkpoint(layer_idx: int) -> bool:
        if layer_idx <= 0 or layer_idx >= model.num_layers:
            return True
        return layer_idx % checkpoint_interval == 0

    async def ensure_activation(layer_idx: int) -> None:
        if layer_idx in activations:
            return

        # Find nearest stored checkpoint below the requested layer.
        start_layer = layer_idx - 1
        while start_layer >= 0 and start_layer not in activations:
            start_layer -= 1
        assert start_layer >= 0, "Input activation must exist for recomputation"

        current_activation = activations[start_layer]
        for recompute_layer in range(start_layer, layer_idx):
            weight_shard = weights[recompute_layer]
            weights[recompute_layer] = await model.all_gather(
                weights[recompute_layer], recompute_layer
            )
            next_activation = model.forward(
                recompute_layer, current_activation, weights[recompute_layer]
            )
            weights[recompute_layer] = weight_shard

            if should_checkpoint(recompute_layer + 1) or recompute_layer + 1 == layer_idx:
                activations[recompute_layer + 1] = next_activation

            current_activation = next_activation

    # Buffers for accumulating full gradients before reduce-scatter.
    pending_full_grads: dict[int, WeightGrad] = {}

    for microbatch_idx, microbatch in enumerate(
        get_next_microbatch(
            model.global_batch_size, model.world_size, micro_batch_size, model.rank
        )
    ):
        activations.clear()
        grad_activations.clear()
        activations[0] = model.get_input_activation(microbatch)

        for layer in range(model.num_layers):
            weight_shard = weights[layer]
            weights[layer] = await model.all_gather(weights[layer], layer)
            activations[layer + 1] = model.forward(layer, activations[layer], weights[layer])
            weights[layer] = weight_shard

            # Drop activations that are not checkpoints to enable recomputation.
            if not should_checkpoint(layer) and layer in activations:
                del activations[layer]
            if not should_checkpoint(layer + 1) and (layer + 1) != model.num_layers:
                del activations[layer + 1]

        grad_activations[model.num_layers] = model.loss(activations[model.num_layers])

        for layer in range(model.num_layers - 1, -1, -1):
            await ensure_activation(layer)

            weight_shard = weights[layer]
            weights[layer] = await model.all_gather(weights[layer], layer)
            new_grad, grad_activations[layer] = model.backward(
                layer, activations[layer], grad_activations[layer + 1], weights[layer]
            )
            weights[layer] = weight_shard

            del grad_activations[layer + 1]

            if not should_checkpoint(layer) and layer in activations:
                del activations[layer]

            if layer not in pending_full_grads:
                pending_full_grads[layer] = new_grad
            else:
                pending_full_grads[layer] += new_grad

        should_flush = (
            (microbatch_idx + 1) % accumulation_steps == 0
            or microbatch_idx + 1 == total_microbatches
        )

        if should_flush:
            for layer, full_grad in list(pending_full_grads.items()):
                shard_grad = await model.reduce_scatter(full_grad, layer)
                if layer not in grad_weights:
                    grad_weights[layer] = shard_grad
                else:
                    grad_weights[layer] += shard_grad
                del pending_full_grads[layer]

    for layer in range(model.num_layers):
        weights[layer], opt_states[layer] = model.update(
            layer, grad_weights[layer], weights[layer], opt_states[layer]
        )
        model.set_final_weight(layer, weights[layer])

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
