import json
import os
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

from lib import BLOCK_SIZE, HIDDEN_SIZE, Model


def get_next_microbatch(
    global_batch_size: int, world_size: int, batch_size: int, rank: int
) -> Generator[List[int], None, None]:
    """
    Yield the next batch for the given rank.
    """
    num_samples_per_rank = global_batch_size // world_size

    all_rank_samples = list(
        range(num_samples_per_rank * rank, num_samples_per_rank * (rank + 1))
    )

    for i in range(0, len(all_rank_samples), batch_size):
        yield all_rank_samples[i : i + batch_size]


def check(models: Sequence[Model]) -> None:
    for l in range(models[0].num_layers):
        weight = None
        for m in models:
            if l in m.final_weights:
                assert m.final_weights[l].step == 1
                if weight is None:
                    weight = m.final_weights[l]
                else:
                    weight = weight.combine(m.final_weights[l])

        assert weight is not None, f"Missing weight {l}"
        assert weight.is_complete(), f"Weight not complete {weight}"


def get_training_stats(models: Sequence[Model]) -> Tuple[int, int, int]:
    check(models)

    execution_time = max(m.log[-1].time for m in models)
    max_memory = 0
    arg_max_memory = None
    for i, m in enumerate(models):
        for event in m.log:
            if event.memory > max_memory:
                max_memory = event.memory
                arg_max_memory = (i, event.time)
    print(
        f"Execution time: {execution_time} \nMaximum Memory: {max_memory} at GPU: {arg_max_memory[0]} time: {arg_max_memory[1]}"
    )
    return execution_time, max_memory, arg_max_memory[0]


def get_batch_labeled_name(event: str, batch_id: int) -> str:
    if batch_id == -1:
        return event
    event = event.split(",")[0]
    event = event[1:]
    event += f" batch: {batch_id}"

    return event


def get_batch_id(global_batch_list: List[List[int]], samples: List[int]) -> str:
    if global_batch_list is None:
        return -1
    try:
        batch_idx = global_batch_list.index(samples)
    except ValueError:
        return -1
    return batch_idx


def to_chrome_trace(
    models: Sequence[Model], global_batch_list: Optional[List[List[int]]] = None
) -> Dict[str, Any]:
    events = []
    for i, m in enumerate(models):
        for event in m.log:
            # Add the original event
            batch_id = get_batch_id(global_batch_list, list(event.samples))
            events.append(
                {
                    "name": get_batch_labeled_name(str(event), batch_id),
                    "ph": "X",
                    "ts": event.time * 1_000_000,
                    "dur": event.duration * 1_000_000,
                    "pid": f"GPU {i}",
                    "tid": "Execution",
                    "args": {
                        "memory": event.memory,
                        "samples": list(event.samples),
                        "layer": event.layer,
                        "batch_id": (
                            str(batch_id)
                            if batch_id != -1
                            else "Across Batch Operation"
                        ),
                    },
                }
            )

            # Add a counter event for memory
            events.append(
                {
                    "name": "memory",
                    "ph": "C",
                    "ts": event.time * 1_000_000,
                    "pid": f"GPU {i}",
                    "tid": "Memory",
                    "args": {"memory": event.memory},
                }
            )

    return {"traceEvents": events}


def write_chrome_trace(
    models: Sequence[Model],
    path: str,
    global_batch_list: Optional[List[List[int]]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(to_chrome_trace(models, global_batch_list), f)


def get_model_theoretical_best_time(model: Model) -> float:
    total_flops = (
        4 * HIDDEN_SIZE * HIDDEN_SIZE * model.global_batch_size * BLOCK_SIZE * model.num_layers
    )

    gpu_max_flops = 1e15

    theoretical_best_time = total_flops / (gpu_max_flops * model.world_size)

    return theoretical_best_time


def get_global_batch_list(global_batch_size, batch_size) -> List[List[int]] :
    global_microbatches = []
    for i in range(global_batch_size // batch_size):
        microbatch_i = list(range(i * batch_size, (i + 1) * batch_size))
        global_microbatches.append(microbatch_i)
    return global_microbatches
