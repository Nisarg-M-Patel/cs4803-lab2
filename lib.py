from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

from calculon.memory import Memory
from calculon.network import Network
from calculon.processor import Processor

HIDDEN_SIZE = 4096
BLOCK_SIZE = 1024
DATATYPE = "float16"
BYTES_PER_ELEMENT = 2


class Barrier:
    """Sync across n world_size"""

    def __init__(self, target: int):
        self.counter = 0
        self.target = target
        self.lock = asyncio.Lock()
        self.round = 0
        self.done = 0

    async def wait(self, rank: int) -> None:
        while self.done > 0:
            await asyncio.sleep(0.01)
        async with self.lock:
            self.counter += 1
        while self.counter < self.target:
            await asyncio.sleep(0.01)
        self.done += 1
        if rank == 0:
            await self.reset()

    async def reset(self) -> None:
        while self.done < self.target:
            await asyncio.sleep(0.01)
        self.counter = 0
        self.done = 0


T = TypeVar("T")


class Reduceable(Protocol[T]):
    """
    A type that can be reduced.
    """

    def __add__(self, other: T) -> T: ...


O = TypeVar("O")


class Gatherable(Protocol[O]):
    """
    A type that can be sharded.
    """

    def shard(self, shard: int, num_shards: int) -> O: ...

    def is_complete(self) -> bool: ...

    def combine(self, other: O) -> O: ...


TO = TypeVar("TO")


class ReduceableGatherable(Reduceable[TO], Gatherable[TO]):
    pass


class Dist:
    def __init__(self, world_size: int) -> None:
        self.reduce: Optional[Any] = None
        self.gather: Optional[Any] = None
        self.world_size = world_size
        self.barrier = Barrier(world_size)
        self.queue: Dict[Tuple[int, int], asyncio.Queue[Any]] = {
            (i, j): asyncio.Queue(maxsize=1)
            for i in range(world_size)
            for j in range(world_size)
            if i != j
        }
        self.mtime = 0

    async def all_reduce(self, rank: int, inp: T, time: int) -> Tuple[T, int]:
        if self.reduce is None:
            self.reduce = inp
        else:
            self.reduce = self.reduce + inp
        self.mtime = max(time, self.mtime)
        await self.barrier.wait(rank)
        q: T = self.reduce
        mtime = self.mtime
        await self.barrier.wait(rank)
        if rank == 0:
            self.reduce = None
            self.mtime = 0
        await self.barrier.wait(rank)
        return q, mtime

    async def all_gather(self, rank: int, inp: O, time: int) -> Tuple[O, int]:
        if self.gather is None:
            self.gather = inp
        else:
            assert type(self.gather) == type(inp)
            self.gather = self.gather.combine(inp)
        self.mtime = max(time, self.mtime)
        await self.barrier.wait(rank)
        q: O = self.gather
        mtime = self.mtime
        await self.barrier.wait(rank)
        if rank == 0:
            self.gather = None
            self.mtime = 0
        await self.barrier.wait(rank)
        return q, mtime

    async def reduce_scatter(self, rank: int, inp: TO, time: int) -> Tuple[TO, int]:
        x, time = await self.all_reduce(rank, inp, time)
        y = x.shard(rank, self.world_size)  # type: ignore
        return y, time  # type: ignore

    async def receive(self, rank_source: int, rank_dest: int) -> Any:
        return await self.queue[(rank_source, rank_dest)].get()

    async def send(self, rank_source: int, rank_dest: int, v: Any) -> None:
        await self.queue[(rank_source, rank_dest)].put(v)


@dataclass
class Weight(Gatherable["Weight"]):
    """
    The weights for a specific layer. Can be sharded.

    Required for forward and backward passes.
    """

    layer: int
    layers: int
    step: int
    shards: FrozenSet[int] = frozenset([0])
    num_shards: int = 1

    def combine(self, other: Weight) -> Weight:
        return Weight(
            self.layer,
            self.layers,
            self.step,
            self.shards | other.shards,
            self.num_shards,
        )

    def memory(self) -> float:
        return (len(self.shards) / self.num_shards) * HIDDEN_SIZE * HIDDEN_SIZE

    def shard(self, shard: int, num_shards: int) -> Weight:
        assert self.is_complete()
        assert shard < num_shards
        return Weight(
            self.layer, self.layers, self.step, frozenset([shard]), num_shards
        )

    def is_complete(self) -> bool:
        return len(self.shards) == self.num_shards


@dataclass
class Activation:
    """
    Activations need for a specific layer for a specific set of samples.
    """

    layer: int
    layers: int
    samples: FrozenSet[int]
    total_samples: int

    def memory(self) -> int:
        return len(self.samples) * HIDDEN_SIZE * BLOCK_SIZE


@dataclass
class WeightGrad(Reduceable["WeightGrad"], Gatherable["WeightGrad"]):
    """
    The gradient of the loss for a specific weight layer.

    May be sharded to correspond to different parts of the weights.

    May be split into different samples.
    """

    layer: int
    layers: int
    samples: FrozenSet[int]
    total_samples: int
    shards: FrozenSet[int] = frozenset([0])
    num_shards: int = 1

    def __add__(self, other: WeightGrad) -> WeightGrad:
        assert self.layer == other.layer, "Only add same layer weight grads"
        assert self.shards == other.shards
        return WeightGrad(
            self.layer,
            self.layers,
            self.samples | other.samples,
            self.total_samples,
            self.shards,
            self.num_shards,
        )

    def combine(self, other: WeightGrad) -> WeightGrad:
        return WeightGrad(
            self.layer,
            self.layers,
            self.samples,
            self.total_samples,
            self.shards | other.shards,
            self.num_shards,
        )

    def memory(self) -> float:
        return (len(self.shards) / self.num_shards) * HIDDEN_SIZE * HIDDEN_SIZE

    def shard(self, shard: int, num_shards: int) -> WeightGrad:
        assert self.is_complete(), f"{self.shards} out of {self.num_shards}"
        assert shard < num_shards
        return WeightGrad(
            self.layer,
            self.layers,
            self.samples,
            self.total_samples,
            frozenset([shard]),
            num_shards,
        )

    def is_complete(self) -> bool:
        return len(self.shards) == self.num_shards


@dataclass
class OptState(Gatherable["OptState"]):
    """
    The state of the optimizer for a specific layer. Can be sharded.

    In pratice this represents ADAM's saved values needed for optimization.

    Required for updating the weights.
    """

    layer: int
    layers: int
    step: int
    shards: FrozenSet[int] = frozenset(
        [
            0,
        ]
    )
    num_shards: int = 1

    def combine(self, other: OptState) -> OptState:
        return OptState(
            self.layer,
            self.layers,
            self.step,
            self.shards | other.shards,
            self.num_shards,
        )

    def memory(self) -> float:
        return HIDDEN_SIZE * HIDDEN_SIZE * (len(self.shards) / self.num_shards)


@dataclass
class ActivationGrad:
    """
    The gradient of the activations for a specific layer.

    May be split into different samples.
    """

    layer: int
    layers: int
    samples: FrozenSet[int]
    total_samples: int

    def memory(self) -> int:
        return len(self.samples) * HIDDEN_SIZE * BLOCK_SIZE


@dataclass
class Event:
    "Internal representations of events in the model for the visualizer"
    typ: str
    layer: Optional[int]
    rank: int
    time: int
    duration: int
    memory: int
    samples: FrozenSet[int] = frozenset()

    def __str__(self) -> str:
        string = f"({self.typ}"

        if self.samples:
            string += f", S={list(self.samples)}"

        string += ")"

        return string


class Model:
    def __init__(
        self,
        rank: int = 1,
        dist: Dist = Dist(1),
        num_layers: int = 2,
        global_batch_size: int = 1,
    ):
        self.rank = rank
        self.log: List[Event] = []
        self.dist = dist
        self.time = 0
        self.world_size = dist.world_size
        self.num_layers = num_layers
        self.global_batch_size = global_batch_size
        self.final_weights: Dict[int, Weight] = {}

        self.weights: Dict[Any, Weight] = {}
        self.opt_states: Dict[Any, OptState] = {}
        self.activations: Dict[Any, Activation] = {}
        self.grad_activations: Dict[Any, ActivationGrad] = {}
        self.grad_weights: Dict[Any, WeightGrad] = {}

        self.runtime_predictor = RuntimePredictor(dist.world_size)

    def storage(
        self,
    ) -> Tuple[
        Dict[Any, Weight],
        Dict[Any, OptState],
        Dict[Any, Activation],
        Dict[Any, ActivationGrad],
        Dict[Any, WeightGrad],
    ]:
        return (
            self.weights,
            self.opt_states,
            self.activations,
            self.grad_activations,
            self.grad_weights,
        )

    def memory(self) -> int:
        mem = 0
        for d in list(self.storage()):
            assert isinstance(d, dict)
            for v in d.values():
                mem += v.memory()
        return mem

    def status(self):
        for d in list(self.storage()):
            for k, v in d.items():
                print(k, type(v), end=",")
        print()

    def log_event(
        self,
        typ: str,
        layer: Optional[int] = None,
        samples: FrozenSet[int] = frozenset({}),
        input: Optional[Any] = None,
        num_shards_on_device: Optional[int] = None,
        num_shards: Optional[int] = None,
    ) -> None:
        duration = 0
        if typ in ["loss"]:
            duration = 0
        if typ in ["forward", "backward"]:
            duration = self.runtime_predictor.get_forward_backward_time(len(samples))
        if typ in ["update"]:
            duration = self.runtime_predictor.get_update_step_time(
                num_shards_on_device, num_shards
            )
        if typ in ["all_reduce", "reduce_scatter", "all_gather"]:
            duration = self.runtime_predictor.get_collective_time(typ, input)
        if typ in ["send", "recv"]:
            duration = self.runtime_predictor.get_send_recv_time(input)
        if typ in ["load_weights"]:
            duration = self.runtime_predictor.get_weight_load_time(
                num_shards_on_device, num_shards
            )

        self.log.append(
            Event(typ, layer, self.rank, self.time, duration, self.memory(), samples)
        )
        self.time += duration

    def load_weights(
        self, layer: int, shard: int = 0, num_shards: int = 1
    ) -> Tuple[Weight, OptState]:
        weight = Weight(layer, self.num_layers, 0, frozenset([shard]), num_shards)
        opt_state = OptState(layer, self.num_layers, 0, frozenset([shard]), num_shards)
        self.log_event(
            "load_weights",
            layer,
            input=weight,
            num_shards_on_device=1,
            num_shards=num_shards,
        )
        return weight, opt_state

    def set_final_weight(self, layer: int, weight: Weight) -> None:
        self.final_weights[layer] = weight

    def get_input_activation(self, samples: Sequence[int]) -> Activation:
        activation = Activation(
            0, self.num_layers, frozenset(samples), self.global_batch_size
        )
        return activation

    def forward(self, layer: int, inp: Activation, weight: Weight) -> Activation:
        self.log_event("forward", layer, inp.samples)
        assert weight.is_complete()
        assert (
            weight.layer == layer
        ), f"Weight should be layer {layer}, but is {weight.layer}"
        assert inp in self.activations.values()
        assert weight in self.weights.values()
        assert inp.layer == layer, f"Input should be layer {layer}, but is {inp.layer}"
        return Activation(
            layer + 1, self.num_layers, inp.samples, self.global_batch_size
        )

    def backward(
        self, layer: int, inp: Activation, grad: ActivationGrad, weight: Weight
    ) -> Tuple[WeightGrad, ActivationGrad]:
        self.log_event("backward", layer, inp.samples)
        assert weight.is_complete()
        assert weight.layer == layer, f"Weight should be layer {layer}"
        assert inp.layer == layer, f"Input should be layer {layer}"
        assert inp in self.activations.values()
        assert weight in self.weights.values()
        assert grad in self.grad_activations.values()
        assert set(inp.samples) == set(
            grad.samples
        ), f"Batch mismatch {set(inp.samples)}"
        assert grad.layer == layer, f"Activation Grad should be layer {layer}"
        return (
            WeightGrad(layer, self.num_layers, inp.samples, self.global_batch_size),
            ActivationGrad(
                layer - 1, self.num_layers, inp.samples, self.global_batch_size
            ),
        )

    def loss(self, inp: Activation) -> ActivationGrad:
        self.log_event("loss", self.num_layers, inp.samples)
        assert inp in self.activations.values()
        assert (
            inp.layer == self.num_layers
        ), f"Input should be final layer {self.num_layers}"
        return ActivationGrad(
            self.num_layers - 1, self.num_layers, inp.samples, self.global_batch_size
        )

    def update(
        self,
        layer: int,
        weight_grad: WeightGrad,
        weight: Weight,
        opt_state: OptState,
    ) -> Tuple[Weight, OptState]:
        assert weight in self.weights.values()
        assert weight_grad in self.grad_weights.values()
        assert opt_state in self.opt_states.values()
        assert weight.layer == layer, f"Weight should be layer {layer}"
        assert weight_grad.layer == layer, f"Grad weight should be layer {layer}"
        assert set(weight_grad.samples) == set(
            range(self.global_batch_size)
        ), f"{set(weight_grad.samples)}"
        assert opt_state.layer == layer
        if weight_grad.num_shards > 1:
            assert weight.shards.issubset(weight_grad.shards), f"Weight {weight.shards}"
            assert opt_state.shards.issubset(
                weight_grad.shards
            ), f"Opt {opt_state.shards}"
        assert weight.step == opt_state.step
        new_opt = OptState(
            layer,
            self.num_layers,
            opt_state.step + 1,
            opt_state.shards,
            opt_state.num_shards,
        )
        new_weight = Weight(
            layer, self.num_layers, weight.step + 1, weight.shards, weight.num_shards
        )
        self.log_event(
            "update",
            layer,
            weight_grad.samples,
            num_shards_on_device=len(weight.shards),
            num_shards=weight.num_shards,
        )
        return new_weight, new_opt

    async def all_reduce(self, v: T, layer: int) -> T:
        v, self.time = await self.dist.all_reduce(self.rank, v, self.time)
        self.log_event("all_reduce", layer, getattr(v, "samples", frozenset()), v)

        return v

    async def reduce_scatter(self, v: TO, layer: int) -> TO:
        v, self.time = await self.dist.reduce_scatter(self.rank, v, self.time)
        self.log_event("reduce_scatter", layer, getattr(v, "samples", frozenset()), v)
        return v

    async def all_gather(self, v: O, layer: int) -> O:
        v, self.time = await self.dist.all_gather(self.rank, v, self.time)
        self.log_event("all_gather", layer, getattr(v, "samples", frozenset()), v)
        return v

    async def send(self, dest: int, v: Any) -> None:
        await self.dist.send(self.rank, dest, (v, self.time))
        self.log_event("send", v.layer, getattr(v, "samples", frozenset()), v)

    async def receive(self, source: int) -> Any:
        v, time = await self.dist.receive(source, self.rank)
        self.time = max(time, self.time)
        self.log_event("recv", v.layer, getattr(v, "samples", frozenset()), v)
        return v


class RuntimePredictor:
    def __init__(
        self, world_size: int, config_file_name: str = "h100_80g_nvl8.json"
    ) -> None:
        self.world_size = world_size

        config_file_path = (
            Path(__file__).parent
            / "third_party"
            / "calculon"
            / "systems"
            / config_file_name
        )
        cfg = json.load(open(config_file_path))

        self.processor = Processor(cfg["matrix"])
        self.mem_hbm = Memory(cfg["mem1"])
        self.mem_pcie = Memory(cfg["mem2"])
        networks = [Network(n) for n in cfg["networks"]]

        # find the network with minimum size that is greater than or equal to world_size
        networks = sorted(networks, key=lambda x: x.size)
        for network in networks:
            if network.size >= world_size:
                self.network = network
                break

    def compute_flops_time(self, flops: int) -> float:
        return flops / self.processor.throughput(DATATYPE, flops)

    def compute_mem_time(self, bytes_read_write: int) -> float:
        return bytes_read_write / self.mem_hbm.throughput(bytes_read_write)

    def get_processing_time(self, flops: int, bytes_read_write: int) -> float:
        return max(
            self.compute_flops_time(flops), self.compute_mem_time(bytes_read_write)
        )

    def get_forward_backward_time(self, batch_size: int) -> float:
        # we are assuming there is no sharding of model state
        # [b * h] @ [h * h] = [b * h]
        input_size = batch_size * BLOCK_SIZE

        flops = 2 * HIDDEN_SIZE * HIDDEN_SIZE * input_size
        bytes_read_write = (
            HIDDEN_SIZE * HIDDEN_SIZE
            + input_size * HIDDEN_SIZE
            + input_size * HIDDEN_SIZE
        )
        bytes_read_write *= BYTES_PER_ELEMENT

        return self.get_processing_time(flops, bytes_read_write)

    def get_update_step_time(self, num_shards_on_device: int, num_shards: int) -> float:
        # [h * h/n] - [h * h/n] = [h * h/n]
        h1 = HIDDEN_SIZE
        h2 = HIDDEN_SIZE * num_shards_on_device // num_shards
        flops = h1 * h2
        bytes_read_write = 3 * h1 * h2
        bytes_read_write *= BYTES_PER_ELEMENT

        return self.get_processing_time(flops, bytes_read_write)

    def get_comm_bytes(self, v: T) -> int:
        if type(v) == Weight:
            comm_bytes = HIDDEN_SIZE * HIDDEN_SIZE
        elif type(v) == WeightGrad:
            comm_bytes = HIDDEN_SIZE * HIDDEN_SIZE
        elif type(v) == Activation:
            comm_bytes = HIDDEN_SIZE * BLOCK_SIZE * len(v.samples)
        elif type(v) == ActivationGrad:
            comm_bytes = HIDDEN_SIZE * BLOCK_SIZE * len(v.samples)
        elif type(v) == OptState:
            comm_bytes = HIDDEN_SIZE * HIDDEN_SIZE // v.num_shards
        else:
            raise ValueError(f"Unknown type {type(v)}")

        comm_bytes *= BYTES_PER_ELEMENT
        return comm_bytes

    def get_collective_time(self, collective_name: str, v: T) -> float:
        comm_size = self.get_comm_bytes(v)
        return self.network.time(collective_name, comm_size, self.world_size)

    def get_send_recv_time(self, v: T) -> float:
        return self.network.time("p2p", self.get_comm_bytes(v), 2)

    def get_weight_load_time(self, num_shards_on_device: int, num_shards: int) -> float:
        bytes_read_write = (
            HIDDEN_SIZE * HIDDEN_SIZE * num_shards_on_device // num_shards
        )
        bytes_read_write *= BYTES_PER_ELEMENT

        return bytes_read_write / self.mem_pcie.throughput(bytes_read_write)
