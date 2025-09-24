# Comprehensive lib.py Documentation

## Table of Contents
1. [Global Constants & Imports](#global-constants--imports)
2. [Protocol Interfaces](#protocol-interfaces)
3. [Utility Classes](#utility-classes)
4. [Data Structure Classes](#data-structure-classes)
5. [Dist Class (Distributed Communication)](#dist-class-distributed-communication)
6. [Model Class (Core Training Interface)](#model-class-core-training-interface)
7. [RuntimePredictor Class (Performance Simulation)](#runtimepredictor-class-performance-simulation)

---

## Global Constants & Imports

### Constants
```python
HIDDEN_SIZE = 4096
BLOCK_SIZE = 1024
DATATYPE = "float16"
BYTES_PER_ELEMENT = 2
```

**Purpose**: These constants define the neural network architecture parameters and data representation used throughout the simulation.

**Details**:
- `HIDDEN_SIZE`: The dimensionality of the hidden layers in the transformer model (4096 = typical large model size)
- `BLOCK_SIZE`: The sequence length or context window size (1024 tokens)
- `DATATYPE`: The numerical precision used for computations ("float16" for memory efficiency)
- `BYTES_PER_ELEMENT`: Memory footprint per element (2 bytes for float16)

**Why These Values**: These represent realistic values for large language models, allowing the simulation to provide meaningful performance estimates.

---

## Protocol Interfaces

### `Reduceable[T]` Protocol

```python
class Reduceable(Protocol[T]):
    def __add__(self, other: T) -> T: ...
```

**Purpose**: Defines the interface for types that can be mathematically reduced (summed) across multiple instances.

**High-Level**: Enables gradient accumulation and parameter averaging in distributed training scenarios.

**Implementation**: Types implementing this protocol must define how two instances combine via addition (typically element-wise sum for tensors).

**Used By**: `WeightGrad` class for gradient accumulation across different samples or GPUs.

### `Gatherable[O]` Protocol

```python
class Gatherable(Protocol[O]):
    def shard(self, shard: int, num_shards: int) -> O: ...
    def is_complete(self) -> bool: ...
    def combine(self, other: O) -> O: ...
```

**Purpose**: Defines the interface for types that can be partitioned (sharded) and reconstructed across distributed systems.

**High-Level**: Enables memory-efficient distribution of large tensors across multiple GPUs.

**Methods**:
- `shard()`: Splits the object into a specific partition
- `is_complete()`: Checks if all shards are present
- `combine()`: Merges multiple sharded instances

**Used By**: `Weight`, `WeightGrad`, and `OptState` classes for parameter sharding in ZeRO optimization.

### `ReduceableGatherable[TO]` Protocol

```python
class ReduceableGatherable(Reduceable[TO], Gatherable[TO]):
    pass
```

**Purpose**: Combines both interfaces for types that need both reduction and gathering capabilities.

**Used By**: `WeightGrad` class, which needs to be both accumulated (reduced) and sharded (gathered).

---

## Utility Classes

### `Barrier` Class

```python
class Barrier:
    def __init__(self, target: int)
    async def wait(self, rank: int) -> None
    async def reset(self) -> None
```

**Purpose**: Implements distributed synchronization primitive to coordinate multiple asynchronous GPU processes.

**High-Level**: Ensures all GPUs reach the same point in execution before any can proceed (critical for collective operations).

#### `__init__(self, target: int)`
- **Purpose**: Initialize barrier for `target` number of participants
- **Parameters**: `target` - Number of processes that must reach the barrier
- **Implementation**: Sets up counters and locks for coordination

#### `async wait(self, rank: int) -> None`
- **Purpose**: Block until all processes reach this point
- **Parameters**: `rank` - Unique identifier for calling process
- **Implementation**: Atomically increments counter, spins until all arrive, handles cleanup
- **Why Async**: Allows other coroutines to run while waiting
- **Relation to Other Methods**: Used by all collective operations in `Dist` class

#### `async reset(self) -> None`
- **Purpose**: Prepare barrier for reuse after all processes have passed
- **Implementation**: Waits for all processes to finish, then resets counters
- **Called By**: Rank 0 process after barrier completion

---

## Data Structure Classes

### `Event` Class

```python
@dataclass
class Event:
    typ: str
    layer: Optional[int]
    rank: int
    time: int
    duration: int
    memory: int
    samples: FrozenSet[int] = frozenset()
```

**Purpose**: Records timestamped events for performance analysis and visualization.

**High-Level**: Enables detailed profiling and debugging of distributed training execution.

**Fields**:
- `typ`: Event type (e.g., "forward", "backward", "all_reduce")
- `layer`: Which model layer (if applicable)
- `rank`: Which GPU/process
- `time`: When event started
- `duration`: How long it took
- `memory`: Memory usage at this point
- `samples`: Which batch samples were involved

**Used By**: `Model.log_event()` method for creating execution traces.

### `Weight` Class

```python
@dataclass
class Weight(Gatherable["Weight"]):
    layer: int
    layers: int
    step: int
    shards: FrozenSet[int] = frozenset([0])
    num_shards: int = 1
```

**Purpose**: Represents neural network parameters for a specific layer, with support for distributed sharding.

**High-Level**: Core data structure for model parameters that can be split across GPUs for memory efficiency.

#### Field Details:
- `layer`: Which layer these weights belong to (0 to num_layers-1)
- `layers`: Total number of layers in the model
- `step`: Training step/iteration (for tracking parameter updates)
- `shards`: Set of shard indices this instance contains
- `num_shards`: Total number of shards the full weight is split into

#### `combine(self, other: Weight) -> Weight`
- **Purpose**: Merge two sharded weight instances into a larger shard collection
- **Implementation**: Union the shard sets, keeping other fields identical
- **Used For**: Reconstructing full parameters from distributed shards
- **Assertion**: Both weights must be from same layer and step

#### `memory(self) -> float`
- **Purpose**: Calculate memory footprint of this weight instance
- **Implementation**: `(len(self.shards) / self.num_shards) * HIDDEN_SIZE * HIDDEN_SIZE`
- **Why**: Memory proportional to fraction of shards held × parameter matrix size
- **Used By**: `Model.memory()` for total memory accounting

#### `shard(self, shard: int, num_shards: int) -> Weight`
- **Purpose**: Extract a specific shard from complete weights
- **Parameters**: `shard` - which shard to extract, `num_shards` - total shard count
- **Preconditions**: Must be complete (`is_complete()` returns True)
- **Implementation**: Create new Weight with only the specified shard
- **Used For**: ZeRO optimization when distributing parameters across GPUs

#### `is_complete(self) -> bool`
- **Purpose**: Check if this instance contains all shards (full parameters)
- **Implementation**: `len(self.shards) == self.num_shards`
- **Used For**: Validation before operations requiring complete parameters

### `Activation` Class

```python
@dataclass
class Activation:
    layer: int
    layers: int
    samples: FrozenSet[int]
    total_samples: int
```

**Purpose**: Represents forward pass activations for specific batch samples at a given layer.

**High-Level**: Intermediate computations passed between layers during forward propagation.

#### Field Details:
- `layer`: Which layer produced these activations
- `layers`: Total layers in model (for validation)
- `samples`: Set of batch sample indices these activations represent
- `total_samples`: Total batch size (for validation)

#### `memory(self) -> int`
- **Purpose**: Calculate memory footprint of activations
- **Implementation**: `len(self.samples) * HIDDEN_SIZE * BLOCK_SIZE`
- **Why**: Memory scales with number of samples × hidden dimension × sequence length
- **Used By**: Memory tracking and optimization decisions

### `WeightGrad` Class

```python
@dataclass
class WeightGrad(Reduceable["WeightGrad"], Gatherable["WeightGrad"]):
    layer: int
    layers: int
    samples: FrozenSet[int]
    total_samples: int
    shards: FrozenSet[int] = frozenset([0])
    num_shards: int = 1
```

**Purpose**: Represents gradients for model parameters, supporting both accumulation and sharding.

**High-Level**: Core data structure for backpropagation that enables distributed gradient computation.

#### `__add__(self, other: WeightGrad) -> WeightGrad`
- **Purpose**: Accumulate gradients from multiple batch samples
- **Implementation**: Union sample sets (gradients sum across samples)
- **Assertions**: Same layer and same shards (can only add compatible gradients)
- **Used For**: Gradient accumulation in distributed data parallel training

#### `combine(self, other: WeightGrad) -> WeightGrad`
- **Purpose**: Merge sharded gradients from different GPUs
- **Implementation**: Union shard sets while keeping samples unchanged
- **Used For**: Reconstructing full gradients from distributed computation

#### `shard(self, shard: int, num_shards: int) -> WeightGrad`
- **Purpose**: Extract specific parameter shard's gradients
- **Preconditions**: Must be complete gradients
- **Used For**: ZeRO-2 optimization (gradient sharding)

#### `is_complete(self) -> bool`
- **Purpose**: Check if gradients cover all parameter shards
- **Used For**: Validation before parameter updates

#### `memory(self) -> float`
- **Purpose**: Calculate memory footprint proportional to shards held
- **Implementation**: Same as `Weight.memory()` (gradients same size as parameters)

### `OptState` Class

```python
@dataclass
class OptState(Gatherable["OptState"]):
    layer: int
    layers: int
    step: int
    shards: FrozenSet[int] = frozenset([0])
    num_shards: int = 1
```

**Purpose**: Represents optimizer state (e.g., ADAM momentum) for a layer, with sharding support.

**High-Level**: Persistent state needed by optimization algorithms, can be distributed to save memory.

#### `combine(self, other: OptState) -> OptState`
- **Purpose**: Merge optimizer state shards from different GPUs
- **Implementation**: Union shard sets
- **Used For**: Reconstructing full optimizer state when needed

#### `memory(self) -> float`
- **Purpose**: Calculate memory footprint
- **Implementation**: `HIDDEN_SIZE * HIDDEN_SIZE * (len(self.shards) / self.num_shards)`
- **Why**: Optimizer state typically same size as parameters (for ADAM: momentum + variance)

### `ActivationGrad` Class

```python
@dataclass
class ActivationGrad:
    layer: int
    layers: int
    samples: FrozenSet[int]
    total_samples: int
```

**Purpose**: Represents gradients flowing backward through activations during backpropagation.

**High-Level**: Intermediate gradients passed between layers during backward pass.

#### `memory(self) -> int`
- **Purpose**: Calculate memory footprint
- **Implementation**: `len(self.samples) * HIDDEN_SIZE * BLOCK_SIZE`
- **Why**: Same size as forward activations (gradient has same shape)

---

## Dist Class (Distributed Communication)

```python
class Dist:
    def __init__(self, world_size: int) -> None
    async def all_reduce(self, rank: int, inp: T, time: int) -> Tuple[T, int]
    async def all_gather(self, rank: int, inp: O, time: int) -> Tuple[O, int]
    async def reduce_scatter(self, rank: int, inp: TO, time: int) -> Tuple[TO, int]
    async def receive(self, rank_source: int, rank_dest: int) -> Any
    async def send(self, rank_source: int, rank_dest: int, v: Any) -> None
```

**Purpose**: Simulates distributed communication patterns used in multi-GPU training.

**High-Level**: Provides the fundamental building blocks for all distributed training strategies.

### `__init__(self, world_size: int) -> None`
- **Purpose**: Initialize distributed communication system
- **Parameters**: `world_size` - Number of participating GPUs/processes
- **Implementation**: Sets up barriers and communication queues between all pairs
- **Fields Created**:
  - `reduce`: Temporary storage for all-reduce operations
  - `gather`: Temporary storage for all-gather operations  
  - `barrier`: Synchronization primitive
  - `queue`: Point-to-point communication channels

### `async all_reduce(self, rank: int, inp: T, time: int) -> Tuple[T, int]`
- **Purpose**: Sum identical data across all GPUs (like gradient averaging)
- **Parameters**: 
  - `rank`: Calling GPU's identifier
  - `inp`: Data to reduce (must implement `__add__`)
  - `time`: Current logical time
- **Returns**: Tuple of (reduced_result, final_time)
- **Implementation**:
  1. Accumulate data from all ranks using `+` operator
  2. Synchronize using barrier (3 barriers for proper coordination)
  3. All ranks get identical result
  4. Rank 0 cleans up shared state
- **Used For**: Gradient synchronization in data parallel training
- **Why Async**: Simulates real network communication delays

### `async all_gather(self, rank: int, inp: O, time: int) -> Tuple[O, int]`
- **Purpose**: Collect and combine sharded data from all GPUs
- **Parameters**: `inp` must implement `Gatherable` protocol
- **Implementation**:
  1. Combine sharded data using `.combine()` method
  2. Synchronize via barriers
  3. All ranks receive complete reconstructed data
- **Used For**: Reconstructing full parameters from ZeRO-sharded weights
- **Difference from all_reduce**: Gathers different shards vs summing identical data

### `async reduce_scatter(self, rank: int, inp: TO, time: int) -> Tuple[TO, int]`
- **Purpose**: Combine data from all ranks, then distribute different shards to each rank
- **Implementation**: Combines `all_reduce` followed by `.shard()` operation
- **Returns**: Each rank gets a different shard of the reduced result
- **Used For**: Efficiently distributing reduced gradients in ZeRO optimization
- **Why Efficient**: Saves memory vs all-reduce + manual sharding

### `async send(self, rank_source: int, rank_dest: int, v: Any) -> None`
- **Purpose**: Send data from one GPU to another
- **Implementation**: Put data in appropriate queue
- **Used For**: Pipeline parallelism (passing activations between pipeline stages)

### `async receive(self, rank_source: int, rank_dest: int) -> Any`
- **Purpose**: Receive data from another GPU
- **Implementation**: Get data from appropriate queue (blocks if empty)
- **Used For**: Pipeline parallelism (receiving activations from previous stage)

---

## Model Class (Core Training Interface)

```python
class Model:
    def __init__(self, rank: int = 1, dist: Dist = Dist(1), num_layers: int = 2, global_batch_size: int = 1)
    def storage(self) -> Tuple[...]
    def memory(self) -> int
    def status(self)
    def log_event(self, typ: str, ...) -> None
    def load_weights(self, layer: int, shard: int = 0, num_shards: int = 1) -> Tuple[Weight, OptState]
    def set_final_weight(self, layer: int, weight: Weight) -> None
    def get_input_activation(self, samples: Sequence[int]) -> Activation
    def forward(self, layer: int, inp: Activation, weight: Weight) -> Activation
    def backward(self, layer: int, inp: Activation, grad: ActivationGrad, weight: Weight) -> Tuple[WeightGrad, ActivationGrad]
    def loss(self, inp: Activation) -> ActivationGrad
    def update(self, layer: int, weight_grad: WeightGrad, weight: Weight, opt_state: OptState) -> Tuple[Weight, OptState]
    async def all_reduce(self, v: T, layer: int) -> T
    async def reduce_scatter(self, v: TO, layer: int) -> TO
    async def all_gather(self, v: O, layer: int) -> O
    async def send(self, dest: int, v: Any) -> None
    async def receive(self, source: int) -> Any
```

**Purpose**: Main interface for neural network training simulation, combining computation and communication.

**High-Level**: Represents a single GPU's view of the distributed training process.

### `__init__(self, rank: int = 1, dist: Dist = Dist(1), num_layers: int = 2, global_batch_size: int = 1)`
- **Purpose**: Initialize a GPU instance for distributed training
- **Parameters**:
  - `rank`: This GPU's unique identifier (0 to world_size-1)
  - `dist`: Distributed communication system
  - `num_layers`: Total layers in the neural network
  - `global_batch_size`: Total samples across all GPUs
- **Fields Initialized**:
  - Storage dictionaries for all data types
  - Performance tracking (`log`, `time`)
  - Runtime predictor for timing simulation

### `storage(self) -> Tuple[Dict[Any, Weight], Dict[Any, OptState], Dict[Any, Activation], Dict[Any, ActivationGrad], Dict[Any, WeightGrad]]`
- **Purpose**: Provide access to all storage dictionaries
- **Returns**: Tuple of all five storage containers
- **Used For**: Training loops to access and modify model state
- **Why Tuple**: Convenient unpacking in training functions

### `memory(self) -> int`
- **Purpose**: Calculate total memory usage across all stored data
- **Implementation**: Sums `.memory()` calls across all stored objects
- **Used For**: Memory optimization and constraint checking
- **Returns**: Total bytes used

### `status(self)`
- **Purpose**: Debug method to print current storage contents
- **Implementation**: Iterates through all storage dictionaries
- **Used For**: Debugging storage state during development

### `log_event(self, typ: str, layer: Optional[int] = None, samples: FrozenSet[int] = frozenset({}), input: Optional[Any] = None, num_shards_on_device: Optional[int] = None, num_shards: Optional[int] = None) -> None`
- **Purpose**: Record timestamped events for performance analysis
- **Parameters**:
  - `typ`: Event type (determines duration calculation)
  - `layer`: Which layer (if applicable)
  - `samples`: Which batch samples involved
  - `input`: Data object (for communication cost calculation)
  - Shard info for memory operations
- **Implementation**:
  1. Calculate duration based on event type using `RuntimePredictor`
  2. Create `Event` object with current state
  3. Append to log and advance time
- **Event Types & Duration Calculation**:
  - "forward"/"backward": Based on batch size
  - "update": Based on shard counts
  - Communication ops: Based on data size
  - "load_weights": Based on shard loading
- **Used By**: All computation and communication methods

### `load_weights(self, layer: int, shard: int = 0, num_shards: int = 1) -> Tuple[Weight, OptState]`
- **Purpose**: Initialize model parameters and optimizer state for a layer
- **Parameters**:
  - `layer`: Which layer to load
  - `shard`: Which shard of the parameters (for ZeRO)
  - `num_shards`: Total shards the layer is split into
- **Returns**: Tuple of (Weight, OptState) both with step=0
- **Implementation**: Creates new Weight and OptState objects with specified sharding
- **Side Effects**: Logs "load_weights" event with timing
- **Used At**: Training start to initialize all required parameters

### `set_final_weight(self, layer: int, weight: Weight) -> None`
- **Purpose**: Store final trained weights after training completion
- **Parameters**: 
  - `layer`: Which layer
  - `weight`: Final weight state
- **Used For**: Validation that training completed correctly
- **Stored In**: `self.final_weights` dictionary

### `get_input_activation(self, samples: Sequence[int]) -> Activation`
- **Purpose**: Create input activations for the first layer
- **Parameters**: `samples` - batch sample indices to process
- **Returns**: Activation object for layer 0
- **Implementation**: Creates Activation with layer=0, specified samples
- **Used At**: Start of forward pass for each microbatch

### `forward(self, layer: int, inp: Activation, weight: Weight) -> Activation`
- **Purpose**: Simulate forward pass computation for one layer
- **Parameters**:
  - `layer`: Which layer to compute
  - `inp`: Input activations from previous layer
  - `weight`: Layer parameters
- **Returns**: Output activations for next layer
- **Assertions**:
  - Weight must be complete (all shards present)
  - Weight and input must be for correct layer
  - Weight must exist in storage
- **Implementation**: Creates new Activation for layer+1 with same samples
- **Side Effects**: Logs "forward" event with timing
- **Why Simulation**: Real computation replaced with timing model

### `backward(self, layer: int, inp: Activation, grad: ActivationGrad, weight: Weight) -> Tuple[WeightGrad, ActivationGrad]`
- **Purpose**: Simulate backward pass computation for one layer
- **Parameters**:
  - `layer`: Which layer to compute gradients for
  - `inp`: Forward activations (saved from forward pass)
  - `grad`: Gradient flowing from next layer
  - `weight`: Layer parameters
- **Returns**: Tuple of (weight gradients, activation gradients for previous layer)
- **Assertions**:
  - All inputs must be for correct layer
  - Samples must match between activations and gradients
  - Objects must exist in storage
- **Implementation**: Creates WeightGrad and ActivationGrad for layer-1
- **Side Effects**: Logs "backward" event
- **Used For**: Computing gradients during backpropagation

### `loss(self, inp: Activation) -> ActivationGrad`
- **Purpose**: Compute loss gradients from final layer activations
- **Parameters**: `inp` - activations from final layer
- **Returns**: ActivationGrad to start backpropagation
- **Assertions**: Input must be from final layer (layer == num_layers)
- **Implementation**: Creates ActivationGrad for layer num_layers-1
- **Side Effects**: Logs "loss" event
- **Used At**: Transition from forward to backward pass

### `update(self, layer: int, weight_grad: WeightGrad, weight: Weight, opt_state: OptState) -> Tuple[Weight, OptState]`
- **Purpose**: Update model parameters using computed gradients
- **Parameters**:
  - `layer`: Which layer to update
  - `weight_grad`: Accumulated gradients
  - `weight`: Current parameters
  - `opt_state`: Current optimizer state
- **Returns**: Tuple of (updated_weight, updated_opt_state)
- **Assertions**:
  - All objects must be for correct layer
  - Gradients must cover all samples
  - Shard compatibility checks
  - Step synchronization between weight and optimizer
- **Implementation**: Creates new Weight and OptState with incremented step
- **Side Effects**: Logs "update" event with shard-dependent timing
- **Used At**: End of training iteration to apply computed gradients

### `async all_reduce(self, v: T, layer: int) -> T`
- **Purpose**: Wrapper around Dist.all_reduce with logging
- **Parameters**: 
  - `v`: Data to reduce
  - `layer`: For logging purposes
- **Implementation**: Delegates to `self.dist.all_reduce`, logs event
- **Used For**: Gradient synchronization in data parallel training

### `async reduce_scatter(self, v: TO, layer: int) -> TO`
- **Purpose**: Wrapper around Dist.reduce_scatter with logging
- **Implementation**: Delegates to `self.dist.reduce_scatter`, logs event
- **Used For**: ZeRO gradient sharding

### `async all_gather(self, v: O, layer: int) -> O`
- **Purpose**: Wrapper around Dist.all_gather with logging
- **Implementation**: Delegates to `self.dist.all_gather`, logs event
- **Used For**: Reconstructing full parameters in ZeRO

### `async send(self, dest: int, v: Any) -> None`
- **Purpose**: Send data to another GPU with logging
- **Parameters**:
  - `dest`: Destination GPU rank
  - `v`: Data to send
- **Implementation**: 
  1. Send (data, current_time) tuple via Dist
  2. Log "send" event
- **Used For**: Pipeline parallelism

### `async receive(self, source: int) -> Any`
- **Purpose**: Receive data from another GPU with time synchronization
- **Parameters**: `source` - Source GPU rank
- **Returns**: Received data
- **Implementation**:
  1. Receive (data, sender_time) tuple
  2. Update local time to max(local_time, sender_time)
  3. Log "recv" event
- **Why Time Sync**: Maintains causal ordering across GPUs
- **Used For**: Pipeline parallelism

---

## RuntimePredictor Class (Performance Simulation)

```python
class RuntimePredictor:
    def __init__(self, world_size: int, config_file_name: str = "h100_80g_nvl8.json") -> None
    def compute_flops_time(self, flops: int) -> float
    def compute_mem_time(self, bytes_read_write: int) -> float
    def get_processing_time(self, flops: int, bytes_read_write: int) -> float
    def get_forward_backward_time(self, batch_size: int) -> float
    def get_update_step_time(self, num_shards_on_device: int, num_shards: int) -> float
    def get_comm_bytes(self, v: T) -> int
    def get_collective_time(self, collective_name: str, v: T) -> float
    def get_send_recv_time(self, v: T) -> float
    def get_weight_load_time(self, num_shards_on_device: int, num_shards: int) -> float
```

**Purpose**: Provides realistic timing estimates for computation and communication operations.

**High-Level**: Bridges the gap between simulation and reality by modeling actual hardware performance.

### `__init__(self, world_size: int, config_file_name: str = "h100_80g_nvl8.json") -> None`
- **Purpose**: Initialize performance models based on hardware configuration
- **Parameters**:
  - `world_size`: Number of GPUs (affects network topology)
  - `config_file_name`: Hardware specification file
- **Implementation**:
  1. Load hardware config (compute, memory, network specs)
  2. Create processor, memory, and network models
  3. Select appropriate network topology for world_size
- **Hardware Components**:
  - `processor`: Compute throughput model
  - `mem_hbm`: High-bandwidth memory model
  - `mem_pcie`: PCIe memory model
  - `network`: Inter-GPU communication model

### `compute_flops_time(self, flops: int) -> float`
- **Purpose**: Calculate time for pure computation
- **Parameters**: `flops` - Number of floating-point operations
- **Implementation**: `flops / self.processor.throughput(DATATYPE, flops)`
- **Returns**: Time in seconds
- **Used By**: Forward/backward pass timing

### `compute_mem_time(self, bytes_read_write: int) -> float`
- **Purpose**: Calculate time for memory operations
- **Parameters**: `bytes_read_write` - Total memory traffic
- **Implementation**: `bytes_read_write / self.mem_hbm.throughput(bytes_read_write)`
- **Returns**: Time in seconds
- **Used By**: Memory-bound operation timing

### `get_processing_time(self, flops: int, bytes_read_write: int) -> float`
- **Purpose**: Calculate time for operations limited by either compute or memory
- **Implementation**: `max(compute_flops_time(flops), compute_mem_time(bytes_read_write))`
- **Why Max**: Operation is limited by the slower of compute vs memory
- **Used By**: All computation timing methods

### `get_forward_backward_time(self, batch_size: int) -> float`
- **Purpose**: Calculate time for forward or backward pass through one layer
- **Parameters**: `batch_size` - Number of samples being processed
- **Implementation**:
  1. Calculate input size: `batch_size * BLOCK_SIZE`
  2. Compute FLOPs: `2 * HIDDEN_SIZE * HIDDEN_SIZE * input_size` (matrix multiply)
  3. Calculate memory traffic: weights + inputs + outputs
  4. Return processing time for both constraints
- **Mathematical Model**: Based on transformer layer computation pattern
- **Used By**: Model.log_event for "forward"/"backward" events

### `get_update_step_time(self, num_shards_on_device: int, num_shards: int) -> float`
- **Purpose**: Calculate time for parameter update (optimizer step)
- **Parameters**: Shard distribution information
- **Implementation**:
  1. Calculate effective parameter size based on sharding
  2. Model optimizer operations (typically 3x memory traffic: grad, momentum, params)
  3. Return processing time
- **Used By**: Model.log_event for "update" events

### `get_comm_bytes(self, v: T) -> int`
- **Purpose**: Calculate communication size for different data types
- **Parameters**: `v` - Data object to communicate
- **Implementation**: Switch on type to calculate appropriate size:
  - `Weight`/`WeightGrad`: `HIDDEN_SIZE * HIDDEN_SIZE * BYTES_PER_ELEMENT`
  - `Activation`/`ActivationGrad`: `HIDDEN_SIZE * BLOCK_SIZE * len(samples) * BYTES_PER_ELEMENT`
  - `OptState`: Size based on sharding
- **Returns**: Total bytes to communicate
- **Used By**: Communication timing methods

### `get_collective_time(self, collective_name: str, v: T) -> float`
- **Purpose**: Calculate time for collective communication operations
- **Parameters**:
  - `collective_name`: "all_reduce", "all_gather", "reduce_scatter"
  - `v`: Data being communicated
- **Implementation**: 
  1. Calculate communication size using `get_comm_bytes`
  2. Use network model for specific collective pattern
- **Returns**: Communication time in seconds
- **Used By**: Communication event logging

### `get_send_recv_time(self, v: T) -> float`
- **Purpose**: Calculate time for point-to-point communication
- **Implementation**: Uses network model with "p2p" pattern and 2 participants
- **Used By**: Pipeline parallelism timing

### `get_weight_load_time(self, num_shards_on_device: int, num_shards: int) -> float`
- **Purpose**: Calculate time to load model parameters from storage
- **Implementation**: Based on PCIe memory bandwidth for parameter loading
- **Used By**: Model initialization timing

---

## Relationships and Interaction Patterns

### Training Loop Flow
1. **Initialization**: `Model.__init__` → `load_weights` for all layers
2. **Forward Pass**: `get_input_activation` → `forward` for each layer → `loss`
3. **Backward Pass**: `backward` for each layer (reverse order)
4. **Communication**: `all_reduce`/`all_gather`/`send`/`receive` as needed
5. **Parameter Update**: `update` for all layers → `set_final_weight`

### Memory Management
- All operations automatically tracked via `memory()` method
- Storage managed through dictionaries returned by `storage()`
- Memory cleanup responsibility of training loop (explicit `del` statements)

### Timing and Logging
- Every operation logs events via `log_event()`
- `RuntimePredictor` provides realistic timing estimates
- Time automatically advances with each logged operation
- Final performance metrics calculated from event logs

### Distributed Patterns
- **Data Parallel**: All GPUs have full model, use `all_reduce` for gradients
- **ZeRO**: Shard parameters/gradients/optimizer, use `all_gather`/`reduce_scatter`
- **Pipeline Parallel**: Split layers across GPUs, use `send`/`receive` for activations
- **Hybrid**: Combine multiple strategies for optimal performance