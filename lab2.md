# CS 8803 SMR Lab 2 : DL Model training puzzles

Welcome to Lab 2 for CS 8803 SMR. In this lab, we'll explore how to distributed model training works using the a simple simulation library. We will simulate one training iteration of a simple model using multiple GPUs.

## Environment Setup
1. Clone the repository
2. Create a new virtual environment using conda/venv.
    - `conda create -n lab2_env python=3.8`
    - or `python -m venv lab2_env`
3. Activate the virtual environment `conda activate lab2_env` or `source lab2_env/bin/activate`
4. Install the required packages using `pip install -r requirements.txt`
5. Install calculon by running `pip install .` in the `third_party/calculon` directory.

## Background

Training large-scale DL models is computationally intensive and often constrained by memory and compute resources. In this lab, we'll explore distributed model training using a simple simulation library. You will implement training strategies for a model with an arbitrary number of layers for a given batch size, simulating one training iteration.

## Task

In this lab, you need to implement the training loop for different parallelization strategies (simulated using the `Dist` class in `lib.py`) for training large DNNs. The training loop signature has been provided to you in 4 different files -- corresponding to each subpart.

A demo training loop is shown in `part0.py` file showing how to perform distributed data parallel training with gradient accumulation. Additionally, you can use the chrome trace file to visualize the training process in batches. This will allow you to visualize the training process in batches and across ranks.

1. ZeRO Optimization (`part1.py`, 2 points):
   - Implement the ZeRO (Zero Redundancy Optimizer) strategy. See [ZeRO paper](https://arxiv.org/abs/1910.02054) for details.
   - Partition the Optimizer states, Gradients and Weights across GPUs (ranks)
2. Pipeline Parallelism (`part2.py`, 3 points):
   - Split the layers and optimizer states equally between GPUs (ranks) [One GPU can have multiple layers as well, so keep that in mind]
   - Pass the full set of batches for activations and gradient activations between layers
   - To help understand the how different configuration parameters affect the performance, try playing with `num_layers` and `world_size` to see how MFU changes. 
3. 1F1B Scheduling (1 Forward 1 Backward Scheduling) (`part3.py`, 5 points):
   - Implement a default 1F1B scheduling strategy. See [1F1B paper](https://arxiv.org/pdf/2104.04473) for details.
   - You only need to implement the the `default` strategy.
4. Custom Strategy (`part4.py`, 5 points):
   - There is no single one-size fits-all solution to distributed model training. You are free to implement your own strategy or a combination of the above strategies or any other strategy (for instance, interleaved 1F1B schedule, 2D parallelism combining data and pipeline parallelism, etc.), or just optimize the configuration parameters of the strategies you have already implemented.
   - Optimize for Model FLOPS Utilization (MFU) -- higher the better.
   - A few example strategy directions to explore are mentioned in `part4.py`
   - You can test out your implementation using the `part4.py` file by running `python part4.py`. 
   - Explore the traces of your training using the generated chrome trace file. Look at performance bottlenecks and how you can optimize your training.

## Implementation Details
- Use the `all_reduce`, `all_gather`, `reduce_scatter`, `send` & `recv` methods to implement distributed training.
- Implement the a single training step for each strategy.
- You are free to import any class from `lib.py` and `utils.py`.
- You're training loops must support different numbers of ranks and batches. Do not hardcode the number of ranks or batches.
- Refer to `part0.py` for an example of how to write a sample training loop.
- Do not modify `lib.py` and `utils.py`.
- In part 4 the code might also be run for the following constraints : `world_size >= num_layers` and also `world_size < num_layers` . 

## Evaluation Criteria
The time and memory metrics will be used to create a leaderboard for this part of the lab. Students in the class will be ranked based on the time and memory metrics. The grading will be based on the following criteria:
- MFU [Model FLOPs Utilization] (higher is better)
- Memory usage (lower is better)
- Training time (lower is better)

The first 3 strategies will be evaluated on just the correctness and being within the memory and time constraints. The custom strategy will be evaluated on the MFU and it's rank on the leaderboard. 

The points for Part 4 will be based on the composite score of the following criteria and are subject to change maximum of 5 points:

`Part 4 Score = 0.75*MFU_Points + 0.25*Memory_Points`

### MFU Point Splits are as follows: 
- 5 (Full points) - For MFU >= 80%
- 4 (Partial points) - For MFU >= 70%
- 3 (Partial points) - For MFU >= 60%
- 2 (Partial points) - For MFU >= 50%
- 1 (Partial points) - For MFU >= 40%

### Memory Point Thresholds are as follows (Higher memory usage per rank):
- 5 (Full Points) For Memory <= 60_000_000_000
- 4 (Partial Points) For Memory <= 69_000_000_000
- 3 (Partial Points) For Memory <= 80_000_000_000
- 0 For Memory > 80_000_000_000

Please note that the thresholds above are subject to change.



## Timing and Memory metric details 

You will also be required to create a custom Model training strategy that uses a combination of techniques above or any other technique that you can think of. The training strategy will be evaluated on the MFU value in displayed in the Leaderboard.

We have provided a utility function to help you measure the memory and time metrics in `utils.py` . You can use the following function to measure the memory and time metrics:

`time, memory, max_mem_rank = get_training_stats(out)`


## Testing Your Code

Test your implementations using the provided utility function: Also note the actual time/memory usage of the model will be different from the one you see because of different model parameters in the tests. 
You can also simply run the particular file using `python3 partX.py` to test your implementation.

```python
async def run_model():
    world_size = 4
    num_layers = 4
    global_batch_size = 16

    dist = Dist(world_size)
    models: List[Model] = [Model(rank, dist, num_layers, global_batch_size) for rank in range(world_size)]

    out = await asyncio.gather(*(your_training_function(model) for model in models))

    time, memory, max_mem_rank = get_training_stats(out)
    
    write_chrome_trace(out, './debug_traces/1f1b.json')



if __name__ == "__main__":
    asyncio.run(run_model())
```

Please note that the actual test cases will be different from the ones provided here. We can use any combination of world_size, num_layers, global_batch_size and batch_size (where applicable).

Replace `your_training_function` with the appropriate function name for each part.

To visualize the training process in batches, you can use the `write_chrome_trace` function. This will generate a chrome trace of the training process. You can open this in `chrome://tracing` to see the training process in batches. Make sure that the training actually happens in the correct order. These will be checked later and you will be penalized if the training process is not correct.

You will have to replace the training function with the appropriate function name for each part to visualize the training process is happening correctly.

We strongly suggest playing around with the code and build simple training loops get a better understanding of the training process and how the `lib.py` and `utils.py` files work.

For reading the chrome trace, you can use the following steps:
1. Load the chrome trace file in chrome://tracing using load button.
2. Then the screen will display the different ranks top to bottom.
3. In any rank, you can see the different operations being performed in that rank from left to right in increasing order of time.
4. Clicking over any operation will show you the details of the operation. (Layer, Batch, Time, Memory, etc.)
5. You can use this to verify the correctness of the training process and also to see if there are any bottlenecks in the training process and optimizations to be made.

## Submission Guidelines

1. Submit your completed `part1.py`, `part2.py`, `part3.py`, and `part4.py` files.


## Credits 

This lab is based on the amazing [LLM Training Puzzles](https://github.com/srush/LLM-Training-Puzzles) by [Sasha Rush](https://rush-nlp.com/) 

## Additional Details and Penalties

- You cannot modify the `lib.py` files. The signature of the skeleton code should not be changed. Any modification to the signature will result in a penalty of up to 100% of the points for that part depending on the severity of the violation.

- If there are any errors or doubts in the code, please contact the TA or the instructor by making a private post on Piazza (Do not E-mail doubts)

Good Luck! 
