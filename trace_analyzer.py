import json
import sys

def analyze_trace(trace_file):
    """
    Reverse engineer training parameters from a Chrome trace file.
    """
    with open(trace_file, 'r') as f:
        trace_data = json.load(f)
    
    events = trace_data.get('traceEvents', [])
    
    # Extract unique process IDs (ranks)
    ranks = set()
    layers = set()
    samples = set()
    
    # Track forward events to understand microbatches
    forward_events = []
    
    for event in events:
        # Skip metadata events
        if event.get('ph') == 'M':
            continue
            
        # Get rank from process ID
        pid = event.get('pid')
        if pid is not None:
            ranks.add(pid)
        
        # Get layer info from event name and args
        name = event.get('name', '')
        args = event.get('args', {})
        
        if 'layer' in args:
            layers.add(args['layer'])
        
        # Extract sample information from forward/backward events
        if 'samples' in args and args['samples']:
            # Parse the samples string or set
            samples_str = str(args['samples'])
            # Try to extract individual sample IDs
            if 'frozenset' in samples_str or '{' in samples_str:
                # Extract numbers from the set representation
                import re
                sample_ids = re.findall(r'\d+', samples_str)
                for sid in sample_ids:
                    samples.add(int(sid))
        
        # Track forward events specifically
        if name == 'forward':
            forward_events.append(event)
    
    # Calculate parameters
    world_size = len(ranks)
    num_layers = len(layers)
    global_batch_size = len(samples) if samples else None
    
    # Estimate batch_size by looking at microbatches
    # Group forward events by rank and find unique sample sets
    microbatch_sizes = set()
    rank_forwards = {}
    
    for event in forward_events:
        pid = event.get('pid')
        args = event.get('args', {})
        
        if pid not in rank_forwards:
            rank_forwards[pid] = []
        
        # Try to determine microbatch size from samples
        samples_str = str(args.get('samples', ''))
        if samples_str:
            import re
            sample_ids = re.findall(r'\d+', samples_str)
            if sample_ids:
                microbatch_sizes.add(len(sample_ids))
    
    # Most common microbatch size
    batch_size = max(microbatch_sizes) if microbatch_sizes else None
    
    # Calculate number of microbatches
    num_microbatches = global_batch_size // batch_size if (global_batch_size and batch_size) else None
    
    # Verify with forward count per rank
    forwards_per_rank = {rank: 0 for rank in ranks}
    for event in forward_events:
        pid = event.get('pid')
        if pid in forwards_per_rank:
            forwards_per_rank[pid] += 1
    
    print("=" * 60)
    print("TRACE ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\n📊 Training Configuration:")
    print(f"  world_size         = {world_size}")
    print(f"  num_layers         = {num_layers}")
    print(f"  global_batch_size  = {global_batch_size}")
    print(f"  batch_size         = {batch_size}")
    print(f"  num_microbatches   = {num_microbatches}")
    
    print(f"\n🔍 Verification:")
    print(f"  Unique ranks: {sorted(ranks)}")
    print(f"  Layer range: {min(layers)} to {max(layers)}")
    print(f"  Layers per rank: {num_layers // world_size if world_size else 'N/A'}")
    print(f"  Sample range: {min(samples) if samples else 'N/A'} to {max(samples) if samples else 'N/A'}")
    
    print(f"\n📈 Forward Passes per Rank:")
    for rank in sorted(forwards_per_rank.keys()):
        count = forwards_per_rank[rank]
        layers_per_rank = num_layers // world_size
        expected_per_layer = num_microbatches
        expected_total = layers_per_rank * expected_per_layer
        print(f"  Rank {rank}: {count} forwards (expected: {expected_total} = {layers_per_rank} layers × {expected_per_layer} microbatches)")
    
    print("\n" + "=" * 60)
    print("💡 Use these parameters in your code:")
    print("=" * 60)
    print(f"""
async def main():
    world_size = {world_size}
    num_layers = {num_layers}
    global_batch_size = {global_batch_size}
    batch_size = {batch_size}
    
    # ... rest of your code
""")
    
    return {
        'world_size': world_size,
        'num_layers': num_layers,
        'global_batch_size': global_batch_size,
        'batch_size': batch_size,
        'num_microbatches': num_microbatches
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trace_analyzer.py <trace_file.json>")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    analyze_trace(trace_file)