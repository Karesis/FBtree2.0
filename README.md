
# FBtree 2.0

FBtree 2.0 is a high-performance, thread-safe tree structure library supporting atomic operations and parallel computation for Python applications.

## Features

- **Thread and Process Safety**: All tree operations are protected with locks to ensure thread safety
- **Concurrent Processing**: Support for both thread-based and process-based parallelism
- **Atomic Operations**: Update tree node attributes safely in multi-threaded environments
- **Flexible Path Tracing**: Efficiently navigate and modify decision paths
- **Attribute System**: Associate arbitrary data with tree nodes
- **Visualization**: Built-in tree visualization capabilities
- **Monte Carlo Tree Search Support**: Ready-to-use MCTS implementation
- **Serialization**: Save and load tree structures

## Installation

```bash
pip install fbtcore
```

## Quick Start

### Basic Usage

```python
from fbtcore import FBTree, Move

# Create a tree
tree = FBTree(attribute_template={"visits": 0, "value": 0.0})

# Create a path tracer to navigate the tree
tracer = tree.create_tracer()

# Add moves to build the tree
tracer.add_move([1, 2, 3])  # Move can be a list, tuple, or numpy array
tracer.add_move([4, 5, 6])

# Get current path
path = tracer.get_current_path()
print(f"Current path: {path}")

# Backtrack and take a different path
tracer.backtrack()
tracer.add_move([7, 8, 9])

# Visualize the tree
tree.visualize(show_attributes=["visits", "value"])
```

### Thread-Safe Operations

```python
from fbtcore import create_safe_tree, ConcurrencyMode

# Create a thread-safe tree
safe_tree = create_safe_tree(
    attribute_template={"visits": 0, "value": 0.0},
    concurrency_mode=ConcurrencyMode.THREAD
)

# Get the root fiber
root = safe_tree.get_root()

# Perform atomic operations
safe_tree.increment(root, "visits")  # Thread-safe increment
safe_tree.set_attribute(root, "value", 1.0)  # Thread-safe attribute update

# Custom atomic operations
def update_operation(fiber):
    visits = fiber.get_attribute("visits")
    value = fiber.get_attribute("value")
    return visits * value

result = safe_tree.atomic_operation(root, update_operation)
```

### Parallel Processing

```python
from fbtcore import create_safe_tree, ConcurrencyMode, Move

safe_tree = create_safe_tree(
    attribute_template={"visits": 0, "value": 0.0, "mean": 0.0},
    concurrency_mode=ConcurrencyMode.PROCESS  # Use process-based parallelism
)

# Define a function to run in parallel (must be module-level for process mode)
def process_fiber(fiber):
    visits = fiber.get_attribute("visits", 0) + 1
    fiber.set_attribute("visits", visits)
    
    value = fiber.get_attribute("value", 0.0) + 0.5
    fiber.set_attribute("value", value)
    
    mean = value / visits
    fiber.set_attribute("mean", mean)
    
    return {"visits": visits, "value": value, "mean": mean}

# Build the tree
root = safe_tree.get_root()
fibers = [root]
for i in range(10):
    next_fiber = safe_tree.add_move(fibers[-1], Move([i]))
    fibers.append(next_fiber)

# Process all fibers in parallel
results = safe_tree.parallel_map(fibers, process_fiber)
```

### Monte Carlo Tree Search

```python
import random
from fbtcore import create_mcts_tree, ucb_select

# Create a tree for MCTS
mcts_tree = create_mcts_tree()
root = mcts_tree.get_root()

# Define search functions
def select_func(node):
    return ucb_select(mcts_tree, node, exploration_weight=1.0)

def simulate_func(node):
    # Run simulation from node
    path = node.get_full_path()
    # Compute reward based on path
    return 0.5 + 0.5 * random.random()

def backpropagate_func(node, reward):
    mcts_tree.increment(node, "visit_count")
    total = mcts_tree.update_attribute(node, "total_value", lambda x: x + reward)
    visits = mcts_tree.get_attribute(node, "visit_count")
    mcts_tree.set_attribute(node, "mean_value", total / visits)

# Run the search
best_node = mcts_tree.search(
    root=root,
    select_func=select_func,
    simulate_func=simulate_func,
    backpropagate_func=backpropagate_func,
    num_simulations=100
)
```

## API Reference

### Core Classes

- **FBTree**: Basic tree implementation with support for attributes, path tracing, and visualization
- **Fiber**: Represents a decision path segment, containing attributes and connections
- **Move**: Represents an action (based on numpy arrays), can be created from lists, tuples, or arrays
- **PathTracer**: Utilities for navigating tree paths, with backtracking support

### Concurrent Classes

- **SafeFBTree**: Thread-safe tree implementation with atomic operations and parallelism support
  - **parallel_map()**: Run functions on multiple fibers in parallel
  - **atomic_operation()**: Execute operations atomically on a fiber
  - **propagate()**: Update attributes along a path
  - **search()**: Run tree search with custom functions
- **SafeTracer**: Thread-safe path tracer for concurrent tree navigation
- **ConcurrencyMode**: Enum for selecting thread or process mode (THREAD/PROCESS)

### Factory Functions

- **create_safe_tree()**: Create a thread or process-safe tree with the specified attributes
- **create_mcts_tree()**: Create a tree optimized for Monte Carlo Tree Search with MCTS-specific attributes
- **ucb_select()**: UCB1 selection function for MCTS with configurable exploration weight

## Performance

FBtree 2.0 provides excellent performance for both single-threaded and parallel operations. The parallel implementation can take advantage of multiple CPU cores for operations like:

- Parallel tree updates
- Concurrent node attribute modifications
- Parallel propagation of values
- Monte Carlo Tree Search

## Requirements

- Python 3.7+
- NumPy
- Matplotlib (optional, for visualization)

## License

LGPL-3.0
