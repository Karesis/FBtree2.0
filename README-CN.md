
# FBtree 2.0

FBtree 2.0 是一个高性能、线程安全的树结构库，为Python应用提供原子操作和并行计算支持。

## 特性

- **线程和进程安全**：所有树操作都使用锁保护，确保线程安全
- **并发处理**：支持基于线程和基于进程的并行计算
- **原子操作**：在多线程环境中安全地更新树节点属性
- **灵活的路径追踪**：高效导航和修改决策路径
- **属性系统**：可以为树节点关联任意数据
- **可视化**：内置树可视化功能
- **蒙特卡洛树搜索支持**：可直接使用的MCTS实现
- **序列化**：保存和加载树结构

## 安装

```bash
pip install fbtcore
```

## 快速入门

### 基本用法

```python
from fbtcore import FBTree, Move

# 创建树
tree = FBTree(attribute_template={"visits": 0, "value": 0.0})

# 创建路径追踪器来导航树
tracer = tree.create_tracer()

# 添加动作构建树
tracer.add_move([1, 2, 3])  # Move可以是列表、元组或numpy数组
tracer.add_move([4, 5, 6])

# 获取当前路径
path = tracer.get_current_path()
print(f"当前路径: {path}")

# 回溯并选择不同路径
tracer.backtrack()
tracer.add_move([7, 8, 9])

# 可视化树
tree.visualize(show_attributes=["visits", "value"])
```

### 线程安全操作

```python
from fbtcore import create_safe_tree, ConcurrencyMode

# 创建线程安全的树
safe_tree = create_safe_tree(
    attribute_template={"visits": 0, "value": 0.0},
    concurrency_mode=ConcurrencyMode.THREAD
)

# 获取根fiber
root = safe_tree.get_root()

# 执行原子操作
safe_tree.increment(root, "visits")  # 线程安全增量
safe_tree.set_attribute(root, "value", 1.0)  # 线程安全属性更新

# 自定义原子操作
def update_operation(fiber):
    visits = fiber.get_attribute("visits")
    value = fiber.get_attribute("value")
    return visits * value

result = safe_tree.atomic_operation(root, update_operation)
```

### 并行处理

```python
from fbtcore import create_safe_tree, ConcurrencyMode, Move

safe_tree = create_safe_tree(
    attribute_template={"visits": 0, "value": 0.0, "mean": 0.0},
    concurrency_mode=ConcurrencyMode.PROCESS  # 使用基于进程的并行
)

# 定义一个在并行环境中运行的函数（对于进程模式必须是模块级函数）
def process_fiber(fiber):
    visits = fiber.get_attribute("visits", 0) + 1
    fiber.set_attribute("visits", visits)
    
    value = fiber.get_attribute("value", 0.0) + 0.5
    fiber.set_attribute("value", value)
    
    mean = value / visits
    fiber.set_attribute("mean", mean)
    
    return {"visits": visits, "value": value, "mean": mean}

# 构建树
root = safe_tree.get_root()
fibers = [root]
for i in range(10):
    next_fiber = safe_tree.add_move(fibers[-1], Move([i]))
    fibers.append(next_fiber)

# 并行处理所有fiber
results = safe_tree.parallel_map(fibers, process_fiber)
```

### 蒙特卡洛树搜索

```python
import random
from fbtcore import create_mcts_tree, ucb_select

# 创建用于MCTS的树
mcts_tree = create_mcts_tree()
root = mcts_tree.get_root()

# 定义搜索函数
def select_func(node):
    return ucb_select(mcts_tree, node, exploration_weight=1.0)

def simulate_func(node):
    # 从节点开始模拟
    path = node.get_full_path()
    # 基于路径计算奖励
    return 0.5 + 0.5 * random.random()

def backpropagate_func(node, reward):
    mcts_tree.increment(node, "visit_count")
    total = mcts_tree.update_attribute(node, "total_value", lambda x: x + reward)
    visits = mcts_tree.get_attribute(node, "visit_count")
    mcts_tree.set_attribute(node, "mean_value", total / visits)

# 运行搜索
best_node = mcts_tree.search(
    root=root,
    select_func=select_func,
    simulate_func=simulate_func,
    backpropagate_func=backpropagate_func,
    num_simulations=100
)
```

## API参考

### 核心类

- **FBTree**：基本树实现，支持属性、路径追踪和可视化
- **Fiber**：表示决策路径段，包含属性和连接关系
- **Move**：表示动作（基于numpy数组），可以从列表、元组或数组创建
- **PathTracer**：用于导航树路径的工具，支持回溯

### 并发类

- **SafeFBTree**：线程安全的树实现，支持原子操作和并行计算
  - **parallel_map()**：并行运行函数处理多个fiber
  - **atomic_operation()**：在fiber上原子执行操作
  - **propagate()**：沿路径更新属性
  - **search()**：使用自定义函数进行树搜索
- **SafeTracer**：线程安全的路径追踪器，用于并发树导航
- **ConcurrencyMode**：用于选择线程或进程模式的枚举（THREAD/PROCESS）

### 工厂函数

- **create_safe_tree()**：创建具有指定属性的线程或进程安全树
- **create_mcts_tree()**：创建针对蒙特卡洛树搜索优化的树，包含MCTS专用属性
- **ucb_select()**：用于MCTS的UCB1选择函数，具有可配置的探索权重

## 性能

FBtree 2.0为单线程和并行操作提供卓越的性能。并行实现可以利用多个CPU核心进行以下操作：

- 并行树更新
- 并发fiber属性修改
- 并行值传播
- 蒙特卡洛树搜索

## 要求

- Python 3.7+
- NumPy
- Matplotlib（可选，用于可视化）

## 许可证

LGPL-3.0
