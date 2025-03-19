"""
FBtree 2.0 - 高性能线程安全的树结构库，支持原子操作和并行计算

该库提供线程安全的树操作、原子更新和并行计算功能，专为需要并发访问的应用设计。
"""

__version__ = "2.0.0"

# 从core模块导入核心类
from .core import FBTree, Fiber, Move, PathTracer

# 从concurrent模块导入并发相关类和函数
from .concurrent import (
    SafeFBTree,
    ConcurrencyMode,
    create_safe_tree,
    create_mcts_tree,
    ucb_select
)

# 公开的API列表
__all__ = [
    # 版本信息
    "__version__",
    
    # 核心类
    "FBTree",
    "Fiber",
    "Move",
    "PathTracer",
    
    # 并发相关
    "SafeFBTree",
    "ConcurrencyMode",
    "create_safe_tree",
    "create_mcts_tree",
    "ucb_select"
]