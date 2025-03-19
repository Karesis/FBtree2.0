"""
FBTree的并发扩展模块，提供进程安全操作和并行计算
"""

import threading
import enum
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from typing import Dict, List, Optional, Any, Union, Sequence, Callable, Tuple
import pickle
import functools

# 导入core模块中的类
from .core import FBTree, Move, Fiber, PathTracer

class ConcurrencyMode(enum.Enum):
    """并发模式枚举"""
    THREAD = "thread"  # 仅用于IO密集型任务
    PROCESS = "process"  # 默认，支持CPU密集型任务

# === 全局函数，用于多进程序列化 ===

def _process_fiber_operation(func_name, fiber_id, fiber_pickle, args=(), kwargs_pickle=None):
    """
    处理序列化后的fiber操作，可跨进程使用
    
    Args:
        func_name: 函数名称 (替代序列化函数)
        fiber_id: fiber的ID
        fiber_pickle: 序列化的fiber
        args: 位置参数元组
        kwargs_pickle: 序列化的关键字参数字典
    
    Returns:
        (fiber_id, 结果)元组
    """
    try:
        # 反序列化fiber
        fiber = pickle.loads(fiber_pickle)
        
        # 处理关键字参数
        kwargs = {}
        if kwargs_pickle:
            kwargs = pickle.loads(kwargs_pickle)
        
        # 查找全局函数（必须是模块级别的函数）
        import sys
        for module_name, module in sys.modules.items():
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                # 执行函数
                result = func(fiber, *args, **kwargs)
                return (fiber_id, result)
                
        # 如果找不到函数，报错
        return (fiber_id, f"ERROR: 找不到函数 {func_name}")
    except Exception as e:
        # 捕获并返回错误，避免进程崩溃
        return (fiber_id, f"ERROR: {str(e)}")

def _safe_pickle(obj):
    """安全序列化对象，处理不可序列化对象"""
    try:
        return pickle.dumps(obj)
    except (pickle.PickleError, TypeError):
        return None

def _process_params_unpacker(params):
    """全局函数用于解包参数给_process_fiber_operation"""
    return _process_fiber_operation(*params)

class SafeFBTree:
    """
    进程安全的FBTree封装
    """
    
    def __init__(self, 
                 base_tree: Optional[FBTree] = None, 
                 attribute_template: Optional[Dict[str, Any]] = None,
                 concurrency_mode: Union[ConcurrencyMode, str] = ConcurrencyMode.PROCESS,
                 max_workers: Optional[int] = None):
        """
        初始化安全的FBTree
        
        Args:
            base_tree: 现有FBTree，如None则创建新的
            attribute_template: 新树的属性模板
            concurrency_mode: 并发模式，默认为PROCESS
            max_workers: 工作进程/线程数量，默认为CPU核心数
        """
        # 创建或使用提供的FBTree
        self.tree = base_tree if base_tree else FBTree(attribute_template)
        
        # 并发模式处理
        if isinstance(concurrency_mode, str):
            try:
                self.concurrency_mode = ConcurrencyMode(concurrency_mode)
            except ValueError:
                self.concurrency_mode = ConcurrencyMode.PROCESS
                print(f"未知并发模式 '{concurrency_mode}'，使用默认进程模式")
        else:
            self.concurrency_mode = concurrency_mode
        
        # 工作池大小，默认为核心数
        self.max_workers = max_workers if max_workers is not None else multiprocessing.cpu_count()
        
        # 锁系统 - 保留线程锁用于单进程内同步
        self._global_lock = threading.RLock()  # 全局树结构锁
        self._fiber_locks = {}                 # 每个Fiber的数据锁
        self._locks_lock = threading.RLock()   # 保护_fiber_locks的锁
    
    # === 锁管理 ===
    
    def _get_fiber_lock(self, fiber: Fiber) -> threading.RLock:
        """获取fiber的锁，如不存在则创建"""
        with self._locks_lock:
            fiber_id = id(fiber)
            if fiber_id not in self._fiber_locks:
                self._fiber_locks[fiber_id] = threading.RLock()
            return self._fiber_locks[fiber_id]
    
    # === 基本操作 ===
    
    def get_root(self) -> Fiber:
        """获取根节点"""
        return self.tree.root
    
    def create_tracer(self) -> 'SafeTracer':
        """创建安全的PathTracer"""
        with self._global_lock:
            tracer = self.tree.create_tracer()
            return SafeTracer(tracer, self)
    
    def visualize(self, max_depth=None, show_attributes=None):
        """可视化树"""
        with self._global_lock:
            self.tree.visualize(max_depth, show_attributes)
    
    def count_fibers(self) -> int:
        """计算fiber数量"""
        with self._global_lock:
            return self.tree.count_fibers()
    
    def save(self, filepath: str):
        """保存树"""
        with self._global_lock:
            self.tree.save(filepath)
    
    @classmethod
    def load(cls, filepath: str, 
            concurrency_mode=ConcurrencyMode.PROCESS, 
            max_workers: Optional[int] = None) -> 'SafeFBTree':
        """从文件加载树"""
        tree = FBTree.load(filepath)
        return cls(base_tree=tree, concurrency_mode=concurrency_mode, max_workers=max_workers)
    
    # === 安全的Fiber操作 ===
    
    def get_attribute(self, fiber: Fiber, name: str, default: Any = None) -> Any:
        """安全地获取属性"""
        with self._get_fiber_lock(fiber):
            return fiber.get_attribute(name, default)
    
    def set_attribute(self, fiber: Fiber, name: str, value: Any):
        """安全地设置属性"""
        with self._get_fiber_lock(fiber):
            fiber.set_attribute(name, value)
    
    def update_attribute(self, fiber: Fiber, name: str, update_func: Callable[[Any], Any]) -> Any:
        """原子地更新属性"""
        with self._get_fiber_lock(fiber):
            value = fiber.get_attribute(name)
            new_value = update_func(value)
            fiber.set_attribute(name, new_value)
            return new_value
    
    def atomic_operation(self, fiber: Fiber, operation: Callable[[Fiber], Any]) -> Any:
        """执行原子操作，确保整个操作序列在同一个锁中完成"""
        with self._get_fiber_lock(fiber):
            return operation(fiber)
    
    def increment(self, fiber: Fiber, name: str, amount: Union[int, float] = 1) -> Union[int, float]:
        """原子增加属性值"""
        with self._get_fiber_lock(fiber):
            value = fiber.get_attribute(name)
            new_value = value + amount
            fiber.set_attribute(name, new_value)
            return new_value
    
    def add_move(self, fiber: Fiber, move: Move) -> Fiber:
        """安全地添加move"""
        with self._global_lock:  # 修改树结构需要全局锁
            return fiber.add_move(move)
    
    # === 并行操作 ===
    
    def parallel_map(self, items, func, chunksize=1, *args, **kwargs):
        """
        并行映射操作到多个项目
        
        Args:
            items: 要处理的对象列表
            func: 处理对象的函数，必须是可序列化的
            chunksize: 每批处理的项目数量
            *args, **kwargs: 传递给func的其他参数
            
        Returns:
            结果列表
        """
        if not items:
            return []
        
        # 使用进程池（默认推荐模式）
        if self.concurrency_mode == ConcurrencyMode.PROCESS:
            # 获取函数名称代替序列化函数
            func_name = func.__name__
            
            # 序列化kwargs（如果有）
            kwargs_pickle = pickle.dumps(kwargs) if kwargs else None
            
            # 准备参数
            fiber_ids = []
            fiber_pickles = []
            
            for item in items:
                fiber_ids.append(id(item))
                pickled = _safe_pickle(item)
                if pickled is None:
                    raise ValueError(f"无法序列化对象，请确保您使用的函数和数据可跨进程传输")
                fiber_pickles.append(pickled)
            
            # 使用进程池执行
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # 构建参数列表，包含kwargs
                args_list = [
                    (func_name, fid, pickle_data, args, kwargs_pickle) 
                    for fid, pickle_data in zip(fiber_ids, fiber_pickles)
                ]
                # 使用全局函数而不是lambda
                raw_results = list(executor.map(_process_params_unpacker, args_list, chunksize=chunksize))
                
                # 创建结果映射，以便保持顺序
                result_map = {fid: res for fid, res in raw_results}
                results = [result_map.get(fid) for fid in fiber_ids]
                
                return results
        
        # 线程池模式（仅用于IO密集型任务）
        elif self.concurrency_mode == ConcurrencyMode.THREAD:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 包装函数，确保锁保护
                @functools.wraps(func)
                def thread_func(item):
                    if isinstance(item, Fiber):
                        with self._get_fiber_lock(item):
                            return func(item, *args, **kwargs)
                    else:
                        return func(item, *args, **kwargs)
                
                # 使用map并传入chunksize
                return list(executor.map(thread_func, items, chunksize=chunksize))
        
        # 同步执行（单线程）
        else:
            return [func(item, *args, **kwargs) for item in items]
    
    def parallel_for_each(self, items, action, chunksize=1, *args, **kwargs):
        """
        并行执行操作但不返回结果（用于副作用）
        
        Args:
            items: 要处理的对象列表
            action: 处理对象的函数
            chunksize: 每批处理的项目数量
            *args, **kwargs: 传递给action的其他参数
        """
        self.parallel_map(items, action, chunksize, *args, **kwargs)
    
    def propagate(self, 
                fiber: Fiber, 
                update_func: Callable[[Fiber], None],
                parallel: bool = True):
        """
        沿路径传播更新
        
        Args:
            fiber: 起始fiber
            update_func: 更新函数
            parallel: True使用并行，False使用单线程
        """
        # 获取完整路径
        path = []
        current = fiber
        while current:
            path.append(current)
            current = current.prev_fiber
        
        if not parallel:
            # 同步执行
            for node in path:
                with self._get_fiber_lock(node):
                    update_func(node)
        else:
            # 并行执行
            self.parallel_for_each(path, update_func)
    
    def search(self, 
              root: Fiber,
              select_func: Callable[[Fiber], Fiber],
              simulate_func: Callable[[Fiber], float],
              backpropagate_func: Optional[Callable[[Fiber, float], None]] = None,
              num_simulations: int = 100) -> Optional[Fiber]:
        """
        并行树搜索
        
        Args:
            root: 根节点
            select_func: 节点选择函数
            simulate_func: 模拟函数，返回评估值
            backpropagate_func: 回溯更新函数，默认为None则使用标准更新
            num_simulations: 模拟次数
            
        Returns:
            最佳子节点
        """
        # 如果没有提供回溯函数，使用默认函数
        if backpropagate_func is None:
            def backpropagate_func(node, reward):
                self.update_attribute(node, 'visit_count', lambda x: x + 1)
                total = self.update_attribute(node, 'total_value', lambda x: x + reward)
                visits = self.get_attribute(node, 'visit_count')
                self.set_attribute(node, 'mean_value', total / visits)
        
        # 定义单次模拟函数
        def run_simulation():
            try:
                # 选择
                leaf = select_func(root)
                
                # 模拟
                reward = simulate_func(leaf)
                
                # 回溯更新
                current = leaf
                while current:
                    with self._get_fiber_lock(current):
                        backpropagate_func(current, reward)
                    current = current.prev_fiber
                
                return reward
            except Exception as e:
                print(f"模拟任务出错: {str(e)}")
                return None
        
        # 创建任务列表
        simulation_tasks = [lambda: run_simulation() for _ in range(num_simulations)]
        
        # 进程模式特殊处理
        if self.concurrency_mode == ConcurrencyMode.PROCESS:
            # 多进程模式下，单线程执行搜索（暂不支持进程间搜索）
            for _ in range(num_simulations):
                run_simulation()
        else:
            # 线程模式并行执行
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                list(executor.map(lambda f: f(), simulation_tasks))
        
        # 选择最佳子节点
        return self._select_best_child(root)
    
    def _select_best_child(self, root: Fiber) -> Optional[Fiber]:
        """选择最佳子节点（访问次数最多）"""
        best_child = None
        best_visits = -1
        
        with self._get_fiber_lock(root):
            for move, child in root.next_fibers.items():
                visits = self.get_attribute(child, 'visit_count', 0)
                if visits > best_visits:
                    best_visits = visits
                    best_child = child
        
        return best_child


class SafeTracer:
    """安全的PathTracer封装类"""
    
    def __init__(self, tracer: PathTracer, safe_tree: SafeFBTree):
        """
        初始化
        
        Args:
            tracer: 原始PathTracer
            safe_tree: 安全树
        """
        self._tracer = tracer
        self._tree = safe_tree
        self._lock = threading.RLock()
    
    def add_move(self, move: Union[Sequence, np.ndarray, Move]) -> bool:
        """添加move"""
        with self._lock:
            with self._tree._global_lock:  # 添加move需要全局锁
                return self._tracer.add_move(move)
    
    def backtrack(self, steps: int = 1) -> bool:
        """回溯指定步数"""
        with self._lock:
            return self._tracer.backtrack(steps)
    
    def get_current_path(self) -> List[Move]:
        """获取当前路径"""
        with self._lock:
            return self._tracer.get_current_path()
    
    def reset(self):
        """重置追踪器"""
        with self._lock:
            self._tracer.reset()
    
    @property
    def current_fiber(self) -> Fiber:
        """获取当前fiber"""
        with self._lock:
            return self._tracer.current_fiber


# ===== 便捷函数 =====

def create_safe_tree(base_tree: Optional[FBTree] = None, 
                     attribute_template: Optional[Dict[str, Any]] = None,
                     concurrency_mode: Union[str, ConcurrencyMode] = ConcurrencyMode.PROCESS,
                     max_workers: Optional[int] = None) -> SafeFBTree:
    """创建安全的FBTree"""
    return SafeFBTree(
        base_tree=base_tree,
        attribute_template=attribute_template,
        concurrency_mode=concurrency_mode,
        max_workers=max_workers
    )


def create_mcts_tree(attribute_template: Optional[Dict[str, Any]] = None,
                     concurrency_mode: Union[str, ConcurrencyMode] = ConcurrencyMode.PROCESS,
                     max_workers: Optional[int] = None) -> SafeFBTree:
    """创建适用于蒙特卡洛树搜索的安全树"""
    default_attributes = {
        'visit_count': 0,
        'total_value': 0.0,
        'mean_value': 0.0
    }
    
    if attribute_template:
        # 合并默认属性和自定义属性
        merged_attributes = default_attributes.copy()
        merged_attributes.update(attribute_template)
        attribute_template = merged_attributes
    else:
        attribute_template = default_attributes
    
    return create_safe_tree(
        attribute_template=attribute_template,
        concurrency_mode=concurrency_mode,
        max_workers=max_workers
    )


def ucb_select(tree: SafeFBTree, node: Fiber, exploration_weight: float = 1.0) -> Fiber:
    """使用UCB公式选择节点"""
    # 创建原子选择函数
    def select_ucb():
        current = node
        
        while True:
            with tree._get_fiber_lock(current):
                # 叶节点直接返回
                if not current.next_fibers:
                    return current
                
                best_score = -float('inf')
                best_child = None
                parent_visits = current.get_attribute('visit_count', 1)
                
                # 遍历所有子节点
                for move, child in current.next_fibers.items():
                    child_visits = tree.get_attribute(child, 'visit_count', 0)
                    
                    # 未访问过的节点优先
                    if child_visits == 0:
                        return child
                    
                    # 计算UCB值
                    mean_value = tree.get_attribute(child, 'mean_value', 0.0)
                    exploration = exploration_weight * np.sqrt(np.log(parent_visits) / child_visits)
                    score = mean_value + exploration
                    
                    if score > best_score:
                        best_score = score
                        best_child = child
                
                # 更新当前节点
                if best_child:
                    current = best_child
                else:
                    return current
    
    # 执行原子选择
    return tree.atomic_operation(node, lambda _: select_ucb())