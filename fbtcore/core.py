import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Sequence, Set


class Move:
    """表示一个动作，支持从Python列表/数组创建"""
    
    def __init__(self, data: Union[Sequence, np.ndarray, int, float]):
        # 标量类型自动包装为列表
        if isinstance(data, (int, float)):
            data = [data]
        # 字符串检查
        elif isinstance(data, str):
            raise TypeError(f"Move需要列表、元组或NumPy数组类型的输入，而不是字符串")
        # 其他非序列类型
        elif not isinstance(data, (np.ndarray, Sequence)):
            raise TypeError(f"Move需要列表、元组或NumPy数组类型的输入，而不是{type(data).__name__}")
        
        # 转换为numpy数组
        self.data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data
        
    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return np.array_equal(self.data, other.data)
    
    def __hash__(self):
        return hash(self.data.tobytes())
    
    def __str__(self):
        return f"Move({self.data})"
    
    def __repr__(self):
        return self.__str__()


class Fiber:
    """表示一个完整的决策路径段"""
    
    def __init__(self, tree: 'FBTree', prev_fiber: Optional['Fiber'] = None, 
                last_move: Optional[Move] = None):
        self.tree = tree
        self.moves: List[Move] = []
        self.next_fibers: Dict[Move, 'Fiber'] = {}
        self.prev_fiber = prev_fiber
        self.last_move = last_move
        self.attributes = self._create_attributes()
    
    def _create_attributes(self) -> Dict[str, Any]:
        """根据树的属性模板创建属性字典"""
        attributes = {}
        for name, default_value in self.tree.attribute_template.items():
            attributes[name] = default_value() if callable(default_value) else default_value
        return attributes
    
    def add_move(self, move: Move) -> 'Fiber':
        """添加一个move到当前fiber，返回对应的fiber"""
        # 如果已存在此move的next_fiber，直接返回
        if move in self.next_fibers:
            return self.next_fibers[move]
            
        # 创建新的next_fiber
        next_fiber = Fiber(self.tree, prev_fiber=self, last_move=move)
        self.next_fibers[move] = next_fiber
        return next_fiber
    
    def get_attribute(self, name: str, default: Any = None) -> Any:
        """获取属性值，不存在则返回默认值"""
        if name not in self.attributes:
            if default is not None:
                return default
            raise KeyError(f"属性'{name}'不存在")
        return self.attributes[name]
    
    def set_attribute(self, name: str, value: Any):
        """设置属性值 - 仅限已定义的属性"""
        if name not in self.attributes:
            raise KeyError(f"无法设置未定义的属性'{name}'。"
                          f"可用属性: {list(self.attributes.keys())}")
        self.attributes[name] = value
    
    def get_full_path(self) -> List[Move]:
        """获取从根到此fiber的完整路径"""
        path = []
        current = self
        
        # 从当前fiber回溯到根
        while current:
            # 添加当前fiber的moves
            if current.moves:
                # 需要逆序插入当前fiber的moves
                for move in reversed(current.moves):
                    path.insert(0, move)
            
            # 添加连接move
            if current.last_move:
                path.insert(0, current.last_move)
            
            current = current.prev_fiber
            
        return path


class PathTracer:
    """路径追踪器 - 用于跟踪当前在树中的位置"""
    
    def __init__(self, tree: 'FBTree'):
        self.tree = tree
        self.current_fiber = tree.root
        self.position = 0  # 在当前fiber中的位置
        self.current_path: List[Move] = []  # 当前经过的完整路径
    
    def add_move(self, move: Union[Sequence, np.ndarray, Move]) -> bool:
        """添加一个move，向前移动路径"""
        # 转换为Move对象
        if not isinstance(move, Move):
            move = Move(move)
            
        # 记录到当前路径
        self.current_path.append(move)
        
        # 判断是否需要切换到next_fiber
        if self.position >= len(self.current_fiber.moves):
            # 当前fiber已结束，进入下一个fiber
            self.current_fiber = self.current_fiber.add_move(move)
            self.position = 0
            return True
            
        # 检查当前位置的move是否匹配
        if move == self.current_fiber.moves[self.position]:
            # move匹配，前进位置
            self.position += 1
            return True
            
        # move不匹配，需要分叉
        return self._fork_path(move)
    
    def _fork_path(self, move: Move) -> bool:
        """创建分叉路径"""
        # 保存剩余moves
        remaining_moves = self.current_fiber.moves[self.position:]
        
        # 修剪当前fiber
        self.current_fiber.moves = self.current_fiber.moves[:self.position]
        
        # 为新move创建分支
        new_fiber = self.current_fiber.add_move(move)
        self.current_fiber = new_fiber
        self.position = 0
        
        # 为原来剩余的moves创建分支
        if remaining_moves:
            original_next = self.current_fiber.prev_fiber.add_move(remaining_moves[0])
            for m in remaining_moves[1:]:
                original_next.moves.append(m)
        
        return True
    
    def backtrack(self, steps: int = 1) -> bool:
        """回溯指定步数"""
        for _ in range(steps):
            if not self._backtrack_one_step():
                return False
        return True
    
    def _backtrack_one_step(self) -> bool:
        """回溯一步"""
        if self.position > 0:
            # 在当前fiber内回溯
            self.position -= 1
            if self.current_path:
                self.current_path.pop()
            return True
        elif self.current_fiber.prev_fiber:
            # 回溯到前一个fiber
            self.current_fiber = self.current_fiber.prev_fiber
            self.position = len(self.current_fiber.moves)
            if self.current_path:
                self.current_path.pop()
            return True
        else:
            # 已在根节点，无法回溯
            return False
    
    def get_current_path(self) -> List[Move]:
        """获取当前路径"""
        return self.current_path.copy()
    
    def reset(self):
        """重置到根节点"""
        self.current_fiber = self.tree.root
        self.position = 0
        self.current_path = []


class FBTree:
    """基于Fiber的决策树"""
    
    def __init__(self, attribute_template: Dict[str, Any] = None):
        """初始化决策树"""
        # 设置属性模板
        self.attribute_template = attribute_template or {'visit_count': 0}
        
        # 创建根fiber
        self.root = Fiber(self)
    
    def iterate_fibers(self):
        """遍历所有fiber的生成器"""
        queue = [self.root]
        visited = {self.root}
        
        while queue:
            current = queue.pop(0)
            yield current
            
            # 添加未访问的子节点
            for next_fiber in current.next_fibers.values():
                if next_fiber not in visited:
                    visited.add(next_fiber)
                    queue.append(next_fiber)
    
    def count_fibers(self) -> int:
        """计算树中的fiber数量"""
        return sum(1 for _ in self.iterate_fibers())
    
    def prune(self, condition):
        """根据条件剪枝"""
        self._prune_recursive(self.root, condition)
        
    def _prune_recursive(self, fiber: Fiber, condition):
        """递归剪枝辅助方法"""
        # 检查所有子fiber
        to_remove = [move for move, next_fiber in fiber.next_fibers.items() 
                    if condition(next_fiber)]
        
        # 继续递归处理未删除的子fiber
        for move, next_fiber in fiber.next_fibers.items():
            if move not in to_remove:
                self._prune_recursive(next_fiber, condition)
                
        # 移除符合条件的子fiber
        for move in to_remove:
            del fiber.next_fibers[move]
    
    def create_tracer(self) -> PathTracer:
        """创建路径追踪器"""
        return PathTracer(self)
    
    def find_path(self, moves: List[Union[Sequence, np.ndarray, Move]]) -> Optional[Fiber]:
        """查找路径，返回对应的fiber"""
        if not moves:
            return self.root
        
        # 转换为Move对象
        move_objects = [Move(m) if not isinstance(m, Move) else m for m in moves]
        
        # 尝试查找路径
        current = self.root
        for move in move_objects:
            if move in current.next_fibers:
                current = current.next_fibers[move]
            else:
                return None
        
        return current
    
    def add_attribute(self, name: str, default_value: Any):
        """添加新属性到所有fiber"""
        if name in self.attribute_template:
            raise ValueError(f"属性'{name}'已存在")
        
        # 更新模板
        self.attribute_template[name] = default_value
        
        # 更新所有fiber
        for fiber in self.iterate_fibers():
            fiber.attributes[name] = default_value() if callable(default_value) else default_value
    
    def update_attribute_template(self, name: str, default_value: Any):
        """更新属性模板"""
        if name not in self.attribute_template:
            raise ValueError(f"属性'{name}'不存在")
        
        self.attribute_template[name] = default_value
    
    def propagate_value(self, fiber: Fiber, update_func):
        """沿路径传播值更新"""
        current = fiber
        while current:
            update_func(current)
            current = current.prev_fiber
    
    def visualize(self, max_depth=None, show_attributes=None):
        """可视化决策树"""
        show_attributes = show_attributes or ['visit_count']
        
        # 生成树的可视化文本
        tree_lines = ["FBTree可视化:"]
        tree_lines.extend(self._build_tree_view(self.root, show_attributes, max_depth))
        tree_lines.append(f"\n总节点数: {self.count_fibers()}")
        
        # 打印树
        print("\n".join(tree_lines))
    
    def _build_tree_view(self, fiber: Fiber, attrs, max_depth=None, prefix="", is_last=True, depth=0):
        """生成树的文本表示"""
        if max_depth is not None and depth > max_depth:
            return []
        
        lines = []
        
        # 节点显示内容
        node_text = self._format_node(fiber, attrs)
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + node_text)
        
        # 子节点
        next_prefix = prefix + ("    " if is_last else "│   ")
        next_fibers = list(fiber.next_fibers.items())
        
        for i, (_, next_fiber) in enumerate(next_fibers):
            is_last_child = (i == len(next_fibers) - 1)
            lines.extend(self._build_tree_view(
                next_fiber, attrs, max_depth, next_prefix, is_last_child, depth + 1))
        
        return lines
    
    def _format_node(self, fiber: Fiber, attrs):
        """格式化节点显示内容"""
        node_info = []
        
        # 节点标识
        if fiber == self.root:
            node_info.append("节点=根节点")
        else:
            node_info.append(f"动作={self._format_move(fiber.last_move)}")
        
        # 属性信息
        for attr in attrs:
            if attr in fiber.attributes:
                node_info.append(f"{attr}={fiber.get_attribute(attr)}")
        
        return "[" + ", ".join(node_info) + "]"
    
    def _format_move(self, move):
        """格式化Move显示"""
        if move is None:
            return "根节点"
        
        data = move.data
        if len(data) <= 10:
            return str(data)
        else:
            nonzeros = np.nonzero(data)[0]
            if len(nonzeros) == 1:
                return f"位置{nonzeros[0]}"
            else:
                return f"非零位置{nonzeros}"
    
    def save(self, filepath: str):
        """保存树到文件"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'root': self.root,
                'attribute_template': self.attribute_template
            }, f)
        print(f"树已保存到 {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """从文件加载树"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        tree = cls(data['attribute_template'])
        tree.root = data['root']
        print(f"已从 {filepath} 加载树")
        return tree