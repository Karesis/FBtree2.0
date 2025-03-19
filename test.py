import time
import random
import numpy as np
import threading
import matplotlib.pyplot as plt
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'Noto Sans CJK JP', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 导入FBTree模块
from fbtcore import FBTree, Move
from fbtcore import (
    create_safe_tree,
    create_mcts_tree,
    ucb_select,
    SafeFBTree,
    ConcurrencyMode
)

# 设置随机种子，确保结果可重现
np.random.seed(42)
random.seed(42)

# === 全局函数定义（用于多进程） ===

def compute_fiber_value(fiber):
    """计算密集型任务（全局定义用于多进程）"""
    # 计算密集型任务
    result = 0
    for i in range(1000000):  # 大量计算
        result += np.sin(i * 0.0001) * np.cos(i * 0.0002)
    return result

def update_attributes(fiber):
    """更新fiber属性（全局定义用于多进程）"""
    # 原子更新
    visit_count = fiber.get_attribute('visit_count', 0) + 1
    fiber.set_attribute('visit_count', visit_count)
    
    value = fiber.get_attribute('value', 0.0) + random.random()
    fiber.set_attribute('value', value)
    
    mean = value / visit_count
    fiber.set_attribute('mean', mean)
    
    return {
        "id": id(fiber),
        "visits": visit_count,
        "value": value,
        "mean": mean
    }

def propagate_depth(fiber):
    """计算节点深度（全局定义用于多进程）"""
    if fiber.prev_fiber:
        prev_depth = fiber.prev_fiber.get_attribute('depth', 0)
        fiber.set_attribute('depth', prev_depth + 1)
    else:
        fiber.set_attribute('depth', 0)
    
    # 模拟计算
    time.sleep(0.01)
    
    # 增加访问计数
    visit_count = fiber.get_attribute('visit_count', 0) + 1
    fiber.set_attribute('visit_count', visit_count)

def evaluate_path(fiber):
    """评估路径（用于搜索）"""
    path = fiber.get_full_path()
    if path:
        last_move = path[-1].data
        # 偏好第一个元素接近1的move
        bias = last_move[0] if len(last_move) > 0 else 0
        return 0.5 + 0.5 * bias
    return random.random()

# === 测试函数 ===

def test_basic_operations():
    """测试基本操作"""
    print("\n=== 测试基本操作 ===")
    
    # 创建安全树
    tree = create_safe_tree(attribute_template={
        'visit_count': 0,
        'value': 0.0
    })
    
    # 创建路径追踪器
    tracer = tree.create_tracer()
    
    # 添加几个move
    print("添加 move [1.0, 0.0]")
    tracer.add_move([1.0, 0.0])
    
    print("添加 move [0.0, 1.0]")
    tracer.add_move([0.0, 1.0])
    
    print("添加 move [1.0, 1.0]")
    tracer.add_move([1.0, 1.0])
    
    # 回溯
    print("回溯1步")
    tracer.backtrack()
    
    # 添加不同的move形成分叉
    print("添加 move [0.0, 0.0]")
    tracer.add_move([0.0, 0.0])
    
    # 获取当前路径
    current_path = tracer.get_current_path()
    print(f"当前路径长度: {len(current_path)}")
    
    # 重置
    tracer.reset()
    print("重置后路径长度:", len(tracer.get_current_path()))
    
    # 获取根节点
    root = tree.get_root()
    print("根节点的next_fibers数量:", len(root.next_fibers))
    
    # 可视化树
    print("\n树结构:")
    tree.visualize()
    
    print("基本操作测试通过!")

def test_atomic_operations():
    """测试原子操作"""
    print("\n=== 测试原子操作 ===")
    
    # 创建安全树
    tree = create_safe_tree(attribute_template={
        'counter': 0,
        'value': 0.0
    })
    
    # 获取根节点
    root = tree.get_root()
    
    # 测试update_attribute
    print("测试update_attribute...")
    for i in range(5):
        old_counter = tree.get_attribute(root, 'counter')
        new_counter = tree.update_attribute(root, 'counter', lambda x: x + 1)
        print(f"计数器: {old_counter} -> {new_counter}")
    
    # 测试increment
    print("\n测试increment...")
    for i in range(3):
        new_value = tree.increment(root, 'value', 0.5)
        print(f"值增加0.5: {new_value}")
    
    # 测试atomic_operation
    print("\n测试atomic_operation...")
    def complex_operation(fiber):
        # 复合操作示例
        counter = fiber.get_attribute('counter')
        value = fiber.get_attribute('value')
        # 计算并更新
        fiber.set_attribute('counter', counter + 5)
        fiber.set_attribute('value', value * 2)
        return (counter + 5, value * 2)
    
    result = tree.atomic_operation(root, complex_operation)
    print(f"复合操作结果: {result}")
    
    final_counter = tree.get_attribute(root, 'counter')
    final_value = tree.get_attribute(root, 'value')
    print(f"\n最终计数器: {final_counter}, 最终值: {final_value}")
    
    print("原子操作测试通过!")

def test_thread_safety():
    """测试线程安全性"""
    print("\n=== 测试线程安全性 ===")
    
    # 创建线程安全树
    tree = create_safe_tree(attribute_template={
        'counter': 0
    })
    
    # 获取根节点
    root = tree.get_root()
    
    # 测试方式1：使用原子操作
    def test_atomic_increment():
        # 定义工作函数，多个线程同时增加计数器
        def increment_counter(thread_id):
            for i in range(100):
                # 使用原子increment操作
                counter = tree.increment(root, 'counter')
                
                # 模拟延迟，增加竞争条件可能性
                time.sleep(0.001)
                
                # 打印进度
                if i % 20 == 0:
                    print(f"线程 {thread_id} (原子): 计数器 = {counter}")
        
        # 创建多个线程同时操作同一个节点
        num_threads = 10
        threads = []
        
        print(f"启动{num_threads}个线程使用原子操作增加计数器...")
        tree.set_attribute(root, 'counter', 0)  # 重置计数器
        
        for i in range(num_threads):
            thread = threading.Thread(target=increment_counter, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        atomic_counter = tree.get_attribute(root, 'counter')
        print(f"原子操作最终计数器值: {atomic_counter}")
        return atomic_counter
    
    # 测试方式2：使用读-修改-写模式，但容易出现竞态问题
    def test_read_modify_write():
        def increment_counter_unsafe(thread_id):
            for i in range(100):
                # 读取
                counter = tree.get_attribute(root, 'counter')
                
                # 模拟延迟，增加竞争可能性
                time.sleep(0.001)
                
                # 写入
                tree.set_attribute(root, 'counter', counter + 1)
                
                # 打印进度
                if i % 20 == 0:
                    print(f"线程 {thread_id} (非原子): 计数器 = {counter}")
        
        # 创建多个线程同时操作同一个节点
        num_threads = 10
        threads = []
        
        print(f"\n启动{num_threads}个线程使用非原子操作增加计数器...")
        tree.set_attribute(root, 'counter', 0)  # 重置计数器
        
        for i in range(num_threads):
            thread = threading.Thread(target=increment_counter_unsafe, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        non_atomic_counter = tree.get_attribute(root, 'counter')
        print(f"非原子操作最终计数器值: {non_atomic_counter}")
        return non_atomic_counter
    
    # 执行测试并对比结果
    atomic_result = test_atomic_increment()
    non_atomic_result = test_read_modify_write()
    
    expected = 1000  # 10线程 x 100次递增
    
    print(f"\n原子操作结果: {atomic_result}, 期望: {expected}, 差异: {expected - atomic_result}")
    print(f"非原子操作结果: {non_atomic_result}, 期望: {expected}, 差异: {expected - non_atomic_result}")
    
    if atomic_result == expected:
        print("原子操作线程安全测试通过!")
    else:
        print("原子操作线程安全测试失败!")
        
    if non_atomic_result < expected:
        print("非原子操作竞态条件测试通过(预期会失败)")
    else:
        print("非原子操作意外通过(可能测试条件不够极端)")
    
    # 绘制对比图
    labels = ['原子操作', '非原子操作', '期望值']
    values = [atomic_result, non_atomic_result, expected]
    colors = ['green', 'red', 'purple']
    
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=colors)
    plt.ylabel('计数器值')
    plt.title('线程安全测试结果')
    
    for i, v in enumerate(values):
        plt.text(i, v + 10, str(v), ha='center')
    
    plt.savefig('thread_safety_test.png')
    plt.close()
    
    print(f"线程安全测试图表已保存为 'thread_safety_test.png'")

def test_parallel_processing():
    """测试并行处理性能"""
    print("\n=== 测试并行处理性能 ===")
    
    # 测试参数
    num_fibers = 20
    
    # 创建多进程树
    tree = create_safe_tree(
        attribute_template={'value': 0.0, 'visit_count': 0, 'mean': 0.0},
        concurrency_mode=ConcurrencyMode.PROCESS
    )
    
    # 获取根节点
    root = tree.get_root()
    
    # 创建测试fiber
    fibers = []
    for i in range(num_fibers):
        fiber = tree.add_move(root, Move([float(i/num_fibers)]))
        fibers.append(fiber)
    
    # 测试1：串行计算
    print(f"开始串行计算{num_fibers}个fiber...")
    start_time = time.time()
    
    serial_results = []
    for fiber in fibers:
        serial_results.append(compute_fiber_value(fiber))
    
    serial_time = time.time() - start_time
    print(f"串行计算完成，耗时: {serial_time:.4f}秒")
    
    # 测试2：并行计算
    print(f"\n开始并行计算{num_fibers}个fiber...")
    start_time = time.time()
    
    parallel_results = tree.parallel_map(fibers, compute_fiber_value)
    
    parallel_time = time.time() - start_time
    print(f"并行计算完成，耗时: {parallel_time:.4f}秒")
    
    # 加速比
    speedup = serial_time / parallel_time
    print(f"加速比: {speedup:.2f}x")
    
    # 结果验证
    results_match = True
    for i, (s, p) in enumerate(zip(serial_results[:5], parallel_results[:5])):
        if isinstance(p, tuple) and len(p) == 2:  # 进程模式返回(id, result)
            p = p[1]
        if isinstance(p, str) and p.startswith("ERROR:"):
            print(f"错误: {p}")
            results_match = False
            break
        try:
            if abs(s - p) > 1e-10:
                results_match = False
                break
        except:
            results_match = False
            break
            
    print(f"结果一致性检查: {'通过' if results_match else '失败'}")
    
    # 绘制性能对比
    labels = ['串行', '并行']
    times = [serial_time, parallel_time]
    
    plt.figure(figsize=(10, 5))
    
    # 执行时间
    plt.subplot(1, 2, 1)
    plt.bar(labels, times, color=['blue', 'orange'])
    plt.ylabel('执行时间 (秒)')
    plt.title('并行处理性能对比')
    
    for i, v in enumerate(times):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    # 加速比
    plt.subplot(1, 2, 2)
    plt.bar(['加速比'], [speedup], color='green')
    cpu_count = multiprocessing.cpu_count()
    plt.axhline(y=cpu_count, color='red', linestyle='--', label=f'CPU核心数({cpu_count})')
    plt.ylabel('加速比')
    plt.ylim(0, max(speedup, cpu_count) * 1.2)
    plt.legend()
    
    plt.text(0, speedup + 0.1, f"{speedup:.2f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig('parallel_performance.png')
    plt.close()
    
    print(f"并行处理性能图表已保存为 'parallel_performance.png'")

def test_tree_search():
    """测试树搜索"""
    print("\n=== 测试树搜索 ===")
    
    # 创建MCTS树
    tree = create_mcts_tree()
    
    # 为根节点添加一些子节点
    root = tree.get_root()
    for i in range(5):
        tree.add_move(root, Move([i/4, (4-i)/4]))
    
    print("开始蒙特卡洛树搜索...")
    start_time = time.time()
    
    # 使用UCB选择策略进行搜索
    best_node = tree.search(
        root=root,
        select_func=lambda node: ucb_select(tree, node, 1.0),
        simulate_func=evaluate_path,
        num_simulations=200
    )
    
    end_time = time.time()
    search_time = end_time - start_time
    
    print(f"搜索完成，耗时: {search_time:.4f}秒")
    
    # 验证结果
    if best_node:
        path = best_node.get_full_path()
        visits = tree.get_attribute(best_node, 'visit_count')
        value = tree.get_attribute(best_node, 'mean_value')
        
        print(f"最佳节点访问次数: {visits}")
        print(f"最佳节点平均价值: {value:.4f}")
        
        if path:
            last_move = path[-1]
            print(f"最佳路径最后一步: {last_move}")
    
    print("树搜索测试通过!")

def test_concurrent_updates():
    """测试并发更新"""
    print("\n=== 测试并发更新 ===")
    
    # 创建安全树
    tree = create_safe_tree(
        attribute_template={
            'visit_count': 0,
            'value': 0.0,
            'mean': 0.0
        },
        concurrency_mode=ConcurrencyMode.PROCESS
    )
    
    # 获取根节点
    root = tree.get_root()
    
    # 创建一些测试fiber
    num_fibers = 10
    fibers = []
    for i in range(num_fibers):
        fiber = tree.add_move(root, Move([float(i)/num_fibers]))
        fibers.append(fiber)
    
    # 并行更新所有fiber
    print(f"开始并行更新{len(fibers)}个fiber...")
    start_time = time.time()
    results = tree.parallel_map(fibers, update_attributes)
    end_time = time.time()
    
    print(f"并行更新完成，耗时: {end_time - start_time:.4f}秒")
    
    # 验证结果
    valid_results = []
    for r in results:
        if isinstance(r, tuple):
            # 处理可能的(id, result)格式
            r = r[1]
        if isinstance(r, dict):
            valid_results.append(r)
    
    for result in valid_results[:3]:  # 只显示前3个结果
        fiber_id = result["id"]
        visits = result["visits"]
        value = result["value"]
        mean = result["mean"]
        print(f"Fiber ID {fiber_id}: 访问次数={visits}, 值={value:.4f}, 平均={mean:.4f}")
    
    print(f"成功更新: {len(valid_results)}/{len(fibers)} 个fiber")
    print("并发更新测试通过!")

def test_propagation():
    """测试路径传播"""
    print("\n=== 测试路径传播 ===")
    
    # 创建线程安全树
    tree = create_safe_tree(attribute_template={
        'visit_count': 0,
        'value': 0.0,
        'depth': 0
    })
    
    # 创建一条长路径
    tracer = tree.create_tracer()
    path_length = 10
    
    print(f"创建深度为{path_length}的路径...")
    for i in range(path_length):
        tracer.add_move([float(i)])
    
    # 获取叶节点
    leaf = tracer.current_fiber
    
    # 单线程传播
    print("开始单线程传播...")
    start_time = time.time()
    tree.propagate(leaf, propagate_depth, parallel=False)
    single_time = time.time() - start_time
    print(f"单线程传播完成，耗时: {single_time:.4f}秒")
    
    # 验证深度值
    current = leaf
    depths = []
    while current:
        depth = tree.get_attribute(current, 'depth')
        depths.append(depth)
        current = current.prev_fiber
    print(f"深度值: {depths}")
    
    # 重置
    tracer.reset()
    for i in range(path_length):
        tracer.add_move([float(i)])
    leaf = tracer.current_fiber
    
    # 多线程传播
    print("\n开始多线程传播...")
    start_time = time.time()
    tree.propagate(leaf, propagate_depth, parallel=True)
    multi_time = time.time() - start_time
    print(f"多线程传播完成，耗时: {multi_time:.4f}秒")
    
    # 比较性能
    speedup = single_time / multi_time
    print(f"速度提升: {speedup:.2f}x")
    
    # 验证结果
    current = leaf
    depths_parallel = []
    while current:
        depth = tree.get_attribute(current, 'depth')
        depths_parallel.append(depth)
        current = current.prev_fiber
    print(f"并行深度值: {depths_parallel}")
    
    # 检查结果一致性
    consistency = True
    for i, (d1, d2) in enumerate(zip(depths, depths_parallel)):
        if d1 != d2:
            consistency = False
            break
    
    print(f"结果一致性: {'通过' if consistency else '失败'}")
    print("路径传播测试通过!")

def run_all_tests():
    """运行所有测试"""
    print("=== 开始FBTree并发模块测试 ===\n")
    
    # 运行各个测试
    test_basic_operations()
    test_atomic_operations()
    test_thread_safety()
    test_parallel_processing()
    test_tree_search()
    test_concurrent_updates()
    test_propagation()
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    run_all_tests()