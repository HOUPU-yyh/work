BESIII Track Analysis - Modular System
=======================================

项目概述
--------
本项目是对BESIII径迹分析代码的模块化重构，将原有的单一文件`clean_dataset.py`拆分为多个功能模块，提高代码的可维护性和可扩展性。

模块结构
--------

1. **main.py** - 主入口文件
   - 整合所有模块功能
   - 提供统一的命令行接口
   - 包含函数列表功能

2. **utils.py** - 基础工具函数模块
   - `calculate_track_parameters()` - 计算径迹参数（中心、半径等）
   - `calculate_momentum_direction()` - 计算动量方向角度
   - `angle_difference()` - 计算角度差异
   - `calculate_distance_from_origin()` - 计算点到原点距离
   - `find_closest_point_to_origin()` - 找到最接近原点的点
   - `sort_track_points_by_distance()` - 按距离排序径迹点

3. **track_filter.py** - 径迹筛选函数模块
   - `filter_central_tracks()` - 筛选中央径迹
   - `calculate_closest_distances_for_tracks()` - 计算径迹间最近距离

4. **analysis.py** - 数据分析函数模块
   - `calculate_hit_relationships()` - 计算击中点关系
   - `analyze_hit_relationships()` - 分析击中点关系分布
   - `analyze_closest_distance_distribution()` - 分析最近距离分布

5. **visualization.py** - 可视化函数模块
   - `visualize_raw_tracks()` - 可视化原始径迹数据
   - `visualize_rec_tracks()` - 可视化重建径迹
   - `plot_closest_distance_distribution_by_process()` - 按过程绘制最近距离分布
   - `plot_hit_relationship_distributions()` - 绘制击中点关系分布

使用说明
--------

### 基本用法
```bash
# 查看所有可用函数
python main.py --list-functions

# 可视化原始径迹数据
python main.py --run-mode raw --input1 raw_data.csv --output display.pdf --events 10

# 可视化重建径迹
python main.py --run-mode rec --input1 raw_data.csv --input2 pred_hits.csv --output display.pdf --events 10

# 分析最近距离分布
python main.py --run-mode distance --input1 raw_data.csv --output distance_analysis.pdf --events 10

# 分析击中点关系
python main.py --run-mode hit-relationships --input1 raw_data.csv --output hit_analysis.pdf --events 10
```

### 高级选项
```bash
# 指定过程列表
python main.py --run-mode raw --input1 raw_data.csv --processes eIoni muIoni hIoni

# 设置筛选参数
python main.py --run-mode raw --input1 raw_data.csv --max-distance 0.1 --max-layer 2

# 显示径迹曲线和起始点标记
python main.py --run-mode raw --input1 raw_data.csv --show-track-curve --show-start-triangle
```

命令行参数详解
--------------

- `--run-mode`: 运行模式（必选）
  - `raw`: 可视化原始径迹数据
  - `rec`: 可视化重建径迹
  - `distance`: 分析最近距离分布
  - `hit-relationships`: 分析击中点关系

- `--input1`: 输入原始数据CSV文件（raw模式必需）
- `--input2`: 输入预测击中点CSV文件（rec模式必需）
- `--output`: 输出PDF文件路径（默认：display.pdf）
- `--events`: 最大事件数（默认：10）

- `--processes`: 过程列表（用于径迹重新分类）
- `--whitelist-processes`: 白名单过程（默认：eIoni, muIoni, hIoni）
- `--max-distance`: 中央径迹最大距离阈值（默认：80cm）
- `--max-layer`: 起始点最大层数（默认：80）
- `--show-start-triangle`: 显示起始点三角形标记
- `--show-track-curve`: 显示径迹曲线

模块依赖关系
------------

```
main.py
├── utils.py (基础工具)
├── track_filter.py (径迹筛选)
├── analysis.py (数据分析)
└── visualization.py (可视化)
```

- `visualization.py` 依赖 `utils.py` 和 `track_filter.py`
- `analysis.py` 依赖 `utils.py`
- 所有模块都可以独立使用

重构优势
--------

1. **模块化设计**: 功能分离，便于维护和扩展
2. **代码复用**: 通用函数可在不同模块间共享
3. **清晰结构**: 按功能分类，逻辑清晰
4. **易于测试**: 每个模块可独立测试
5. **文档完善**: 每个函数都有详细注释

文件对比
--------

| 原文件 | 新模块 | 功能描述 |
|--------|--------|----------|
| clean_dataset.py | main.py | 主入口和参数解析 |
| clean_dataset.py | utils.py | 基础计算工具 |
| clean_dataset.py | track_filter.py | 径迹筛选逻辑 |
| clean_dataset.py | analysis.py | 数据分析功能 |
| clean_dataset.py | visualization.py | 可视化功能 |

注意事项
--------

1. 确保所有输入CSV文件格式正确
2. 输出PDF文件路径需要可写权限
3. 对于大数据集，适当调整`--events`参数
4. 可视化功能需要matplotlib和pandas库支持

版本信息
--------
- 重构版本: 1.0.0
- 原文件: clean_dataset.py
- 重构日期: 2025-12-26

作者
----
BESIII Track Analysis Team

许可证
------
MIT License