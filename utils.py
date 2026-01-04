import numpy as np
import pandas as pd

# Define a set of distinct colors with high visibility and contrast
distinct_colors = [
    '#FF0000',   # Bright Red
    '#00FF00',   # Bright Green
    '#0000FF',   # Bright Blue
    '#FFA500',   # Orange
    '#FF00FF',   # Magenta
    '#00FFFF',   # Cyan
    '#FFFF00',   # Yellow
    '#FF4500',   # Orange Red
    '#32CD32',   # Lime Green
    '#1E90FF',   # Dodger Blue
    '#FF1493',   # Deep Pink
    '#00CED1',   # Dark Turquoise
    '#FFD700',   # Gold
    '#8A2BE2',   # Blue Violet
    '#DC143C'    # Crimson
]

def calculate_track_parameters(track_data):
    """Calculate the center and radius of the track using the first point's data"""
    if track_data.empty:
        return None, None, None, None, None
    
    # 使用径迹的第一个点来计算参数
    first_point = track_data.iloc[0]
    
    # 检查必要的字段是否存在
    if 'momX' not in first_point or 'momY' not in first_point or 'charge' not in first_point:
        return None, None, None, None, None
    
    # 计算横向动量 (pt)
    pt = np.sqrt(first_point['momX']**2 + first_point['momY']** 2)
    # 计算曲率半径 (转换为米)
    r = pt * 333.564 / 100  # Convert to meters
    # 确定电荷符号
    sign = 1 if first_point['charge'] >= 0.5 else -1
    # 计算动量方向角度
    angle = np.arctan2(first_point['momY'], first_point['momX'])
    # 计算圆心坐标
    cx = first_point['posX'] / 100 - r * np.cos(angle - sign * np.pi/2)
    cy = first_point['posY'] / 100 - r * np.sin(angle - sign * np.pi/2)
    
    return pt, r, sign, cx, cy

def calculate_momentum_direction(momX, momY):
    """Calculate the momentum direction angle in radians"""
    return np.arctan2(momY, momX)

def angle_difference(angle1, angle2):
    """Calculate the minimal difference between two angles (in radians), always between 0 and π"""
    diff = abs(angle1 - angle2)
    # Normalize to [0, π]
    diff = min(diff, 2*np.pi - diff)
    return diff

def calculate_distance_from_origin(x, y):
    """计算点到原点(0,0)的距离"""
    return np.sqrt(x**2 + y**2)

def find_closest_point_to_origin(track_data):
    """找到径迹中最接近原点的点"""
    if track_data.empty:
        return None, None, None, float('inf')
    
    # 转换坐标为米
    x_coords = track_data['posX'].values / 100
    y_coords = track_data['posY'].values / 100
    
    # 计算每个点到原点的距离
    distances = np.sqrt(x_coords**2 + y_coords**2)
    
    # 找到最小距离及其索引
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    
    return x_coords[min_idx], y_coords[min_idx], min_idx, min_distance

def sort_track_points_by_distance(track_data):
    """
    将径迹点按距离原点从小到大排序
    
    参数:
        track_data (DataFrame): 单条径迹的数据
    
    返回:
        DataFrame: 排序后的径迹数据
    """
    if track_data.empty:
        return track_data
    
    # 计算每个点到原点的距离
    distances = np.sqrt((track_data['posX']/100)**2 + (track_data['posY']/100)**2)
    
    # 添加距离列
    track_data_sorted = track_data.copy()
    track_data_sorted['distance_to_origin'] = distances
    
    # 按距离排序
    track_data_sorted = track_data_sorted.sort_values('distance_to_origin')
    
    return track_data_sorted

def identify_good_tracks(track_data):
    """识别好track：满足hit>6且layer>6条件"""
    if track_data.empty:
        return False
    
    hit_count = len(track_data)
    unique_layers = track_data['layer'].nunique()
    
    return hit_count > 6 and unique_layers > 6

def get_ionization_process_list():
    """返回固定的电离process列表"""
    return ["eIoni", "hIoni", "muIoni"]

def calculate_hit_geometry(hit_data):
    """计算hit点的几何参数：圆心为o，半径为r
    对于斜丝(sLayer in [0,1,5,6,7,8])，尝试使用posZ计算更精确的着火点
    """
    if hit_data.empty:
        return None, None
    
    row = hit_data.iloc[0]
    
    # 默认使用 middleX/middleY
    o_x = row['middleX'] / 100
    o_y = row['middleY'] / 100
    
    # 检查是否为斜丝并尝试计算精确位置
    stereo_layers = [0, 1, 5, 6, 7, 8]
    required_cols = ['sLayer', 'posZ', 'eastX', 'eastY', 'eastZ', 'westX', 'westY', 'westZ']
    
    if all(col in row for col in required_cols):
        if row['sLayer'] in stereo_layers:
            try:
                z_diff = row['westZ'] - row['eastZ']
                if abs(z_diff) > 1e-6:
                    t = (row['posZ'] - row['eastZ']) / z_diff
                    o_x = (row['eastX'] + t * (row['westX'] - row['eastX'])) / 100
                    o_y = (row['eastY'] + t * (row['westY'] - row['eastY'])) / 100
            except Exception:
                pass # Fallback to middleX/middleY
    
    # 半径（rawDriftDist）转换为米
    r = row['rawDriftDist'] / 100
    
    return (o_x, o_y), r


def create_angle_difference_histogram(data, title="Angle Differences Distribution", 
                                    xlabel="Angle Difference (degrees)", ylabel="Count",
                                    threshold=30, bins=30, figsize=(10, 8), 
                                    color='lightcoral', threshold_color='blue',
                                    show_percentage=True, show_stats=True):
    """
    创建角度差分布直方图的可复用函数
    
    参数:
        data (list): 角度差数据列表（单位：度）
        title (str): 图表标题
        xlabel (str): X轴标签
        ylabel (str): Y轴标签
        threshold (float): 阈值线位置
        bins (int): 直方图的分箱数
        figsize (tuple): 图形尺寸
        color (str): 直方图颜色
        threshold_color (str): 阈值线颜色
        show_percentage (bool): 是否显示百分比曲线
        show_stats (bool): 是否显示统计信息
    
    返回:
        tuple: (fig, ax) matplotlib图形和坐标轴对象
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not data:
        print("Warning: No data provided for histogram.")
        return None, None
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制直方图
    counts, bins, patches = ax.hist(data, bins=bins, 
                                   edgecolor='black', alpha=0.7, color=color)
    
    # 添加阈值线
    threshold_count = sum(1 for d in data if d <= threshold)
    ax.axvline(x=threshold, color=threshold_color, linestyle='--', linewidth=2, 
               label=f'{threshold}° threshold ({threshold_count} points)')
    
    # 设置坐标轴标签和标题
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # 添加百分比曲线（如果需要）
    if show_percentage:
        percentages = counts / len(data) * 100
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax_percent = ax.twinx()
        ax_percent.plot(bin_centers, percentages, 'r-', linewidth=2, marker='o', 
                       markersize=4, label='Percentage')
        ax_percent.set_ylabel('Percentage (%)', fontsize=14, color='red')
        ax_percent.tick_params(axis='y', labelcolor='red')
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_percent.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # 添加统计信息（如果需要）
    if show_stats:
        filtered_data = [d for d in data if d <= threshold]
        stats_text = (f'Statistics:\n'
                     f'Total points: {len(data)}\n'
                     f'Mean: {np.mean(data):.2f}°\n'
                     f'Median: {np.median(data):.2f}°\n'
                     f'Std: {np.std(data):.2f}°\n'
                     f'Min: {np.min(data):.2f}°\n'
                     f'Max: {np.max(data):.2f}°\n'
                     f'Points ≤{threshold}°: {len(filtered_data)} '
                     f'({len(filtered_data)/len(data)*100:.1f}%)')
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax


def create_detailed_histogram(data, title="Detailed Distribution", 
                             xlabel="Angle Difference (degrees)", ylabel="Count",
                             threshold=30, bins=20, figsize=(8, 6), 
                             color='lightgreen', threshold_color='blue',
                             show_stats=True):
    """
    创建详细的分布直方图（通常用于阈值范围内的数据）
    
    参数:
        data (list): 数据列表
        title (str): 图表标题
        xlabel (str): X轴标签
        ylabel (str): Y轴标签
        threshold (float): 阈值
        bins (int): 直方图的分箱数
        figsize (tuple): 图形尺寸
        color (str): 直方图颜色
        threshold_color (str): 阈值线颜色
        show_stats (bool): 是否显示统计信息
    
    返回:
        tuple: (fig, ax) matplotlib图形和坐标轴对象
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not data:
        print("Warning: No data provided for detailed histogram.")
        return None, None
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制直方图
    ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, color=color)
    
    # 添加阈值线
    ax.axvline(x=threshold, color=threshold_color, linestyle='--', linewidth=2, 
               label=f'{threshold}° threshold ({len(data)} points)')
    
    # 设置坐标轴标签和标题
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # 添加统计信息（如果需要）
    if show_stats:
        stats_text = (f'Statistics:\n'
                     f'Mean: {np.mean(data):.2f}°\n'
                     f'Std: {np.std(data):.2f}°\n'
                     f'Median: {np.median(data):.2f}°\n'
                     f'Min: {np.min(data):.2f}°\n'
                     f'Max: {np.max(data):.2f}°')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax

def calculate_distance_to_track(hit_center, hit_radius, track_center, track_radius):
    """计算hit点到track曲线的距离：d = R - |O - o|"""
    if None in [hit_center, hit_radius, track_center, track_radius]:
        return None
    
    # 计算hit点圆心到track圆心的欧氏距离
    o_x, o_y = hit_center
    O_x, O_y = track_center
    
    distance_centers = np.sqrt((O_x - o_x)**2 + (O_y - o_y)**2)
    
    # 计算距离参数d，加上绝对值
    d = abs(track_radius - distance_centers - hit_radius)
    
    return d
