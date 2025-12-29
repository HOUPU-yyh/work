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
    """计算hit点的几何参数：圆心为o，半径为r"""
    if hit_data.empty:
        return None, None
    
    # 圆心坐标（middleX, middleY）转换为米
    o_x = hit_data['middleX'].iloc[0] / 100
    o_y = hit_data['middleY'].iloc[0] / 100
    
    # 半径（rawDriftDist）转换为米
    r = hit_data['rawDriftDist'].iloc[0] / 100
    
    return (o_x, o_y), r

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
