import numpy as np
import pandas as pd
from utils import find_closest_point_to_origin, sort_track_points_by_distance

def filter_central_tracks(track_data, max_distance_from_origin=0.1, max_layer_for_origin=2):
    """
    筛选从探测器中央打出的径迹
    
    筛选逻辑（改进版）：
    1. 找到径迹上距离原点最近的点
    2. 计算该点距原点的距离
    3. 如果距离小于阈值(max_distance_from_origin)，且该点所在的层数较小，则认为是从中央打出的径迹
    
    参数:
        track_data (DataFrame): 单条径迹的数据
        max_distance_from_origin (float): 最大允许的起始点偏离距离(m)
        max_layer_for_origin (int): 起始点允许的最大层数
    
    返回:
        bool: 如果是从中央打出的径迹返回True, 否则返回False
    """
    if track_data.empty:
        return False, None, None, float('inf')
    
    # 找到最接近原点的点
    closest_x, closest_y, closest_idx, min_distance = find_closest_point_to_origin(track_data)
    
    if closest_x is None:
        return False, None, None, float('inf')
    
    # 获取最近点所在的层数
    closest_layer = track_data.iloc[closest_idx]['layer'] if 'layer' in track_data.columns else 0
    
    # 判断是否从中央打出（考虑距离和层数）
    is_central = (min_distance <= max_distance_from_origin) and (closest_layer <= max_layer_for_origin)
    
    return is_central, closest_x, closest_y, min_distance

def calculate_closest_distances_for_tracks(track_data_list):
    """
    计算每条径迹离中心最近的hit点的距离
    
    参数:
        track_data_list (list): 包含多条径迹DataFrame的列表
    
    返回:
        list: 每条径迹最近点距离的列表（单位：米）
    """
    closest_distances = []
    
    for track_data in track_data_list:
        if track_data.empty:
            continue
        
        # 找到最接近原点的点
        closest_x, closest_y, closest_idx, min_distance = find_closest_point_to_origin(track_data)
        
        if closest_x is not None:
            closest_distances.append(min_distance)
    
    return closest_distances