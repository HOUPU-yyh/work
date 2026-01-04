import numpy as np
import pandas as pd
from utils import find_closest_point_to_origin, sort_track_points_by_distance

def filter_central_tracks(track_data):
    """
    筛选径迹（不再进行距离和层数限制）
    
    参数:
        track_data (DataFrame): 单条径迹的数据
    
    返回:
        bool: 总是返回True，表示接受所有径迹
    """
    if track_data.empty:
        return False, None, None, float('inf')
    
    # 找到最接近原点的点
    closest_x, closest_y, closest_idx, min_distance = find_closest_point_to_origin(track_data)
    
    if closest_x is None:
        return False, None, None, float('inf')
    
    # 不再进行任何筛选，接受所有径迹
    is_central = True
    
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