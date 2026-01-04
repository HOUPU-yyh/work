import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import sort_track_points_by_distance, angle_difference
from track_filter import calculate_closest_distances_for_tracks

def calculate_hit_relationships(track_data_list):
    """
    计算径迹中每个hit与其最近hit之间的关系
    参数:
        track_data_list (list): 包含多条径迹DataFrame的列表
    返回:
        tuple: (角度分布列表, 动量向量角度差分布列表, 距离分布列表)
    """
    angle_distributions = []
    momentum_angle_differences = []
    distance_distributions = []
    
    for track_data in track_data_list:
        if track_data.empty or len(track_data) < 2:
            continue
        
        # 按距离排序，确保从近到远处理
        track_data_sorted = sort_track_points_by_distance(track_data)
        
        # 计算每个hit与其最近hit的关系
        for i in range(1, len(track_data_sorted)):
            current_hit = track_data_sorted.iloc[i]
            previous_hit = track_data_sorted.iloc[i-1]
            
            # 计算位置向量
            pos_vector = np.array([current_hit['posX'] - previous_hit['posX'], 
                                 current_hit['posY'] - previous_hit['posY']])
            
            # 计算当前hit和最近hit的动量向量
            current_mom_vector = np.array([current_hit['momX'], current_hit['momY']])
            previous_mom_vector = np.array([previous_hit['momX'], previous_hit['momY']])
            
            # 1. 计算位置向量角度差（hit点和离他最近的hit点的位置向量之间角度差）
            if np.linalg.norm(pos_vector) > 0:
                # 计算当前hit和最近hit的位置向量
                current_pos_vector = np.array([current_hit['posX'], current_hit['posY']])
                previous_pos_vector = np.array([previous_hit['posX'], previous_hit['posY']])
                
                # 归一化位置向量
                current_pos_norm = current_pos_vector / np.linalg.norm(current_pos_vector)
                previous_pos_norm = previous_pos_vector / np.linalg.norm(previous_pos_vector)
                
                # 计算位置向量之间的夹角（弧度转角度）
                cos_pos_angle = np.dot(current_pos_norm, previous_pos_norm)
                pos_angle = np.arccos(np.clip(cos_pos_angle, -1.0, 1.0)) * 180 / np.pi
                angle_distributions.append(pos_angle)
            
            # 2. 计算动量向量角度差（hit点和离他最近的hit点的动量向量之间角度差）
            if np.linalg.norm(current_mom_vector) > 0 and np.linalg.norm(previous_mom_vector) > 0:
                # 归一化动量向量
                current_mom_norm = current_mom_vector / np.linalg.norm(current_mom_vector)
                previous_mom_norm = previous_mom_vector / np.linalg.norm(previous_mom_vector)
                
                # 计算动量向量之间的夹角（弧度转角度）
                cos_mom_angle = np.dot(current_mom_norm, previous_mom_norm)
                mom_angle = np.arccos(np.clip(cos_mom_angle, -1.0, 1.0)) * 180 / np.pi
                momentum_angle_differences.append(mom_angle)
            
            # 3. 计算距离分布（hit间的距离）
            distance = np.linalg.norm(pos_vector)
            distance_distributions.append(distance)
    
    return angle_distributions, momentum_angle_differences, distance_distributions

def analyze_hit_relationships(args, process_list=None):
    """
    分析径迹中hit间的关系(角度、位置动量、距离分布)
    
    参数:
        args: 命令行参数
        process_list: 要处理的process列表
    """
    if process_list is None:
        process_list = []
    
    # 读取原始数据
    df_rawData = pd.read_csv(args.input1, index_col=False)
    
    # 检查数据中是否有process列
    if 'process' not in df_rawData.columns:
        print("Warning: 'process' column not found in data. Using default process name.")
        df_rawData['process'] = 'default'
    
    # 按process分组
    process_groups = df_rawData.groupby('process')
    
    # 如果指定了process_list，只处理指定的process
    if process_list:
        # 检查process_list中哪些process实际存在于数据中
        available_processes = set(process_groups.keys())
        requested_processes = set(process_list)
        
        # 找到实际存在的process
        valid_processes = available_processes.intersection(requested_processes)
        # 找到不存在的process
        missing_processes = requested_processes - available_processes
        
        if missing_processes:
            print(f"Warning: The following processes were requested but not found in data: {', '.join(sorted(missing_processes))}")
        
        process_groups = {name: group for name, group in process_groups if name in valid_processes}
        print(f"Found {len(process_groups)} processes to analyze (from {len(valid_processes)} valid processes in request)")
    else:
        process_groups = {name: group for name, group in process_groups}
        print(f"Found {len(process_groups)} processes to analyze")
    print(f"Track selection criteria: hits ≥ 6 and layers ≥ 6")
    
    # 初始化按process分类的数据字典
    angle_by_process = {}
    mom_angle_by_process = {}
    distance_by_process = {}
    
    # 处理每个process
    for process_name, process_data in process_groups.items():
        print(f"\nProcessing process: {process_name}")
        
        # 按run和event分组
        grouped_data = process_data.groupby(['run', 'event'])
        
        # 检查并调整最大事件数
        max_possible_events = len(grouped_data)
        if args.events > max_possible_events:
            print(f"  Reached maximum event limit for {process_name}: {max_possible_events}")
            events_to_process = max_possible_events
        else:
            events_to_process = args.events
        
        # 初始化当前process的数据列表
        process_angles = []
        process_mom_angles = []
        process_distances = []
        
        # 遍历当前process的每个事件
        for event_num, ((run_id, event_id), event_data) in enumerate(grouped_data):
            if event_num >= events_to_process:
                break
            
            # 获取所有非零的径迹（trackIndex != 0）
            signal_data = event_data[event_data['trackIndex'] != 0]
            valid_tracks_data = []
            
            # 获取所有非零的径迹索引
            track_indices = sorted(signal_data['trackIndex'].unique())
            
            for track_idx in track_indices:
                # 获取当前径迹的数据
                track_data = signal_data[signal_data['trackIndex'] == track_idx]
                
                # 直接添加到有效径迹列表中，不进行hit数和层数筛选
                valid_tracks_data.append(track_data)
            
            # 计算当前事件中所有径迹的hit间关系
            event_angles, event_mom_angles, event_distances = calculate_hit_relationships(valid_tracks_data)
            process_angles.extend(event_angles)
            process_mom_angles.extend(event_mom_angles)
            process_distances.extend(event_distances)
            
            if (event_num + 1) % 10 == 0:  # 每10个事件打印一次进度
                print(f"  Processed {event_num + 1}/{events_to_process} events, collected {len(event_angles)} hit pairs")
        
        # 存储当前process的数据
        angle_by_process[process_name] = process_angles
        mom_angle_by_process[process_name] = process_mom_angles
        distance_by_process[process_name] = process_distances
        print(f"  Process {process_name}: processed {events_to_process} events, total {len(process_angles)} hit pairs")
    
    # 使用用户设置的输出文件名
    relationship_pdf_path = args.output
    
    # 分离Generator过程和其他process的数据
    generator_angles = []
    generator_mom_angles = []
    generator_distances = []
    
    other_angles = []
    other_mom_angles = []
    other_distances = []
    
    for process_name in angle_by_process:
        if 'generator' in process_name.lower():
            generator_angles.extend(angle_by_process[process_name])
            generator_mom_angles.extend(mom_angle_by_process[process_name])
            generator_distances.extend(distance_by_process[process_name])
        else:
            other_angles.extend(angle_by_process[process_name])
            other_mom_angles.extend(mom_angle_by_process[process_name])
            other_distances.extend(distance_by_process[process_name])
    
    # 分别绘制Generator过程和其他process的图表
    if generator_angles or generator_mom_angles or generator_distances:
        generator_pdf_path = args.output.replace('.pdf', '_generator.pdf')
        plot_hit_relationship_distributions(generator_angles, generator_mom_angles, generator_distances, 
                                           generator_pdf_path, "(Generator Processes)")
    
    if other_angles or other_mom_angles or other_distances:
        other_pdf_path = args.output.replace('.pdf', '_other.pdf')
        plot_hit_relationship_distributions(other_angles, other_mom_angles, other_distances, 
                                           other_pdf_path, "(Other Processes)")
    
    # 合并所有process的数据用于总体分析
    all_angles = generator_angles + other_angles
    all_mom_angles = generator_mom_angles + other_mom_angles
    all_distances = generator_distances + other_distances
    
    if all_angles or all_mom_angles or all_distances:
        plot_hit_relationship_distributions(all_angles, all_mom_angles, all_distances, 
                                           relationship_pdf_path, "(All Processes)")
    
    print(f"\nHit relationship analysis complete.")
    
    # 打印总体统计信息
    total_hit_pairs = len(all_angles)
    print(f"Total hit pairs analyzed: {total_hit_pairs}")
    print(f"Processes analyzed: {len(angle_by_process)}")
    
    # 打印每个process组的统计摘要
    print("\nProcess group summary:")
    if generator_angles:
        angles_array = np.array(generator_angles)
        print(f"  Generator processes: {len(generator_angles)} hit pairs, mean angle: {np.mean(angles_array):.1f}°")
    if other_angles:
        angles_array = np.array(other_angles)
        print(f"  Other processes: {len(other_angles)} hit pairs, mean angle: {np.mean(angles_array):.1f}°")
    
    # 打印每个process的统计摘要
    print("\nIndividual process summary:")
    for process_name in angle_by_process:
        angles = angle_by_process[process_name]
        if angles:
            angles_array = np.array(angles)
            print(f"  {process_name}: {len(angles)} hit pairs, mean angle: {np.mean(angles_array):.1f}°")

def analyze_closest_distance_distribution(args, process_list=None):
    """
    分析每条径迹离中心最近的hit点的距离分布
    
    参数:
        args: 命令行参数
        process_list: 要处理的process列表
    """
    if process_list is None:
        process_list = []
    
    # 读取原始数据
    df_rawData = pd.read_csv(args.input1, index_col=False)
    
    # 检查数据中是否有process列
    if 'process' not in df_rawData.columns:
        print("Warning: 'process' column not found in data. Using default process name.")
        df_rawData['process'] = 'default'
    
    # 按process分组
    process_groups = df_rawData.groupby('process')
    
    # 如果指定了process_list，只处理指定的process
    if process_list:
        process_groups = {name: group for name, group in process_groups if name in process_list}
    else:
        process_groups = {name: group for name, group in process_groups}
    
    print(f"Found {len(process_groups)} processes to analyze")
    print(f"Track selection criteria: hits ≥ 6 and layers ≥ 6")
    
    # 初始化按process分类的距离字典
    distance_by_process = {}
    
    # 处理每个process
    for process_name, process_data in process_groups.items():
        print(f"\nProcessing process: {process_name}")
        
        # 按run和event分组
        grouped_data = process_data.groupby(['run', 'event'])
        
        # 检查并调整最大事件数
        max_possible_events = len(grouped_data)
        if args.events > max_possible_events:
            print(f"  Reached maximum event limit for {process_name}: {max_possible_events}")
            events_to_process = max_possible_events
        else:
            events_to_process = args.events
        
        # 初始化当前process的距离列表
        process_distances = []
        
        # 遍历当前process的每个事件
        for event_num, ((run_id, event_id), event_data) in enumerate(grouped_data):
            if event_num >= events_to_process:
                break
            
            # 获取所有非零的径迹（trackIndex != 0）
            signal_data = event_data[event_data['trackIndex'] != 0]
            valid_tracks_data = []
            
            # 获取所有非零的径迹索引
            track_indices = sorted(signal_data['trackIndex'].unique())
            
            for track_idx in track_indices:
                # 获取当前径迹的数据
                track_data = signal_data[signal_data['trackIndex'] == track_idx]
                
                # 直接添加到有效径迹列表中，不进行hit数和层数筛选
                valid_tracks_data.append(track_data)
            
            # 计算当前事件中所有径迹的最近点距离
            event_distances = calculate_closest_distances_for_tracks(valid_tracks_data)
            process_distances.extend(event_distances)
            
            if (event_num + 1) % 10 == 0:  # 每10个事件打印一次进度
                print(f"  Processed {event_num + 1}/{events_to_process} events, collected {len(event_distances)} distances")
        
        # 存储当前process的距离数据
        distance_by_process[process_name] = process_distances
        print(f"  Process {process_name}: processed {events_to_process} events, total {len(process_distances)} tracks")
    
    # 使用用户设置的输出文件名
    distance_pdf_path = args.output
    
    # 绘制按process分类的距离分布图
    plot_closest_distance_distribution_by_process(distance_by_process, distance_pdf_path)
    
    print(f"\nDistance analysis complete. Plot saved to: {distance_pdf_path}")
    
    # 打印总体统计信息
    total_tracks = sum(len(distances) for distances in distance_by_process.values())
    print(f"Total tracks analyzed: {total_tracks}")
    print(f"Processes analyzed: {len(distance_by_process)}")
    
    # 打印每个process的统计摘要
    print("\nProcess summary:")
    for process_name, distances in distance_by_process.items():
        if distances:
            distances_array = np.array(distances)
            print(f"  {process_name}: {len(distances)} tracks, mean distance: {np.mean(distances_array):.3f} m")