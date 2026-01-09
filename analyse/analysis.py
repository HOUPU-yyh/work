import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import (
    sort_track_points_by_distance,
    angle_difference,
    calculate_turn_id,
    identify_good_tracks,
    get_ionization_process_list,
    calculate_track_parameters,
    calculate_hit_geometry,
    calculate_distance_to_track,
    compute_good_track_params,
    compute_hit_to_track_distance,
)
from track_filter import calculate_closest_distances_for_tracks
from visualization import (
    plot_closest_distance_distribution_by_process,
    plot_hit_relationship_distributions,
    analyze_distance_to_tracks_distribution,
)

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


def analyze_ionization_distance_distribution(args, process_list=None, distance_threshold=0.01, bins=60):
    """
    分析 Step 2 中电离 hit 到最近好径迹的距离分布，帮助选择阈值。

    参数:
        args: 命令行参数，需包含 input1, output, events
        process_list: 可选，限制只处理指定 process
        distance_threshold: 阈值（米），用于在图中标记和统计
        bins: 直方图分箱数
    """
    if process_list is None:
        process_list = []

    df = pd.read_csv(args.input1, index_col=False)
    ionization_processes = get_ionization_process_list()

    if 'process' not in df.columns:
        print("Error: 'process' column not found in data")
        return

    total_ion_hits = df['process'].isin(ionization_processes).sum()
    if process_list:
        filtered_ion_hits = df[df['process'].isin(ionization_processes) & df['process'].isin(process_list)]
        print(
            f"Filter processes {process_list}, ionization hits kept: "
            f"{len(filtered_ion_hits)}/{total_ion_hits}"
        )

    grouped = df.groupby(['run', 'event'])
    total_events = len(grouped)
    events_limit = getattr(args, 'events', total_events)
    if events_limit is None:
        events_limit = total_events
    events_to_process = min(events_limit, total_events)
    print(f"Processing {events_to_process}/{total_events} events for ionization distance study")

    best_distances = []
    generator_track_distances = []
    hits_considered = 0
    hits_without_track = 0
    hits_missing_geometry = 0

    for i, ((run, event), event_data) in enumerate(grouped):
        if i >= events_to_process:
            break
        if (i + 1) % 100 == 0:
            print(f"  Event {i + 1}/{events_to_process} (Run: {run}, Event: {event})")

        # 识别好径迹并按圈生成几何参数（可重新计算turnId）
        good_track_params = compute_good_track_params(
            event_data,
            only_good_tracks=True,
            recalc_turn_id=True,
        )

        # 计算Generator好track上的hit到自身track曲线的距离
        for t_idx, params_list in good_track_params.items():
            track_data = event_data[event_data['trackIndex'] == t_idx]
            if track_data.empty:
                continue
            # 仅处理Generator process的好track
            track_process = track_data['process'].iloc[0] if 'process' in track_data.columns else None
            if track_process and 'Generator' in track_process:
                for _, hit_row in track_data.iterrows():
                    dist, _, missing_geom = compute_hit_to_track_distance(hit_row, {t_idx: params_list})
                    if missing_geom or dist is None:
                        continue
                    generator_track_distances.append(dist)

        ion_hits = event_data[event_data['process'].isin(ionization_processes)]
        if process_list:
            ion_hits = ion_hits[ion_hits['process'].isin(process_list)]
        hits_considered += len(ion_hits)

        for idx, row in ion_hits.iterrows():
            best_dist, best_track, missing_geometry = compute_hit_to_track_distance(row, good_track_params)
            if missing_geometry:
                hits_missing_geometry += 1
                continue
            if best_track is None or best_dist is None:
                hits_without_track += 1
                continue

            best_distances.append(best_dist)

    print("\n" + "=" * 60)
    print("IONIZATION HIT DISTANCE STATISTICS")
    print("=" * 60)
    print(f"Ionization hits considered: {hits_considered}")
    print(f"Valid distances collected: {len(best_distances)}")
    print(f"Hits without good track: {hits_without_track}")
    print(f"Hits missing geometry: {hits_missing_geometry}")
    print(f"Generator good-track hits collected: {len(generator_track_distances)}")

    if not best_distances:
        print("No distances to plot.")
        return

    best_array = np.array(best_distances)
    within = (best_array < distance_threshold).sum()
    print(f"Hits within {distance_threshold:.3f} m: {within} ({within/len(best_array)*100:.1f}%)")
    print(f"Mean: {best_array.mean():.4f} m, Median: {np.median(best_array):.4f} m")
    print(f"Min: {best_array.min():.4f} m, Max: {best_array.max():.4f} m")

    with PdfPages(args.output) as pdf:
        # 创建2x2四宫格布局
        if generator_track_distances:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            ax1 = axes[0, 0]
            ax2 = axes[0, 1]
            ax3 = axes[1, 0]
            ax_empty = axes[1, 1]
            # 隐藏右下角空白子图
            ax_empty.axis('off')
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        
        # 图1: 电离hit距离分布 (左上)
        ax1.hist(best_distances, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(distance_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {distance_threshold:.3f} m')
        ax1.set_xlabel('Distance to nearest good track (m)', fontsize=10)
        ax1.set_ylabel('Hit count', fontsize=10)
        ax1.set_title('Ionization hit reassignment distance', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)

        stats_text = (
            f"Total: {len(best_array)}\n"
            f"<= threshold: {within} ({within/len(best_array)*100:.1f}%)\n"
            f"Mean: {best_array.mean():.4f} m\n"
            f"Median: {np.median(best_array):.4f} m\n"
            f"Std: {best_array.std():.4f} m\n"
            f"Min: {best_array.min():.4f} m\n"
            f"Max: {best_array.max():.4f} m"
        )
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 图2: Generator好track hit距离分布 (右上)
        if generator_track_distances:
            gen_array = np.array(generator_track_distances)
            gen_within = (gen_array < distance_threshold).sum()
            
            ax2.hist(generator_track_distances, bins=bins, edgecolor='black', alpha=0.7, color='green')
            ax2.axvline(distance_threshold, color='red', linestyle='--', linewidth=2,
                           label=f'Threshold: {distance_threshold:.3f} m')
            ax2.set_xlabel('Distance to own track curve (m)', fontsize=10)
            ax2.set_ylabel('Hit count', fontsize=10)
            ax2.set_title('Generator good-track hit distance to own curve', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=9)

            gen_stats_text = (
                f"Total: {len(gen_array)}\n"
                f"<= threshold: {gen_within} ({gen_within/len(gen_array)*100:.1f}%)\n"
                f"Mean: {gen_array.mean():.4f} m\n"
                f"Median: {np.median(gen_array):.4f} m\n"
                f"Std: {gen_array.std():.4f} m\n"
                f"Min: {gen_array.min():.4f} m\n"
                f"Max: {gen_array.max():.4f} m"
            )
            ax2.text(0.98, 0.98, gen_stats_text, transform=ax2.transAxes, fontsize=8,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # 图3: 对比图 - Generator好track hit vs 电离hit (左下)
            # 使用相同的bins范围以便对比
            all_data = np.concatenate([best_distances, generator_track_distances])
            max_dist = np.percentile(all_data, 99)  # 使用99%分位数避免极端值影响
            bins_range = np.linspace(0, max_dist, bins)
            
            ax3.hist(generator_track_distances, bins=bins_range, alpha=0.6, 
                        color='red', label='Generator good-track hits', density=True)
            ax3.hist(best_distances, bins=bins_range, alpha=0.6, 
                        color='blue', label='Ionization hits', density=True)
            
            ax3.axvline(distance_threshold, color='black', linestyle='--', linewidth=2,
                           label=f'Threshold: {distance_threshold:.3f} m')
            ax3.set_xlabel('Distance to track curve (m)', fontsize=10)
            ax3.set_ylabel('Density', fontsize=10)
            ax3.set_title('Comparison: Generator good-track hits vs Ionization hits', fontsize=11, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Plots saved to {args.output}")

def analyze_decay_hit_distribution(args):
    """
    分析Decay过程中motherPID != -1的hit数分布
    用于确定Step 0中的阈值设置
    
    参数:
        args: 命令行参数，包含input1和output
    """
    print(f"Reading input file: {args.input1}")
    df = pd.read_csv(args.input1, index_col=False)
    
    # 检查必要的列是否存在
    if 'motherPID' not in df.columns:
        print("Error: motherPID column not found in data")
        return
    
    if 'process' not in df.columns:
        print("Error: process column not found in data")
        return
    
    # 按event分组
    grouped = df.groupby(['run', 'event'])
    
    # 存储每个motherPID组的hit数
    hit_counts_mother_eq_neg1 = []  # motherPID == -1 (所有)
    hit_counts_mother_ne_neg1 = []  # motherPID != -1 (所有)
    hit_counts_mother_ne_neg1_multiturn = []  # motherPID != -1 且 turnId > 1
    hit_counts_mother_ne_neg1_singleturn = []  # motherPID != -1 且 turnId == 1
    hit_counts_mother_eq_neg1_singleturn = []  # motherPID == -1 且 turnId == 1
    hit_counts_mother_eq_neg1_multiturn = []  # motherPID == -1 且 turnId > 1

    # 用于折线图的 (hit_count, turn_count) 对
    hit_turn_pairs_neg1_single = []
    hit_turn_pairs_neg1_multi = []
    hit_turn_pairs_ne_neg1_single = []
    hit_turn_pairs_ne_neg1_multi = []

    # Share vs Hit 细分：turn==1, turn==2, turn>2, turn<=2
    hits_turn1_neg1 = []
    hits_turn2_neg1 = []
    hits_turngt2_neg1 = []
    hits_turnle2_neg1 = []
    hits_turn1_ne_neg1 = []
    hits_turn2_ne_neg1 = []
    hits_turngt2_ne_neg1 = []
    hits_turnle2_ne_neg1 = []
    
    total_events = len(grouped)
    print(f"Processing {total_events} events...")
    
    for i, ((run, event), group) in enumerate(grouped):
        if (i + 1) % 100 == 0:
            print(f"Processing event {i + 1}/{total_events}")
        
        # 获取所有Decay过程的hits
        decay_hits = group[group['process'] == 'Decay'].copy()
        
        if len(decay_hits) > 0:
            # 按motherPID分组
            for mother_pid, mother_group in decay_hits.groupby('motherPID'):
                hit_count = len(mother_group)
                
                # 重新计算 turnId
                mother_group_copy = mother_group.copy()
                recalc_turn_ids = calculate_turn_id(mother_group_copy)
                max_turn_id = recalc_turn_ids.max() if len(recalc_turn_ids) > 0 else 0
                
                if mother_pid == -1:
                    hit_counts_mother_eq_neg1.append(hit_count)
                    # 根据圈数分类
                    if max_turn_id == 1:
                        hit_counts_mother_eq_neg1_singleturn.append(hit_count)
                        hit_turn_pairs_neg1_single.append((hit_count, max_turn_id))
                        hits_turn1_neg1.append(hit_count)
                        hits_turnle2_neg1.append(hit_count)
                    elif max_turn_id > 1:
                        hit_counts_mother_eq_neg1_multiturn.append(hit_count)
                        hit_turn_pairs_neg1_multi.append((hit_count, max_turn_id))
                        if max_turn_id == 2:
                            hits_turn2_neg1.append(hit_count)
                            hits_turnle2_neg1.append(hit_count)
                        elif max_turn_id > 2:
                            hits_turngt2_neg1.append(hit_count)
                else:
                    hit_counts_mother_ne_neg1.append(hit_count)
                    # 根据圈数分类
                    if max_turn_id == 1:
                        hit_counts_mother_ne_neg1_singleturn.append(hit_count)
                        hit_turn_pairs_ne_neg1_single.append((hit_count, max_turn_id))
                        hits_turn1_ne_neg1.append(hit_count)
                        hits_turnle2_ne_neg1.append(hit_count)
                    elif max_turn_id > 1:
                        hit_counts_mother_ne_neg1_multiturn.append(hit_count)
                        hit_turn_pairs_ne_neg1_multi.append((hit_count, max_turn_id))
                        if max_turn_id == 2:
                            hits_turn2_ne_neg1.append(hit_count)
                            hits_turnle2_ne_neg1.append(hit_count)
                        elif max_turn_id > 2:
                            hits_turngt2_ne_neg1.append(hit_count)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("DECAY HIT DISTRIBUTION STATISTICS")
    print("="*60)
    
    if hit_counts_mother_eq_neg1:
        print(f"\nMotherPID == -1 (All):")
        print(f"  Total groups: {len(hit_counts_mother_eq_neg1)}")
        print(f"  Mean hits per group: {np.mean(hit_counts_mother_eq_neg1):.2f}")
        print(f"  Median hits per group: {np.median(hit_counts_mother_eq_neg1):.2f}")
        print(f"  Min hits: {np.min(hit_counts_mother_eq_neg1)}")
        print(f"  Max hits: {np.max(hit_counts_mother_eq_neg1)}")
        print(f"  Std dev: {np.std(hit_counts_mother_eq_neg1):.2f}")
    else:
        print("\nMotherPID == -1 (All): No data found")
    
    if hit_counts_mother_eq_neg1_singleturn:
        print(f"\nMotherPID == -1 AND TurnId == 1 (Single-turn):")
        print(f"  Total groups: {len(hit_counts_mother_eq_neg1_singleturn)}")
        print(f"  Mean hits per group: {np.mean(hit_counts_mother_eq_neg1_singleturn):.2f}")
        print(f"  Median hits per group: {np.median(hit_counts_mother_eq_neg1_singleturn):.2f}")
        print(f"  Min hits: {np.min(hit_counts_mother_eq_neg1_singleturn)}")
        print(f"  Max hits: {np.max(hit_counts_mother_eq_neg1_singleturn)}")
        print(f"  Std dev: {np.std(hit_counts_mother_eq_neg1_singleturn):.2f}")
        if hit_counts_mother_eq_neg1:
            print(f"  Percentage of all motherPID==-1: {len(hit_counts_mother_eq_neg1_singleturn)/len(hit_counts_mother_eq_neg1)*100:.1f}%")
    else:
        print("\nMotherPID == -1 AND TurnId == 1: No data found")
    
    if hit_counts_mother_eq_neg1_multiturn:
        print(f"\nMotherPID == -1 AND TurnId > 1 (Multi-turn):")
        print(f"  Total groups: {len(hit_counts_mother_eq_neg1_multiturn)}")
        print(f"  Mean hits per group: {np.mean(hit_counts_mother_eq_neg1_multiturn):.2f}")
        print(f"  Median hits per group: {np.median(hit_counts_mother_eq_neg1_multiturn):.2f}")
        print(f"  Min hits: {np.min(hit_counts_mother_eq_neg1_multiturn)}")
        print(f"  Max hits: {np.max(hit_counts_mother_eq_neg1_multiturn)}")
        print(f"  Std dev: {np.std(hit_counts_mother_eq_neg1_multiturn):.2f}")
        if hit_counts_mother_eq_neg1:
            print(f"  Percentage of all motherPID==-1: {len(hit_counts_mother_eq_neg1_multiturn)/len(hit_counts_mother_eq_neg1)*100:.1f}%")
    else:
        print("\nMotherPID == -1 AND TurnId > 1: No data found")
    
    if hit_counts_mother_ne_neg1:
        print(f"\nMotherPID != -1 (All):")
        print(f"  Total groups: {len(hit_counts_mother_ne_neg1)}")
        print(f"  Mean hits per group: {np.mean(hit_counts_mother_ne_neg1):.2f}")
        print(f"  Median hits per group: {np.median(hit_counts_mother_ne_neg1):.2f}")
        print(f"  Min hits: {np.min(hit_counts_mother_ne_neg1)}")
        print(f"  Max hits: {np.max(hit_counts_mother_ne_neg1)}")
        print(f"  Std dev: {np.std(hit_counts_mother_ne_neg1):.2f}")
        
        # 计算不同阈值下的数量分布
        print("\n  Distribution by threshold:")
        thresholds = [10, 20, 30, 40, 50, 60, 80, 100]
        for threshold in thresholds:
            count_below = sum(1 for c in hit_counts_mother_ne_neg1 if c < threshold)
            count_above_eq = sum(1 for c in hit_counts_mother_ne_neg1 if c >= threshold)
            print(f"    < {threshold:3d} hits: {count_below:4d} groups ({count_below/len(hit_counts_mother_ne_neg1)*100:.1f}%)")
            print(f"    >={threshold:3d} hits: {count_above_eq:4d} groups ({count_above_eq/len(hit_counts_mother_ne_neg1)*100:.1f}%)")
            print()
    else:
        print("\nMotherPID != -1 (All): No data found")
    
    if hit_counts_mother_ne_neg1_singleturn:
        print(f"\nMotherPID != -1 AND TurnId == 1 (Single-turn):")
        print(f"  Total groups: {len(hit_counts_mother_ne_neg1_singleturn)}")
        print(f"  Mean hits per group: {np.mean(hit_counts_mother_ne_neg1_singleturn):.2f}")
        print(f"  Median hits per group: {np.median(hit_counts_mother_ne_neg1_singleturn):.2f}")
        print(f"  Min hits: {np.min(hit_counts_mother_ne_neg1_singleturn)}")
        print(f"  Max hits: {np.max(hit_counts_mother_ne_neg1_singleturn)}")
        print(f"  Std dev: {np.std(hit_counts_mother_ne_neg1_singleturn):.2f}")
        if hit_counts_mother_ne_neg1:
            print(f"  Percentage of all motherPID!=-1: {len(hit_counts_mother_ne_neg1_singleturn)/len(hit_counts_mother_ne_neg1)*100:.1f}%")
        
        # 计算不同阈值下的数量分布
        print("\n  Distribution by threshold:")
        thresholds = [10, 20, 30, 40, 50, 60, 80, 100]
        for threshold in thresholds:
            count_below = sum(1 for c in hit_counts_mother_ne_neg1_singleturn if c < threshold)
            count_above_eq = sum(1 for c in hit_counts_mother_ne_neg1_singleturn if c >= threshold)
            print(f"    < {threshold:3d} hits: {count_below:4d} groups ({count_below/len(hit_counts_mother_ne_neg1_singleturn)*100:.1f}%)")
            print(f"    >={threshold:3d} hits: {count_above_eq:4d} groups ({count_above_eq/len(hit_counts_mother_ne_neg1_singleturn)*100:.1f}%)")
            print()
    else:
        print("\nMotherPID != -1 AND TurnId == 1: No data found")
    
    if hit_counts_mother_ne_neg1_multiturn:
        print(f"\nMotherPID != -1 AND TurnId > 1 (Multi-turn):")
        print(f"  Total groups: {len(hit_counts_mother_ne_neg1_multiturn)}")
        print(f"  Mean hits per group: {np.mean(hit_counts_mother_ne_neg1_multiturn):.2f}")
        print(f"  Median hits per group: {np.median(hit_counts_mother_ne_neg1_multiturn):.2f}")
        print(f"  Min hits: {np.min(hit_counts_mother_ne_neg1_multiturn)}")
        print(f"  Max hits: {np.max(hit_counts_mother_ne_neg1_multiturn)}")
        print(f"  Std dev: {np.std(hit_counts_mother_ne_neg1_multiturn):.2f}")
        print(f"  Percentage of all motherPID!=-1: {len(hit_counts_mother_ne_neg1_multiturn)/len(hit_counts_mother_ne_neg1)*100:.1f}%")
        
        # 计算不同阈值下的数量分布
        print("\n  Distribution by threshold:")
        thresholds = [10, 20, 30, 40, 50, 60, 80, 100]
        for threshold in thresholds:
            count_below = sum(1 for c in hit_counts_mother_ne_neg1_multiturn if c < threshold)
            count_above_eq = sum(1 for c in hit_counts_mother_ne_neg1_multiturn if c >= threshold)
            print(f"    < {threshold:3d} hits: {count_below:4d} groups ({count_below/len(hit_counts_mother_ne_neg1_multiturn)*100:.1f}%)")
            print(f"    >={threshold:3d} hits: {count_above_eq:4d} groups ({count_above_eq/len(hit_counts_mother_ne_neg1_multiturn)*100:.1f}%)")
            print()
    else:
        print("\nMotherPID != -1 AND TurnId > 1: No data found")
    
    print("="*60 + "\n")
    
    # 绘制分布图 (3x3布局)
    fig, axes = plt.subplots(3, 3, figsize=(21, 18))
    
    # 图1: motherPID == -1 (All) 的hit数分布
    if hit_counts_mother_eq_neg1:
        axes[0, 0].hist(hit_counts_mother_eq_neg1, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Hit Count per Group', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('motherPID == -1 (All)', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(np.mean(hit_counts_mother_eq_neg1), color='r', linestyle='--', 
                           label=f'Mean: {np.mean(hit_counts_mother_eq_neg1):.1f}')
        axes[0, 0].axvline(np.median(hit_counts_mother_eq_neg1), color='g', linestyle='--', 
                           label=f'Median: {np.median(hit_counts_mother_eq_neg1):.1f}')
        axes[0, 0].legend(fontsize=9)
    else:
        axes[0, 0].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        axes[0, 0].set_title('motherPID == -1 (All)', fontsize=13, fontweight='bold')
    
    # 图2: motherPID == -1, TurnId == 1 的hit数分布
    if hit_counts_mother_eq_neg1_singleturn:
        axes[0, 1].hist(hit_counts_mother_eq_neg1_singleturn, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 1].set_xlabel('Hit Count per Group', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('motherPID == -1, TurnId == 1', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(np.mean(hit_counts_mother_eq_neg1_singleturn), color='r', linestyle='--', 
                           label=f'Mean: {np.mean(hit_counts_mother_eq_neg1_singleturn):.1f}')
        axes[0, 1].axvline(np.median(hit_counts_mother_eq_neg1_singleturn), color='g', linestyle='--', 
                           label=f'Median: {np.median(hit_counts_mother_eq_neg1_singleturn):.1f}')
        axes[0, 1].legend(fontsize=9)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        axes[0, 1].set_title('motherPID == -1, TurnId == 1', fontsize=13, fontweight='bold')
    
    # 图3: motherPID == -1, TurnId > 1 的hit数分布
    if hit_counts_mother_eq_neg1_multiturn:
        axes[0, 2].hist(hit_counts_mother_eq_neg1_multiturn, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
        axes[0, 2].set_xlabel('Hit Count per Group', fontsize=11)
        axes[0, 2].set_ylabel('Frequency', fontsize=11)
        axes[0, 2].set_title('motherPID == -1, TurnId > 1', fontsize=13, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axvline(np.mean(hit_counts_mother_eq_neg1_multiturn), color='r', linestyle='--', 
                           label=f'Mean: {np.mean(hit_counts_mother_eq_neg1_multiturn):.1f}')
        axes[0, 2].axvline(np.median(hit_counts_mother_eq_neg1_multiturn), color='g', linestyle='--', 
                           label=f'Median: {np.median(hit_counts_mother_eq_neg1_multiturn):.1f}')
        axes[0, 2].legend(fontsize=9)
    else:
        axes[0, 2].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        axes[0, 2].set_title('motherPID == -1, TurnId > 1', fontsize=13, fontweight='bold')
    
    # 图4: motherPID != -1 (All) 的hit数分布
    if hit_counts_mother_ne_neg1:
        axes[1, 0].hist(hit_counts_mother_ne_neg1, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Hit Count per Group', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('motherPID != -1 (All)', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(np.mean(hit_counts_mother_ne_neg1), color='r', linestyle='--', 
                           label=f'Mean: {np.mean(hit_counts_mother_ne_neg1):.1f}')
        axes[1, 0].axvline(np.median(hit_counts_mother_ne_neg1), color='g', linestyle='--', 
                           label=f'Median: {np.median(hit_counts_mother_ne_neg1):.1f}')
        axes[1, 0].axvline(40, color='purple', linestyle=':', linewidth=2, label='Threshold: 40')
        axes[1, 0].legend(fontsize=9)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        axes[1, 0].set_title('motherPID != -1 (All)', fontsize=13, fontweight='bold')
    
    # 图5: motherPID != -1, TurnId == 1 的hit数分布
    if hit_counts_mother_ne_neg1_singleturn:
        axes[1, 1].hist(hit_counts_mother_ne_neg1_singleturn, bins=50, edgecolor='black', alpha=0.7, color='gold')
        axes[1, 1].set_xlabel('Hit Count per Group', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('motherPID != -1, TurnId == 1', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(np.mean(hit_counts_mother_ne_neg1_singleturn), color='r', linestyle='--', 
                           label=f'Mean: {np.mean(hit_counts_mother_ne_neg1_singleturn):.1f}')
        axes[1, 1].axvline(np.median(hit_counts_mother_ne_neg1_singleturn), color='g', linestyle='--', 
                           label=f'Median: {np.median(hit_counts_mother_ne_neg1_singleturn):.1f}')
        axes[1, 1].axvline(40, color='purple', linestyle=':', linewidth=2, label='Threshold: 40')
        axes[1, 1].legend(fontsize=9)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        axes[1, 1].set_title('motherPID != -1, TurnId == 1', fontsize=13, fontweight='bold')
    
    # 图6: motherPID != -1, TurnId > 1 的hit数分布
    if hit_counts_mother_ne_neg1_multiturn:
        axes[1, 2].hist(hit_counts_mother_ne_neg1_multiturn, bins=50, edgecolor='black', alpha=0.7, color='red')
        axes[1, 2].set_xlabel('Hit Count per Group', fontsize=11)
        axes[1, 2].set_ylabel('Frequency', fontsize=11)
        axes[1, 2].set_title('motherPID != -1, TurnId > 1', fontsize=13, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axvline(np.mean(hit_counts_mother_ne_neg1_multiturn), color='r', linestyle='--', 
                           label=f'Mean: {np.mean(hit_counts_mother_ne_neg1_multiturn):.1f}')
        axes[1, 2].axvline(np.median(hit_counts_mother_ne_neg1_multiturn), color='g', linestyle='--', 
                           label=f'Median: {np.median(hit_counts_mother_ne_neg1_multiturn):.1f}')
        axes[1, 2].axvline(40, color='purple', linestyle=':', linewidth=2, label='Threshold: 40')
        axes[1, 2].legend(fontsize=9)
    else:
        axes[1, 2].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        axes[1, 2].set_title('motherPID != -1, TurnId > 1', fontsize=13, fontweight='bold')
    
    # 图7: motherPID != -1 (All) 的累积分布函数（CDF）
    if hit_counts_mother_ne_neg1:
        sorted_hits = np.sort(hit_counts_mother_ne_neg1)
        cumulative = np.arange(1, len(sorted_hits) + 1) / len(sorted_hits)
        axes[2, 0].plot(sorted_hits, cumulative, linewidth=2, color='orange')
        axes[2, 0].set_xlabel('Hit Count per Group', fontsize=11)
        axes[2, 0].set_ylabel('Cumulative Probability', fontsize=11)
        axes[2, 0].set_title('CDF: motherPID != -1 (All)', fontsize=13, fontweight='bold')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].axvline(40, color='purple', linestyle=':', linewidth=2, label='Threshold: 40')
        # 添加百分位线
        percentiles = [50, 75, 90]
        for p in percentiles:
            val = np.percentile(hit_counts_mother_ne_neg1, p)
            axes[2, 0].axhline(p/100, color='gray', linestyle='--', alpha=0.4, linewidth=0.7)
            axes[2, 0].axvline(val, color='gray', linestyle='--', alpha=0.4, linewidth=0.7)
            axes[2, 0].text(val, p/100, f'{p}%: {val:.0f}', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2, 0].legend(fontsize=9)
    else:
        axes[2, 0].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        axes[2, 0].set_title('CDF: motherPID != -1 (All)', fontsize=13, fontweight='bold')
    
    # 图8: motherPID != -1, TurnId == 1 的累积分布函数（CDF）
    if hit_counts_mother_ne_neg1_singleturn:
        sorted_hits_single = np.sort(hit_counts_mother_ne_neg1_singleturn)
        cumulative_single = np.arange(1, len(sorted_hits_single) + 1) / len(sorted_hits_single)
        axes[2, 1].plot(sorted_hits_single, cumulative_single, linewidth=2, color='gold')
        axes[2, 1].set_xlabel('Hit Count per Group', fontsize=11)
        axes[2, 1].set_ylabel('Cumulative Probability', fontsize=11)
        axes[2, 1].set_title('CDF: motherPID != -1, TurnId == 1', fontsize=13, fontweight='bold')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].axvline(40, color='purple', linestyle=':', linewidth=2, label='Threshold: 40')
        # 添加百分位线
        percentiles = [50, 75, 90]
        for p in percentiles:
            val = np.percentile(hit_counts_mother_ne_neg1_singleturn, p)
            axes[2, 1].axhline(p/100, color='gray', linestyle='--', alpha=0.4, linewidth=0.7)
            axes[2, 1].axvline(val, color='gray', linestyle='--', alpha=0.4, linewidth=0.7)
            axes[2, 1].text(val, p/100, f'{p}%: {val:.0f}', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2, 1].legend(fontsize=9)
    else:
        axes[2, 1].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        axes[2, 1].set_title('CDF: motherPID != -1, TurnId == 1', fontsize=13, fontweight='bold')
    
    # 图9: motherPID != -1, TurnId > 1 的累积分布函数（CDF）
    if hit_counts_mother_ne_neg1_multiturn:
        sorted_hits_multi = np.sort(hit_counts_mother_ne_neg1_multiturn)
        cumulative_multi = np.arange(1, len(sorted_hits_multi) + 1) / len(sorted_hits_multi)
        axes[2, 2].plot(sorted_hits_multi, cumulative_multi, linewidth=2, color='red')
        axes[2, 2].set_xlabel('Hit Count per Group', fontsize=11)
        axes[2, 2].set_ylabel('Cumulative Probability', fontsize=11)
        axes[2, 2].set_title('CDF: motherPID != -1, TurnId > 1', fontsize=13, fontweight='bold')
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].axvline(40, color='purple', linestyle=':', linewidth=2, label='Threshold: 40')
        # 添加百分位线
        percentiles = [50, 75, 90]
        for p in percentiles:
            val = np.percentile(hit_counts_mother_ne_neg1_multiturn, p)
            axes[2, 2].axhline(p/100, color='gray', linestyle='--', alpha=0.4, linewidth=0.7)
            axes[2, 2].axvline(val, color='gray', linestyle='--', alpha=0.4, linewidth=0.7)
            axes[2, 2].text(val, p/100, f'{p}%: {val:.0f}', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2, 2].legend(fontsize=9)
    else:
        axes[2, 2].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        axes[2, 2].set_title('CDF: motherPID != -1, TurnId > 1', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # 使用 PdfPages 保存多页
    print(f"Saving plot to: {args.output}")
    with PdfPages(args.output) as pdf:
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 追加折线图：横轴 hit 数，纵轴为在该阈值下（<= hit 数）的 track 占比
        fig_line, axes_line = plt.subplots(1, 2, figsize=(14, 5))

        def plot_cumulative_share(ax, hits_t1, hits_t2, hits_tgt2, hits_tle2, title):
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Hit Count threshold (<=)', fontsize=11)
            ax.set_ylabel('Share within Decay tracks', fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_ylim(0, 1.05)

            if (not hits_t1) and (not hits_t2) and (not hits_tgt2) and (not hits_tle2):
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10)
                return

            arr1 = np.sort(np.array(hits_t1)) if hits_t1 else np.array([])
            arr2 = np.sort(np.array(hits_t2)) if hits_t2 else np.array([])
            arr3 = np.sort(np.array(hits_tgt2)) if hits_tgt2 else np.array([])
            arr_le2 = np.sort(np.array(hits_tle2)) if hits_tle2 else np.array([])

            thresholds = np.unique(np.concatenate([arr1, arr2, arr3, arr_le2])) if len(arr1) + len(arr2) + len(arr3) + len(arr_le2) > 0 else np.array([])

            xs = []
            ys1 = []
            ys2 = []
            ys3 = []
            ys_le2 = []

            idx1 = idx2 = idx3 = idx_le2 = 0
            c1 = c2 = c3 = c_le2 = 0

            for t in thresholds:
                while idx1 < len(arr1) and arr1[idx1] <= t:
                    c1 += 1
                    idx1 += 1
                while idx2 < len(arr2) and arr2[idx2] <= t:
                    c2 += 1
                    idx2 += 1
                while idx3 < len(arr3) and arr3[idx3] <= t:
                    c3 += 1
                    idx3 += 1
                while idx_le2 < len(arr_le2) and arr_le2[idx_le2] <= t:
                    c_le2 += 1
                    idx_le2 += 1

                # 分母应该是所有Decay track的总数 (c1 + c2 + c3)，c_le2 = c1 + c2 所以不应该加入分母
                denom = c1 + c2 + c3
                if denom == 0:
                    continue
                xs.append(t)
                ys1.append(c1 / denom)
                ys2.append(c2 / denom)
                ys3.append(c3 / denom)
                # TurnId <= 2 的占比应该是 (c1 + c2) / denom，注意c_le2应该等于c1+c2
                ys_le2.append((c1 + c2) / denom)

            if xs:
                ax.plot(xs, ys1, label='TurnId == 1 (share)', color='blue', linewidth=1.8)
                ax.plot(xs, ys2, label='TurnId == 2 (share)', color='orange', linewidth=1.8)
                ax.plot(xs, ys3, label='TurnId > 2 (share)', color='red', linewidth=1.8)
                ax.plot(xs, ys_le2, label='TurnId <= 2 (share)', color='green', linewidth=1.8, linestyle='--')
            else:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10)

            ax.legend(fontsize=9)

        plot_cumulative_share(
            axes_line[0],
            hits_turn1_neg1,
            hits_turn2_neg1,
            hits_turngt2_neg1,
            hits_turnle2_neg1,
            'Share vs Hit (motherPID == -1)'
        )

        plot_cumulative_share(
            axes_line[1],
            hits_turn1_ne_neg1,
            hits_turn2_ne_neg1,
            hits_turngt2_ne_neg1,
            hits_turnle2_ne_neg1,
            'Share vs Hit (motherPID != -1)'
        )

        plt.tight_layout()
        pdf.savefig(fig_line, dpi=150, bbox_inches='tight')
        plt.close(fig_line)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Analysis entry point")
    parser.add_argument(
        "--run-mode",
        required=True,
        choices=[
            "distance",
            "hit-relationships",
            "distance-to-tracks",
            "decay-hit-distribution",
            "ionization-distance",
        ],
        help="distance: closest-distance histogram; hit-relationships: pairwise hit metrics; distance-to-tracks: ionization hit vs good track; decay-hit-distribution: Decay hit count by motherPID; ionization-distance: Step2 reassignment distance study",
    )
    parser.add_argument("--input1", help="Primary input CSV", required=True)
    parser.add_argument("--output", default="analysis.pdf", help="Output PDF path")
    parser.add_argument("--events", type=int, default=10, help="Max events to process")
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.01,
        help="Threshold in meters for ionization-distance mode",
    )
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins for ionization-distance mode")
    args = parser.parse_args()

    if args.run_mode == "distance":
        print("=== Running closest distance analysis ===")
        analyze_closest_distance_distribution(args)

    elif args.run_mode == "hit-relationships":
        print("=== Running hit relationship analysis ===")
        analyze_hit_relationships(args)

    elif args.run_mode == "distance-to-tracks":
        print("=== Running ionization vs good-track distance analysis ===")
        analyze_distance_to_tracks_distribution(args)

    elif args.run_mode == "decay-hit-distribution":
        print("=== Running Decay hit distribution analysis ===")
        analyze_decay_hit_distribution(args)

    elif args.run_mode == "ionization-distance":
        print("=== Running Step2 ionization reassignment distance analysis ===")
        analyze_ionization_distance_distribution(
            args,
            distance_threshold=args.distance_threshold,
            bins=args.bins,
        )


if __name__ == "__main__":
    main()