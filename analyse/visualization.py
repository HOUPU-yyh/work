import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages
from utils import sort_track_points_by_distance, calculate_momentum_direction, angle_difference, calculate_track_parameters, distinct_colors
from track_filter import filter_central_tracks

def visualize_raw_tracks(args, process_list=None):
    """Core visualization process for raw data with multiple tracks, showing track ID, PID, charge, hit count and unique layers in legend"""
    if process_list is None:
        process_list = []  # 如果没有提供process列表，默认为空
    
    # 读取原始数据
    df_rawData = pd.read_csv(args.input1, index_col=False)
    # 按run和event分组
    grouped_rawData = df_rawData.groupby(['run', 'event'])
    
    # 检查并调整最大事件数
    max_possible_events = len(grouped_rawData)
    if args.events > max_possible_events:
        print(f"Reached maximum event limit: {max_possible_events}. Stopping.")
        args.events = max_possible_events

    # 创建主输出PDF文件
    with PdfPages(args.output) as pdf:
        # 初始化存储所有事件的角度差数据
        all_angle_differences = []
        reassignment_counts = []  # 记录每个事件中重新分配的点的数量
        
        # 遍历每个事件
        for event_num, ((run_id, event_id), raw_event_data) in enumerate(grouped_rawData):
            if event_num >= args.events:
                break
            print(f"\nProcessing run: {run_id}, event: {event_id}")

            # 创建图形和坐标轴 - 使用固定尺寸确保每页大小相等
            fig, ax = plt.subplots(figsize=(12, 8))  # 固定尺寸
            # 绘制探测器边界（外圆和内圆）
            ax.add_patch(Circle((0,0), 0.81, fill=False, linestyle='-', alpha=0.5, color='black'))
            ax.add_patch(Circle((0,0), 0.06, fill=False, linestyle='-', alpha=0.5, color='black'))
            
            # 标记原点（探测器中心）
            ax.scatter(0, 0, color='red', marker='+', s=100, zorder=10, label='Detector Center')
            
            # 设置坐标轴比例为相等，使图形内容为正方形
            ax.set_aspect('equal')

            # 第一步：重新分类数据（基于hit数和layers数）
            # 将trackIndex==0的点标记为噪声
            noise_rawData = raw_event_data[raw_event_data['trackIndex'] == 0]
            
            # 将trackIndex!=0的点进一步筛选
            signal_rawData = raw_event_data[raw_event_data['trackIndex'] != 0]
            
            # 初始化列表来存储真正的信号径迹和额外的噪声径迹
            real_signal_data = []
            extra_noise_data = []
            
            # 获取所有非零的径迹索引
            track_indices = sorted(signal_rawData['trackIndex'].unique())
            # print(f"  Found {len(track_indices)} potential tracks")
            
            for track_idx in track_indices:
                # 获取当前径迹的数据
                track_data = signal_rawData[signal_rawData['trackIndex'] == track_idx]
                
                # 计算hit数和unique layers数
                hit_count = len(track_data)
                unique_layers = track_data['layer'].nunique()
                
                # 首先按距离排序
                track_data_sorted = sort_track_points_by_distance(track_data)
                
                # 判断是否应该被标记为噪声
                # 检查是否在白名单中（豁免cut1条件）
                is_whitelisted = False
                if args.whitelist_processes and 'process' in track_data.columns:
                    # 只有在有process列时才使用白名单功能
                    track_processes = track_data['process'].unique()
                    for proc in track_processes:
                        if proc in args.whitelist_processes:
                            is_whitelisted = True
                            break
                else:
                    # 如果没有process列，白名单功能无法使用
                    is_whitelisted = False
                    
                    # 应用cut1条件，除非在白名单中
                    if (hit_count < 6 or unique_layers < 6) and not is_whitelisted:
                        # 添加到额外的噪声数据中
                        extra_noise_data.append(track_data)
                        print(f"    Track {track_idx}: marked as noise (hits={hit_count}, layers={unique_layers})")
                    else:
                        # 应用中央径迹筛选（使用排序后的数据）
                        is_central, closest_x, closest_y, min_distance = filter_central_tracks(
                            track_data_sorted, 
                            max_distance_from_origin=args.max_distance,
                            max_layer_for_origin=args.max_layer
                        )
                        
                        if is_central:
                            # 添加到真正的信号径迹中（保持排序后的顺序）
                            real_signal_data.append(track_data_sorted)
                            print(f"    Track {track_idx}: ACCEPTED as central track")
                            print(f"      - hits={hit_count}, layers={unique_layers}")
                            print(f"      - closest point: ({closest_x:.3f}, {closest_y:.3f}) m")
                            print(f"      - min distance: {min_distance:.3f} m")
                            print(f"      - closest layer: {track_data_sorted.iloc[0]['layer'] if 'layer' in track_data_sorted.columns else 'N/A'}")
                        else:
                            # 即使hit数和layers数达标，但不是从中央打出，也标记为噪声
                            extra_noise_data.append(track_data)
                            print(f"    Track {track_idx}: REJECTED (not from center)")
                            print(f"      - hits={hit_count}, layers={unique_layers}")
                            print(f"      - closest point: ({closest_x:.3f}, {closest_y:.3f}) m")
                            print(f"      - min distance: {min_distance:.3f} m (threshold: {args.max_distance} m)")
                            if 'layer' in track_data_sorted.columns:
                                print(f"      - closest layer: {track_data_sorted.iloc[0]['layer']} (max allowed: {args.max_layer})")
                
                # print(f"\n  Summary:")
                # print(f"    Central tracks accepted: {len(real_signal_data)}")
                # print(f"    Tracks marked as noise: {len(extra_noise_data)}")
                
                # 初始化当前事件的角度差数据
                event_angle_differences = []
                event_reassigned_count = 0
                
                # 第二步：处理process列表中的点
                if process_list and len(process_list) > 0:
                    # 获取所有指定process的点
                    process_points = raw_event_data[raw_event_data['process'].isin(process_list)]
                    
                    if not process_points.empty:
                        # 创建DataFrame来存储需要重新分配到track的process点
                        reassigned_points = []
                        
                        # 对每个process点进行处理
                        for _, proc_point in process_points.iterrows():
                            proc_x = proc_point['posX'] / 100
                            proc_y = proc_point['posY'] / 100
                            
                            # 计算process点的动量方向
                            if 'momX' in proc_point and 'momY' in proc_point:
                                proc_mom_dir = calculate_momentum_direction(proc_point['momX'], proc_point['momY'])
                            else:
                                # 如果没有动量信息，跳过这个点
                                continue
                            
                            # 找到最近的track（基于真正的信号径迹）
                            min_distance = float('inf')
                            closest_track_idx = None
                            closest_track_points = None
                            
                            for track_data in real_signal_data:
                                if track_data.empty:
                                    continue
                                
                                # 计算这个track的所有点与process点的距离
                                track_x = track_data['posX'].values / 100
                                track_y = track_data['posY'].values / 100
                                
                                distances = np.sqrt((track_x - proc_x)**2 + (track_y - proc_y)**2)
                                min_track_distance = np.min(distances)
                                
                                if min_track_distance < min_distance:
                                    min_distance = min_track_distance
                                    closest_track_idx = track_data['trackIndex'].iloc[0]
                                    closest_track_points = track_data
                            
                            # 如果有找到最近的track
                            if closest_track_idx is not None and closest_track_points is not None:
                                # 计算track的动量方向（取track点的平均动量）
                                track_avg_momX = closest_track_points['momX'].mean()
                                track_avg_momY = closest_track_points['momY'].mean()
                                track_mom_dir = calculate_momentum_direction(track_avg_momX, track_avg_momY)
                                
                                # 计算动量方向的角度差（转换为度数）
                                angle_diff = angle_difference(proc_mom_dir, track_mom_dir)
                                angle_diff_deg = angle_diff * 180 / np.pi
                                
                                # 保存角度差数据
                                event_angle_differences.append(angle_diff_deg)
                                all_angle_differences.append(angle_diff_deg)
                                
                                # 如果在30度范围内，则重新分配这个点
                                if angle_diff_deg <= 30:
                                    event_reassigned_count += 1
                                    
                                    # 复制这个点并修改trackIndex
                                    reassigned_point = proc_point.copy()
                                    reassigned_point['trackIndex'] = closest_track_idx
                                    reassigned_points.append(reassigned_point)
                                    
                                    # 从原来的位置移除这个点
                                    # 如果是噪声，从噪声数据中移除
                                    if proc_point['trackIndex'] == 0:
                                        noise_rawData = noise_rawData[noise_rawData.index != proc_point.name]
                                    # 如果是其他track，从相应的track数据中移除
                                    else:
                                        found_in_real = False
                                        # 先检查real_signal_data
                                        for i, track_data in enumerate(real_signal_data):
                                            if not track_data.empty and track_data['trackIndex'].iloc[0] == proc_point['trackIndex']:
                                                # 从track数据中移除这个点
                                                real_signal_data[i] = track_data[track_data.index != proc_point.name]
                                                found_in_real = True
                                                break
                                        
                                        # 如果在real_signal_data中没找到，检查extra_noise_data
                                        if not found_in_real:
                                            for i, track_data in enumerate(extra_noise_data):
                                                if not track_data.empty and track_data['trackIndex'].iloc[0] == proc_point['trackIndex']:
                                                    # 从噪声数据中移除这个点
                                                    extra_noise_data[i] = track_data[track_data.index != proc_point.name]
                                                    break
                        
                        # 将重新分配的process点添加到相应的track中
                        if reassigned_points:
                            reassigned_df = pd.DataFrame(reassigned_points)
                            
                            # 按trackIndex分组并添加到相应的track数据中
                            for track_idx, group in reassigned_df.groupby('trackIndex'):
                                track_found = False
                                
                                # 添加到真正的信号径迹
                                for i, track_data in enumerate(real_signal_data):
                                    if not track_data.empty and track_data['trackIndex'].iloc[0] == track_idx:
                                        # 合并数据
                                        real_signal_data[i] = pd.concat([track_data, group], ignore_index=True)
                                        track_found = True
                                        break
                                
                                if not track_found:
                                    # 如果没有找到对应的track，创建一个新的
                                    print(f"Warning: No track found for index {track_idx}")
                
                # 记录当前事件的重新分配计数
                reassignment_counts.append(event_reassigned_count)
                
                # 第三步：合并所有的噪声数据（过滤掉空DataFrame）
                valid_extra_noise = [df for df in extra_noise_data if not df.empty]
                if valid_extra_noise:
                    extra_noise_df = pd.concat(valid_extra_noise, ignore_index=True)
                    # 将额外的噪声数据与原始的噪声数据合并
                    all_noise_data = pd.concat([noise_rawData, extra_noise_df], ignore_index=True)
                else:
                    all_noise_data = noise_rawData
                
                # 合并真正的信号径迹（过滤掉空DataFrame）
                valid_real_signal = [df for df in real_signal_data if not df.empty]
                if valid_real_signal:
                    real_signal_df = pd.concat(valid_real_signal, ignore_index=True)
                else:
                    real_signal_df = pd.DataFrame(columns=signal_rawData.columns)
                
                # 第四步：绘制所有噪声点（包括原来的噪声和新增的噪声）
                if not all_noise_data.empty:
                    ax.scatter(all_noise_data['posX']/100, all_noise_data['posY']/100, 
                              color='gray', marker='.', s=30,
                              label=f'noise ({len(all_noise_data)} points)', alpha=0.7)
                
                # 获取真正的信号径迹索引并排序
                real_track_indices = sorted(real_signal_df['trackIndex'].unique())
                num_tracks = len(real_track_indices)
                
                # 第五步：绘制每条真正的信号径迹
                for i, track_idx in enumerate(real_track_indices):
                    # 循环使用颜色列表
                    color = distinct_colors[i % len(distinct_colors)]
                    # 获取当前径迹的数据
                    track_data = real_signal_df[real_signal_df['trackIndex'] == track_idx]
                    
                    if track_data.empty:
                        continue
                    
                    # 按距离重新排序（为了找到起始点）
                    track_data_sorted = sort_track_points_by_distance(track_data)
                    
                    # 找到起始点（最靠近中心的点）
                    start_x = track_data_sorted['posX'].iloc[0] / 100
                    start_y = track_data_sorted['posY'].iloc[0] / 100
                    start_distance = np.sqrt(start_x**2 + start_y**2)
                    
                    # 获取径迹信息
                    pid = track_data['PID'].iloc[0] if 'PID' in track_data.columns else 'N/A'
                    process = track_data['process'].iloc[0] if 'process' in track_data.columns else 'N/A'
                    hit_count = len(track_data)
                    unique_layers = track_data['layer'].nunique()
                    mother_pid = track_data['motherPID'].iloc[0] if 'motherPID' in track_data.columns else 'N/A'
                    
                    # 简化图例文本，将process信息放在最后
                    legend_text = f'track {track_idx}: hits={hit_count}, layers={unique_layers}, process={process}'
                    
                    # 绘制径迹点（空心圆点，半径使用rawDriftDist）
                    # 将rawDriftDist转换为米，并计算点的显示面积（与坐标轴物理尺度匹配）
                    # 探测器半径0.81米对应图形尺寸，需要将物理尺寸转换为显示尺寸
                    point_diameters_m = track_data['rawDriftDist'] / 100  # 转换为米（直径）
                    point_sizes = (point_diameters_m / 0.81) * 500  # 按比例缩放
                    ax.scatter(track_data['middleX']/100, track_data['middleY']/100, 
                              color=color, marker='o', s=point_sizes,
                              facecolors='none', edgecolors=color, linewidths=0.5,
                              label=legend_text, 
                              alpha=1)
                    # 输出posX, posY, rawDriftDist
                    # 输出event，run，trackIndex
                    # print(f"event: {track_data['event'].iloc[0]}")
                    # print(f"run: {track_data['run'].iloc[0]}")
                    # print(f"trackIndex: {track_idx}")
                    # print(f"posX: {track_data['posX'].values / 100}")
                    # print(f"posY: {track_data['posY'].values / 100}")
                    # print(f"rawDriftDist: {track_data['rawDriftDist'].values * 100}")


                    # 在起始点处标记一个三角形（根据开关控制）
                    if args.show_start_triangle:
                        ax.scatter(start_x, start_y, color=color, marker='^', s=120, zorder=10, 
                                  edgecolors='black', linewidths=1)
                    

                    
                    # 绘制track的曲线（根据开关控制）
                    if args.show_track_curve:
                        # 计算track参数
                        pt, r, sign, cx, cy = calculate_track_parameters(track_data)
                        
                        if cx is not None and cy is not None and r is not None and r > 0:
                            # 使用原始顺序的track点（保持hit点的自然顺序）
                            track_x = track_data['posX'].values / 100
                            track_y = track_data['posY'].values / 100
                            
                            # 计算每个点相对于圆心的角度
                            track_angles = np.arctan2(track_y - cy, track_x - cx)
                            
                            # 找到最小和最大角度（考虑角度跨越问题）
                            # 计算原始角度的最小和最大值
                            min_angle = np.min(track_angles)
                            max_angle = np.max(track_angles)
                            
                            # 对于逆时针径迹，需要特殊处理角度范围
                            if sign < 0:  # 负电荷，逆时针
                                # 检查是否有边界跨越
                                # 计算原始角度范围的跨度
                                angle_span = max_angle - min_angle
                                
                                # 如果跨度小于π，说明可能有边界跨越
                                # 我们需要检查是否有角度接近0或2π
                                has_angle_near_zero = any(abs(angle) < np.pi/4 for angle in track_angles)
                                has_angle_near_2pi = any(abs(angle - 2*np.pi) < np.pi/4 for angle in track_angles)
                                
                                if angle_span < np.pi and (has_angle_near_zero or has_angle_near_2pi):
                                    # 有边界跨越，需要调整角度范围
                                    # 对于逆时针径迹，我们需要从最小角度到最大角度 + 2π
                                    max_angle += 2*np.pi
                            else:
                                # 正电荷，顺时针处理方式保持不变
                                angles_sorted = np.sort(track_angles)
                                angle_gaps = np.diff(angles_sorted)
                                max_gap_idx = np.argmax(angle_gaps)
                                
                                if angle_gaps[max_gap_idx] > np.pi:  # 如果最大间隔超过π，说明跨越了0度
                                    # 重新排列角度，使连续的部分在一起
                                    angles_shifted = np.concatenate([angles_sorted[max_gap_idx+1:], 
                                                                    angles_sorted[:max_gap_idx+1] + 2*np.pi])
                                    
                                    # 找到连续部分的最小和最大角度
                                    min_angle = angles_shifted[0]
                                    max_angle = angles_shifted[-1]
                                    
                                    # 调整到0-2π范围内
                                    if max_angle > 2*np.pi:
                                        max_angle -= 2*np.pi
                                        min_angle -= 2*np.pi
                                else:
                                    # 没有跨越0度，直接使用最小和最大角度
                                    min_angle = angles_sorted[0]
                                    max_angle = angles_sorted[-1]
                            
                            # 根据电荷符号调整角度范围
                            if sign > 0:  # 正电荷，顺时针
                                # 从最大角度到最小角度（顺时针方向）
                                theta_start = max_angle
                                theta_end = min_angle
                                if theta_start > theta_end:
                                    theta_end += 2 * np.pi
                            else:  # 负电荷，逆时针
                                # 从最小角度到最大角度（逆时针方向）
                                theta_start = min_angle
                                theta_end = max_angle
                                if theta_start > theta_end:
                                    theta_start -= 2 * np.pi
                            
                            # 生成圆弧点（只在有hit点的角度区间绘制曲线）
                            # 获取所有hit点的角度（相对于圆心）
                            hit_angles = track_angles
                            
                            # 对hit点角度进行排序
                            sorted_hit_angles = np.sort(hit_angles)
                            
                            # 生成曲线点，但只在有hit点的角度区间绘制
                            curve_segments = []
                            
                            # 根据电荷符号确定绘制方向
                            if sign > 0:  # 正电荷，顺时针
                                # 从最大角度到最小角度（顺时针方向）
                                sorted_hit_angles = np.sort(hit_angles)[::-1]  # 降序排列
                            else:  # 负电荷，逆时针
                                # 从最小角度到最大角度（逆时针方向）
                                sorted_hit_angles = np.sort(hit_angles)  # 升序排列
                            
                            # 为每对相邻的hit点生成曲线段
                            for i in range(len(sorted_hit_angles) - 1):
                                start_angle = sorted_hit_angles[i]
                                end_angle = sorted_hit_angles[i + 1]
                                
                                # 确保角度范围正确（考虑边界跨越）
                                if sign > 0:  # 顺时针
                                    if start_angle < end_angle:
                                        end_angle += 2 * np.pi
                                else:  # 逆时针
                                    if start_angle > end_angle:
                                        start_angle -= 2 * np.pi
                                
                                # 生成这个区间的曲线点
                                num_segment_points = max(10, int(20 * (end_angle - start_angle) / (2 * np.pi)))
                                theta_segment = np.linspace(start_angle, end_angle, num_segment_points)
                                
                                # 计算圆弧上的点
                                segment_x = cx + r * np.cos(theta_segment)
                                segment_y = cy + r * np.sin(theta_segment)
                                
                                # 过滤超出探测器范围的曲线点
                                valid_segment_indices = []
                                for j in range(len(segment_x)):
                                    dist = np.sqrt(segment_x[j]**2 + segment_y[j]**2)
                                    if dist <= 0.81:  # 探测器半径
                                        valid_segment_indices.append(j)
                                
                                if len(valid_segment_indices) > 1:
                                    # 只绘制有效的连续曲线段
                                    valid_segment_x = segment_x[valid_segment_indices]
                                    valid_segment_y = segment_y[valid_segment_indices]
                                    curve_segments.append((valid_segment_x, valid_segment_y))
                            
                            # 绘制所有有效的曲线段
                            for segment_x, segment_y in curve_segments:
                                ax.plot(segment_x, segment_y, color=color, linestyle='-', 
                                       alpha=0.9, linewidth=0.5, zorder=5)
                
                # 配置绘图设置
                ax.set(xlim=(-0.81,0.81), ylim=(-0.81,0.81),
                      xlabel='X (m)', ylabel='Y (m)',
                      title=f'Run {int(run_id)}, Event {int(event_id)} ({num_tracks} Central Tracks)\n'
                            f'Start point threshold: {args.max_distance} m, Max layer: {args.max_layer}')

                # 配置图例 - 使用固定位置和尺寸
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    # 将图例放在图形右侧外部，使用固定位置
                    # bbox_to_anchor=(1.05, 1.0) 表示在图形右上角外侧
                    ax.legend(handles, labels, loc='upper left',
                             fontsize=6, ncol=1, columnspacing=0.5, handletextpad=0.5,
                             bbox_to_anchor=(1.05, 1.0), borderaxespad=0.0,
                             frameon=True, fancybox=True, shadow=False,
                             prop={'family': 'monospace', 'size': 6})  # 使用等宽字体保持对齐
                
                # 固定图形布局，确保每页大小相等
                # 设置固定的边界框，不考虑图例大小
                ax.set_position([0.1, 0.1, 0.65, 0.8])  # [left, bottom, width, height]
                
                # 保存当前图形到PDF，使用固定边界框
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
        
        # 创建汇总的角度差分布图
        if all_angle_differences:
            print(f"\n=== Creating summary angle difference plots ===")
            print(f"Total angle difference points across all events: {len(all_angle_differences)}")
            print(f"Total reassigned points: {sum(reassignment_counts)}")
            
            # 创建直方图
            fig_summary, ax_summary = plt.subplots(figsize=(10, 8))
            
            # 绘制直方图
            counts, bins, patches = ax_summary.hist(all_angle_differences, bins=30, 
                                                   edgecolor='black', alpha=0.7, color='lightcoral')
            ax_summary.axvline(x=30, color='blue', linestyle='--', linewidth=2, 
                              label=f'30° threshold ({sum(1 for d in all_angle_differences if d <= 30)} points)')
            ax_summary.set_xlabel('Angle Difference (degrees)', fontsize=14)
            ax_summary.set_ylabel('Count', fontsize=14)
            ax_summary.set_title(f'All Events: Angle Differences Distribution\nTotal: {len(all_angle_differences)} points | ' +
                               f'Events processed: {len(reassignment_counts)}', 
                               fontsize=16)
            ax_summary.grid(True, alpha=0.3)
            ax_summary.legend(fontsize=12)
            
            # 在直方图上添加百分比线
            percentages = counts / len(all_angle_differences) * 100
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax_percent = ax_summary.twinx()
            ax_percent.plot(bin_centers, percentages, 'r-', linewidth=2, marker='o', 
                           markersize=4, label='Percentage')
            ax_percent.set_ylabel('Percentage (%)', fontsize=14, color='red')
            ax_percent.tick_params(axis='y', labelcolor='red')
            
            # 合并图例
            lines1, labels1 = ax_summary.get_legend_handles_labels()
            lines2, labels2 = ax_percent.get_legend_handles_labels()
            ax_summary.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
            
            # 添加统计信息
            filtered_angles = [d for d in all_angle_differences if d <= 30]
            stats_text = (f'Global Statistics:\n'
                         f'Total points: {len(all_angle_differences)}\n'
                         f'Mean: {np.mean(all_angle_differences):.2f}°\n'
                         f'Median: {np.median(all_angle_differences):.2f}°\n'
                         f'Std: {np.std(all_angle_differences):.2f}°\n'
                         f'Min: {np.min(all_angle_differences):.2f}°\n'
                         f'Max: {np.max(all_angle_differences):.2f}°\n'
                         f'Points ≤30°: {len(filtered_angles)} '
                         f'({len(filtered_angles)/len(all_angle_differences)*100:.1f}%)\n'
                         f'Total reassigned: {sum(reassignment_counts)}')
            
            ax_summary.text(0.98, 0.98, stats_text, transform=ax_summary.transAxes,
                          fontsize=10, verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            angle_pdf.savefig(fig_summary)
            plt.close(fig_summary)
            
            # 创建详细的≤30度分布直方图
            if filtered_angles:
                fig_detail, ax_detail = plt.subplots(figsize=(8, 6))
                ax_detail.hist(filtered_angles, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
                ax_detail.axvline(x=30, color='blue', linestyle='--', linewidth=2, 
                                 label=f'30° threshold ({len(filtered_angles)} points)')
                ax_detail.set_xlabel('Angle Difference (degrees)', fontsize=12)
                ax_detail.set_ylabel('Count', fontsize=12)
                ax_detail.set_title(f'Angle Differences ≤30° (Detail)\nTotal: {len(filtered_angles)} points | ' +
                                   f'{len(filtered_angles)/len(all_angle_differences)*100:.1f}% of all points', 
                                   fontsize=14)
                ax_detail.grid(True, alpha=0.3)
                ax_detail.legend(fontsize=10)
                
                # 添加统计信息
                detail_stats = (f'Statistics for ≤30° points:\n'
                               f'Mean: {np.mean(filtered_angles):.2f}°\n'
                               f'Std: {np.std(filtered_angles):.2f}°\n'
                               f'Median: {np.median(filtered_angles):.2f}°\n'
                               f'Min: {np.min(filtered_angles):.2f}°\n'
                               f'Max: {np.max(filtered_angles):.2f}°')
                
                ax_detail.text(0.02, 0.98, detail_stats, transform=ax_detail.transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                
                plt.tight_layout()
                angle_pdf.savefig(fig_detail)
                plt.close(fig_detail)
            
            print(f"Summary angle difference plots saved to: {angle_pdf_path}")
        else:
            print("\nNo angle difference data found. No summary plots created.")
    
    # print(f"\nAngle difference analysis complete. Plots saved to: {angle_pdf_path}")


def visualize_rec_tracks(args):
    """Visualize reconstructed tracks with truth and predicted information"""
    df_rawData = pd.read_csv(args.input1, index_col=False)
    df_predHits = pd.read_csv(args.input2)
    
    grouped_predHits = df_predHits.groupby(['t_run', 't_event'])
    grouped_rawData = df_rawData.groupby(['run', 'event'])
    
    max_possible_events = len(grouped_predHits)
    if args.events > max_possible_events:
        print(f"Reached maximum event limit: {max_possible_events}. Stopping.")
        args.events = max_possible_events

    distinct_colors = ['#7575EB', '#D89CCD', '#F2A930', '#8CB266']
    num_colors = len(distinct_colors)
    
    with PdfPages(args.output) as pdf:
        for event_num, ((t_run_id, t_event_id), p_event_data) in enumerate(grouped_predHits):
            if event_num >= args.events:
                break
            print(f"Processing t_run: {int(t_run_id)}, t_event: {int(t_event_id)}")

            # 使用固定尺寸确保每页大小相等
            fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
            ax.add_patch(Circle((0,0), 0.81, fill=True, linestyle='-', alpha=0.6, facecolor='whitesmoke', edgecolor='gray'))
            ax.add_patch(Circle((0,0), 0.06, fill=False, linestyle='-', alpha=0.6, color='black'))
            ax.set_aspect('equal')
            ax.set_xlabel('X (m)', fontsize=9)
            ax.set_ylabel('Y (m)', fontsize=9)
            ax.set_title(f'Run: {int(t_run_id)}, Event: {int(t_event_id)}', fontsize=10)

            try:
                grouped_rawData_copy = grouped_rawData.get_group((t_run_id, t_event_id)).copy()
            except KeyError:
                print(f"Warning: No raw data found for (run={t_run_id}, event={t_event_id}), skipping.")
                plt.close(fig) 
                continue

            valid_track_indices = grouped_rawData_copy['trackIndex'].drop_duplicates()
            valid_track_indices = valid_track_indices[
                (valid_track_indices >= 1) & 
                (valid_track_indices == valid_track_indices.astype(int))
            ].sort_values().tolist()
            print(f"Valid track indices in this event: {valid_track_indices}")

            if not valid_track_indices:
                print(f"No valid tracks in event (run={t_run_id}, event={t_event_id}), skipping.")
                plt.close(fig)
                continue

            original_tracks = {}  # 字典存储每条轨迹数据
            used_labels = set()   # 避免Track N图例重复
            y_offset = 0.98       # p_T文本垂直起始位置
            offset_step = 0.04    # 文本垂直间隔

            # 1. 绘制噪声点
            noise_data = grouped_rawData_copy[grouped_rawData_copy['trackIndex'] == 0]
            if not noise_data.empty:
                ax.scatter(
                    noise_data['posX']/100,
                    noise_data['posY']/100,
                    color="black",
                    marker='o',
                    s=5,
                    label="Noise",
                    alpha=1,
                    zorder=5
                )

            # 关键：新增3个图例标记变量
            cond_label_added = False    # 控制五角星（Condensate）图例
            correct_label_added = False # 控制正确重建空心圆图例
            wrong_label_added = False   # 控制错误重建空心圆图例

            # 循环处理每条轨迹
            for i, idx in enumerate(valid_track_indices):
                print(f"Processing track index: {idx}")

                # 计算truth p_T
                rawData = grouped_rawData_copy[grouped_rawData_copy['trackIndex'] == idx]
                if not rawData.empty:
                    selected_signal = rawData.sample(n=1).iloc[0]
                    t_pt_value = np.sqrt(selected_signal['initialMomX']**2 + selected_signal['initialMomY']** 2)
                    t_pt_text = f"Track {idx}: $p_T^{{truth}} = {t_pt_value:.3f}$ GeV/$c$"
                else:
                    t_pt_text = f"Track {idx}: $p_T^{{truth}} = None$"

                # 2. 绘制五角星（Condensate）
                cond_points = p_event_data[(p_event_data['t_trackIndex'] == idx) & (p_event_data['if_cond'] == 1)]
                if not cond_points.empty:
                    cond = cond_points.iloc[0]
                    p_pt_value = np.sqrt(cond['p_momx']**2 + cond['p_momy']** 2)
                    p_pt_text = f"$p_T^{{pred}} = {p_pt_value:.3f}$ GeV/$c$"

                    # 判断是否已添加五角星图例
                    if not cond_label_added:
                        ax.scatter(
                            cond['t_middle_x'], cond['t_middle_y'],
                            marker='*', 
                            facecolor=(45/255, 64/255, 105/255),
                            edgecolor='none',
                            linewidth=0.5,
                            s=80,
                            label='Condensate',
                            zorder=10,
                            alpha=0.9
                        )
                        cond_label_added = True
                    else:
                        ax.scatter(
                            cond['t_middle_x'], cond['t_middle_y'],
                            marker='*', 
                            facecolor=(45/255, 64/255, 105/255),
                            edgecolor='none',
                            linewidth=0.5,
                            s=80,
                            label="_nolegend_",
                            zorder=10,
                            alpha=0.9
                        )

                else:
                    p_pt_text = f"$p_T^{{pred}} = None$"

                # 绘制p_T文本
                info_text = f"{t_pt_text} | {p_pt_text}"
                ax.text(
                    0.02, y_offset, info_text,
                    transform=ax.transAxes, 
                    ha='left', 
                    va='top',
                    bbox=dict(
                        facecolor='none',
                        edgecolor='none',
                        linewidth=0.8,
                        alpha=1
                    ),
                    fontsize=6
                )
                y_offset -= offset_step
                if y_offset < 0.1: 
                    break

                # 3. 绘制轨迹点
                original_tracks[idx] = p_event_data[p_event_data['t_trackIndex'] == idx]
                if not original_tracks[idx].empty:
                    track_color = distinct_colors[(idx - 1) % num_colors]
                    track_label = f"Track {idx}"
                    if track_label not in used_labels:
                        ax.scatter(
                            original_tracks[idx]['t_middle_x'],
                            original_tracks[idx]['t_middle_y'],
                            color=track_color,
                            marker='o',
                            s=5,
                            label=track_label,
                            alpha=1,
                            zorder=2
                        )
                        used_labels.add(track_label)
                    else:
                        ax.scatter(
                            original_tracks[idx]['t_middle_x'],
                            original_tracks[idx]['t_middle_y'],
                            color=track_color,
                            marker='o',
                            s=5,
                            label="_nolegend_",
                            alpha=1,
                            zorder=2
                        )

                # 4. 绘制正确/错误重建空心圆
                # 正确重建
                correct_reco = p_event_data[(p_event_data['t_trackIndex'] == idx) & (p_event_data['if_rec'] == 1)]
                if not correct_reco.empty:
                    if not correct_label_added:
                        ax.scatter(
                            correct_reco['t_middle_x'], correct_reco['t_middle_y'],
                            edgecolor=track_color, facecolor='none', marker='o', s=25,
                            label='Correct Reconstruction',
                            zorder=3
                        )
                        correct_label_added = True
                    else:
                        ax.scatter(
                            correct_reco['t_middle_x'], correct_reco['t_middle_y'],
                            edgecolor=track_color, facecolor='none', marker='o', s=25,
                            label="_nolegend_",
                            zorder=3
                        )

                # 错误重建
                wrong_reco = p_event_data[(p_event_data['t_trackIndex'] == 0) & (p_event_data['if_rec'] == 1)]
                if not wrong_reco.empty:
                    if not wrong_label_added:
                        ax.scatter(
                            wrong_reco['t_middle_x'], wrong_reco['t_middle_y'],
                            edgecolor='crimson', facecolor='none', marker='o', s=25,
                            label='Wrong Reconstruction',
                            zorder=3
                        )
                        wrong_label_added = True
                    else:
                        ax.scatter(
                            wrong_reco['t_middle_x'], wrong_reco['t_middle_y'],
                            edgecolor='crimson', facecolor='none', marker='o', s=25,
                            label="_nolegend_",
                            zorder=3
                        )

            # 5. 添加图例 - 使用固定位置
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles, labels,
                    loc='upper left',
                    fontsize=6,
                    frameon=True,
                    fancybox=True,
                    shadow=False,
                    bbox_to_anchor=(1.05, 1.0),
                    borderaxespad=0.0
                )

            ax.set(xlim=(-0.9,0.9), ylim=(-0.9,0.9), xlabel='X (m)', ylabel='Y (m)')

            # 固定图形位置，确保每页大小相等
            ax.set_position([0.1, 0.1, 0.65, 0.8])
            
            # 保存图形，使用固定边界框
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)


def plot_closest_distance_distribution_by_process(distance_dict, output_path):
    """
    按process分类绘制每条径迹离中心最近的hit点的距离分布图
    
    参数:
        distance_dict (dict): 按process分类的距离数据字典
        output_path (str): 输出PDF文件路径
    """
    if not distance_dict:
        print("No distance data to plot")
        return
    
    with PdfPages(output_path) as pdf:
        # 为每个process绘制单独的图表
        for process_name, distances in distance_dict.items():
            if not distances:
                print(f"No distance data for process: {process_name}")
                continue
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 转换为numpy数组便于计算
            distances_array = np.array(distances)
            
            # 绘制直方图
            counts, bins, patches = ax.hist(distances_array, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax.set_xlabel('Distance from Origin (m)', fontsize=14)
            ax.set_ylabel('Count', fontsize=14)
            ax.set_title(f'Distribution of Closest Hit Distance from Origin\nProcess: {process_name} | Total Tracks: {len(distances)}', fontsize=16)
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息到直方图
            stats_text = (f'Statistics:\n'
                         f'Min: {np.min(distances_array):.3f} m\n'
                         f'Max: {np.max(distances_array):.3f} m\n'
                         f'Mean: {np.mean(distances_array):.3f} m\n'
                         f'Std: {np.std(distances_array):.3f} m\n'
                         f'Median: {np.median(distances_array):.3f} m')
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
            print(f"  Process {process_name}: {len(distances)} tracks, range: {np.min(distances_array):.3f} - {np.max(distances_array):.3f} m")
        
        # 添加合并分布图（除了Generator之外的所有process）
        non_generator_distances = []
        for process_name, distances in distance_dict.items():
            if 'generator' not in process_name.lower():
                non_generator_distances.extend(distances)
        
        if non_generator_distances:
            # 创建合并分布图
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 转换为numpy数组便于计算
            distances_array = np.array(non_generator_distances)
            
            # 绘制直方图
            counts, bins, patches = ax.hist(distances_array, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
            ax.set_xlabel('Distance from Origin (m)', fontsize=14)
            ax.set_ylabel('Count', fontsize=14)
            ax.set_title(f'Distribution of Closest Hit Distance from Origin\nAll Processes (excluding Generator) | Total Tracks: {len(non_generator_distances)}', fontsize=16)
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息到直方图
            stats_text = (f'Statistics:\n'
                         f'Min: {np.min(distances_array):.3f} m\n'
                         f'Max: {np.max(distances_array):.3f} m\n'
                         f'Mean: {np.mean(distances_array):.3f} m\n'
                         f'Std: {np.std(distances_array):.3f} m\n'
                         f'Median: {np.median(distances_array):.3f} m')
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
            print(f"  Combined (non-Generator): {len(non_generator_distances)} tracks, range: {np.min(distances_array):.3f} - {np.max(distances_array):.3f} m")
        
        print(f"Closest distance distribution plots saved to: {output_path}")
        print(f"Total processes analyzed: {len(distance_dict)}")
        if non_generator_distances:
            print(f"Combined non-Generator tracks: {len(non_generator_distances)}")


def plot_hit_relationship_distributions(angle_data, mom_angle_data, distance_data, output_path, title_suffix=""):
    """
    绘制hit间关系的三个分布图
    
    参数:
        angle_data (list): 位置向量角度差分布数据
        mom_angle_data (list): 动量向量角度差分布数据
        distance_data (list): 距离分布数据
        output_path (str): 输出PDF文件路径
        title_suffix (str): 标题后缀, 用于区分不同process组
    """
    if not angle_data and not mom_angle_data and not distance_data:
        print("No hit relationship data to plot")
        return
    
    with PdfPages(output_path) as pdf:
        # 创建包含三个子图的图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 转换为numpy数组便于计算
        angles = np.array(angle_data) if angle_data else np.array([])
        distances = np.array(distance_data) if distance_data else np.array([])
        
        # 1. 位置向量角度差分布图（左上）
        if len(angles) > 0:
            ax1 = axes[0, 0]
            counts, bins, patches = ax1.hist(angles, bins=30, edgecolor='black', alpha=0.7, color='lightblue')
            ax1.set_xlabel('Position Vector Angle Difference (degrees)', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
            ax1.set_title(f'Position Vector Angle Difference between Consecutive Hits {title_suffix}', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            # 添加统计信息
            stats_text = (f'Statistics:\n'
                         f'Min: {np.min(angles):.1f}°\n'
                         f'Max: {np.max(angles):.1f}°\n'
                         f'Mean: {np.mean(angles):.1f}°\n'
                         f'Std: {np.std(angles):.1f}°\n'
                         f'Median: {np.median(angles):.1f}°')
            
            ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            axes[0, 0].text(0.5, 0.5, 'No position angle difference data', horizontalalignment='center', 
                           verticalalignment='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title(f'Position Angle Difference (No Data) {title_suffix}', fontsize=14)
        
        # 2. 动量向量角度差分布图（右上）
        if len(mom_angle_data) > 0:
            ax2 = axes[0, 1]
            mom_angles = np.array(mom_angle_data)
            
            counts, bins, patches = ax2.hist(mom_angles, bins=30, edgecolor='black', alpha=0.7, color='coral')
            ax2.set_xlabel('Momentum Vector Angle Difference (degrees)', fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_title(f'Momentum Vector Angle Difference between Consecutive Hits {title_suffix}', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # 添加统计信息
            stats_text = (f'Statistics:\n'
                         f'Min: {np.min(mom_angles):.1f}°\n'
                         f'Max: {np.max(mom_angles):.1f}°\n'
                         f'Mean: {np.mean(mom_angles):.1f}°\n'
                         f'Std: {np.std(mom_angles):.1f}°\n'
                         f'Median: {np.median(mom_angles):.1f}°')
            
            ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            axes[0, 1].text(0.5, 0.5, 'No momentum angle difference data', horizontalalignment='center', 
                           verticalalignment='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title(f'Momentum Angle Difference (No Data) {title_suffix}', fontsize=14)
        
        # 3. 距离分布图（左下）
        if len(distances) > 0:
            ax3 = axes[1, 0]
            counts, bins, patches = ax3.hist(distances, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
            ax3.set_xlabel('Distance between Hits (m)', fontsize=12)
            ax3.set_ylabel('Count', fontsize=12)
            ax3.set_title(f'Distance Distribution between Consecutive Hits {title_suffix}', fontsize=14)
            ax3.grid(True, alpha=0.3)
            
            # 添加统计信息
            stats_text = (f'Statistics:\n'
                         f'Min: {np.min(distances):.3f} m\n'
                         f'Max: {np.max(distances):.3f} m\n'
                         f'Mean: {np.mean(distances):.3f} m\n'
                         f'Std: {np.std(distances):.3f} m\n'
                         f'Median: {np.median(distances):.3f} m')
            
            ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            axes[1, 0].text(0.5, 0.5, 'No distance data', horizontalalignment='center', 
                           verticalalignment='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title(f'Distance Distribution (No Data) {title_suffix}', fontsize=14)
        
        # 隐藏右下角的空子图
        axes[1, 1].axis('off')
        
        # 添加总标题
        fig.suptitle(f'Hit Relationship Analysis {title_suffix}\nTotal Hit Pairs: {len(angle_data) if angle_data else 0}', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        print(f"Hit relationship distribution plots saved to: {output_path}")
        print(f"Total hit pairs analyzed: {len(angle_data) if angle_data else 0}")

def analyze_distance_to_tracks_distribution(args):
    """分析电离process hit点到好track曲线的距离分布"""
    from utils import get_ionization_process_list, identify_good_tracks, calculate_track_parameters, calculate_hit_geometry, calculate_distance_to_track
    
    # 读取原始数据
    df_rawData = pd.read_csv(args.input1, index_col=False)
    
    # 按run和event分组
    grouped_rawData = df_rawData.groupby(['run', 'event'])
    
    # 获取电离process列表
    ionization_processes = get_ionization_process_list()
    
    # 存储所有距离数据
    all_distances = []
    
    # 处理事件数限制
    max_possible_events = len(grouped_rawData)
    events_to_process = min(args.events, max_possible_events) if hasattr(args, 'events') else max_possible_events
    
    print(f"Processing {events_to_process} events for distance-to-tracks analysis...")
    print(f"Ionization processes: {ionization_processes}")
    
    # 遍历每个事件
    for event_num, ((run_id, event_id), raw_event_data) in enumerate(grouped_rawData):
        if event_num >= events_to_process:
            break
        
        print(f"Processing run: {run_id}, event: {event_id}")
        
        # 第一步：识别好track
        good_tracks = []
        
        # 获取所有非零的径迹索引
        track_indices = sorted(raw_event_data['trackIndex'].unique())
        
        for track_idx in track_indices:
            if track_idx == 0:  # 跳过噪声点
                continue
                
            track_data = raw_event_data[raw_event_data['trackIndex'] == track_idx]
            
            # 检查是否是好track
            if identify_good_tracks(track_data):
                good_tracks.append(track_data)
                print(f"  Found good track {track_idx}: hits={len(track_data)}, layers={track_data['layer'].nunique()}")
        
        # 如果没有好track，跳过这个事件
        if not good_tracks:
            print(f"  No good tracks found in this event")
            continue
        
        # 第二步：获取电离process的hit点
        ionization_hits = raw_event_data[raw_event_data['process'].isin(ionization_processes)]
        
        if ionization_hits.empty:
            print(f"  No ionization process hits found in this event")
            continue
        
        print(f"  Found {len(ionization_hits)} ionization process hits")
        
        # 第三步：计算每个电离hit点到最近好track的距离
        for _, hit_data in ionization_hits.iterrows():
            # 将单行数据转换为DataFrame
            hit_df = pd.DataFrame([hit_data])
            
            # 计算hit点的几何参数
            hit_center, hit_radius = calculate_hit_geometry(hit_df)
            
            if hit_center is None or hit_radius is None:
                continue
            
            # 存储该hit点到所有好track的距离
            distances_to_tracks = []
            
            # 对每个好track计算距离
            for track_data in good_tracks:
                # 计算track的几何参数
                pt, track_radius, sign, track_center_x, track_center_y = calculate_track_parameters(track_data)
                
                if track_center_x is None or track_center_y is None or track_radius is None:
                    continue
                
                track_center = (track_center_x, track_center_y)
                
                # 计算距离参数d
                distance = calculate_distance_to_track(hit_center, hit_radius, track_center, track_radius)
                
                if distance is not None:
                    distances_to_tracks.append(distance)
                    print(f"    Distance to track {track_data['trackIndex'].iloc[0]}: {distance:.3f} m")
            
            # 只保留最小距离值（到最近track的距离）
            if distances_to_tracks:
                min_distance = min(distances_to_tracks)
                all_distances.append(min_distance)
                print(f"    Minimum distance to nearest track: {min_distance:.3f} m")
    
    # 第四步：绘制距离分布图
    if all_distances:
        # 创建输出文件路径
        output_path = args.output.replace('.pdf', '_distance_to_tracks.pdf')
        
        with PdfPages(output_path) as pdf:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 转换为numpy数组
            distances = np.array(all_distances)
            
            # 绘制直方图
            counts, bins, patches = ax.hist(distances, bins=50, edgecolor='black', alpha=0.7, color='lightblue')
            ax.set_xlabel('Distance to Track Curve (d = R - |O - o|) (m)', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'Distance Distribution of Ionization Hits to Good Tracks\nTotal Hits: {len(all_distances)}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            stats_text = (f'Statistics:\n'
                         f'Min: {np.min(distances):.3f} m\n'
                         f'Max: {np.max(distances):.3f} m\n'
                         f'Mean: {np.mean(distances):.3f} m\n'
                         f'Std: {np.std(distances):.3f} m\n'
                         f'Median: {np.median(distances):.3f} m')
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 添加电离process列表信息
            process_text = f'Ionization Processes:\n{chr(10).join(ionization_processes)}'
            ax.text(0.02, 0.98, process_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
            print(f"Distance distribution plot saved to: {output_path}")
            print(f"Total distance measurements: {len(all_distances)}")
            print(f"Distance statistics: min={np.min(distances):.3f}, max={np.max(distances):.3f}, mean={np.mean(distances):.3f}, std={np.std(distances):.3f}")
    else:
        print("No distance data collected for plotting")
    
    return all_distances