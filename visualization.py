import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages
from utils import sort_track_points_by_distance, calculate_momentum_direction, angle_difference, calculate_track_parameters, distinct_colors
from track_filter import filter_central_tracks

def visualize_raw_tracks(args):
    """Core visualization process for raw data with multiple tracks, showing track ID, PID, charge, hit count and unique layers in legend"""
    
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
            print(f"  Found {len(track_indices)} potential tracks")
            
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
                
                # 应用cut1条件，除非在白名单中
                if (hit_count < 6 or unique_layers < 6) and not is_whitelisted:
                    # 添加到额外的噪声数据中
                    extra_noise_data.append(track_data)
                    print(f"    Track {track_idx}: marked as noise (hits={hit_count}, layers={unique_layers})")
                else:
                    # 应用中央径迹筛选（使用排序后的数据）
                    is_central, closest_x, closest_y, min_distance = filter_central_tracks(
                        track_data_sorted
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
                        print(f"      - min distance: {min_distance:.3f} m")
                        if 'layer' in track_data_sorted.columns:
                            print(f"      - closest layer: {track_data_sorted.iloc[0]['layer']}")
            
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
                ax.scatter(all_noise_data['middleX']/100, all_noise_data['middleY']/100, 
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
                
                # 计算动量方向并绘制箭头（为径迹起始点）
                if args.show_momentum_arrows and len(track_data_sorted) >= 2:
                    # 获取径迹起始点的动量分量
                    start_mom_x = track_data_sorted['momX'].iloc[0]
                    start_mom_y = track_data_sorted['momY'].iloc[0]
                    
                    # 计算动量大小
                    mom_magnitude = np.sqrt(start_mom_x**2 + start_mom_y**2)
                    
                    if mom_magnitude > 0:
                        # 归一化动量方向向量
                        mom_dir_x = start_mom_x / mom_magnitude
                        mom_dir_y = start_mom_y / mom_magnitude
                        
                        # 计算箭头起始点（径迹起始点）
                        arrow_start_x = start_x
                        arrow_start_y = start_y
                        
                        # 计算箭头长度（根据动量大小调整）
                        base_length = 0.1  # 基础箭头长度
                        scale_factor = np.log1p(mom_magnitude) / 10  # 对数缩放因子
                        arrow_length = base_length * (1 + scale_factor)
                        
                        # 计算箭头终点
                        arrow_end_x = arrow_start_x + mom_dir_x * arrow_length
                        arrow_end_y = arrow_start_y + mom_dir_y * arrow_length
                        
                        # 绘制箭头
                        ax.arrow(arrow_start_x, arrow_start_y, 
                                arrow_end_x - arrow_start_x, arrow_end_y - arrow_start_y,
                                head_width=0.01, head_length=0.02, 
                                fc=color, ec=color, alpha=0.8, zorder=10, linewidth=0.5)
                
                # 简化图例文本，将process信息放在最后
                legend_text = f'track {track_idx}: hits={hit_count}, layers={unique_layers}, process={process}'
                label_added = False

                # 区分直丝和斜丝
                stereo_layers = [0, 1, 5, 6, 7, 8]
                
                if 'sLayer' in track_data.columns:
                    stereo_hits = track_data[track_data['sLayer'].isin(stereo_layers)]
                    axial_hits = track_data[~track_data['sLayer'].isin(stereo_layers)]
                else:
                    stereo_hits = pd.DataFrame()
                    axial_hits = track_data

                # 1. 绘制直丝点（空心圆点）
                if not axial_hits.empty:
                    point_diameters_m = axial_hits['rawDriftDist'] / 100
                    point_sizes = (point_diameters_m / 0.81) * 500
                    ax.scatter(axial_hits['middleX']/100, axial_hits['middleY']/100, 
                              color=color, marker='o', s=point_sizes,
                              facecolors='none', edgecolors=color, linewidths=0.5,
                              label=legend_text, 
                              alpha=1)
                    label_added = True

                # 2. 绘制斜丝点（计算着火点并绘制空心圆）
                if not stereo_hits.empty:
                    required_cols = ['eastX', 'eastY', 'eastZ', 'westX', 'westY', 'westZ', 'posZ']
                    use_calculated_pos = False
                    
                    if all(col in stereo_hits.columns for col in required_cols):
                        try:
                            # 计算着火点
                            # 利用Z坐标确定在丝上的位置比例 t
                            # z(t) = eastZ + t * (westZ - eastZ) => t = (posZ - eastZ) / (westZ - eastZ)
                            z_diff = stereo_hits['westZ'] - stereo_hits['eastZ']
                            
                            # 确保分母不为0（虽然物理上丝是有长度的）
                            if (np.abs(z_diff) > 1e-6).all():
                                t = (stereo_hits['posZ'] - stereo_hits['eastZ']) / z_diff
                                
                                # 计算对应的X和Y坐标
                                hit_x = stereo_hits['eastX'] + t * (stereo_hits['westX'] - stereo_hits['eastX'])
                                hit_y = stereo_hits['eastY'] + t * (stereo_hits['westY'] - stereo_hits['eastY'])
                                
                                point_diameters_m = stereo_hits['rawDriftDist'] / 100
                                point_sizes = (point_diameters_m / 0.81) * 500
                                
                                current_label = legend_text if not label_added else "_nolegend_"
                                
                                ax.scatter(hit_x/100, hit_y/100, 
                                          color=color, marker='o', s=point_sizes,
                                          facecolors='none', edgecolors=color, linewidths=0.5,
                                          label=current_label,
                                          alpha=1)
                                label_added = True
                                use_calculated_pos = True
                        except Exception as e:
                            print(f"    Warning: Failed to calculate stereo hit positions: {e}")

                    if not use_calculated_pos:
                        # 回退到使用 middleX/middleY 画圆圈
                        point_diameters_m = stereo_hits['rawDriftDist'] / 100
                        point_sizes = (point_diameters_m / 0.81) * 500
                        
                        current_label = legend_text if not label_added else "_nolegend_"
                        
                        ax.scatter(stereo_hits['middleX']/100, stereo_hits['middleY']/100, 
                                  color=color, marker='o', s=point_sizes,
                                  facecolors='none', edgecolors=color, linewidths=0.5,
                                  label=current_label,
                                  alpha=1)
                        label_added = True
                
                # 为特定process类型的hit点绘制动量方向箭头
                if args.show_momentum_arrows and 'process' in track_data.columns and 'momX' in track_data.columns and 'momY' in track_data.columns:
                    # 定义需要绘制箭头的process类型
                    target_processes = ['eIoni', 'muIoni', 'hIoni']
                    
                    # 筛选出目标process类型的hit点
                    target_hits = track_data[track_data['process'].isin(target_processes)]
                    
                    if not target_hits.empty:
                        # 为每个目标hit点绘制箭头
                        for _, hit in target_hits.iterrows():
                            # 获取hit点坐标（转换为米）
                            hit_x = hit['middleX'] / 100
                            hit_y = hit['middleY'] / 100
                            
                            # 获取动量分量
                            mom_x = hit['momX']
                            mom_y = hit['momY']
                            
                            # 计算动量大小
                            mom_magnitude = np.sqrt(mom_x**2 + mom_y**2)
                            
                            if mom_magnitude > 0:  # 避免除零错误
                                # 归一化动量方向向量
                                mom_dir_x = mom_x / mom_magnitude
                                mom_dir_y = mom_y / mom_magnitude
                                
                                # 根据动量大小调整箭头长度（对数缩放避免过大差异）
                                base_length = 0.08  # 基础箭头长度
                                scale_factor = np.log1p(mom_magnitude) / 10  # 对数缩放因子
                                arrow_length = base_length * (1 + scale_factor)
                                
                                # 计算箭头终点
                                arrow_end_x = hit_x + mom_dir_x * arrow_length
                                arrow_end_y = hit_y + mom_dir_y * arrow_length
                                
                                # 根据process类型设置箭头颜色
                                process_color_map = {
                                    'eIoni': 'red',
                                    'muIoni': 'blue', 
                                    'hIoni': 'green'
                                }
                                arrow_color = process_color_map.get(hit['process'], 'orange')
                                
                                # 绘制箭头
                                ax.arrow(hit_x, hit_y, 
                                        arrow_end_x - hit_x, arrow_end_y - hit_y,
                                        head_width=0.015, head_length=0.02, 
                                        fc=arrow_color, ec=arrow_color, alpha=0.9, 
                                        linewidth=0.5, zorder=15)
                
                # 绘制track的曲线（根据开关控制）
                if args.show_track_curve:
                    # 计算track参数
                    pt, r, sign, cx, cy = calculate_track_parameters(track_data)
                    
                    if cx is not None and cy is not None and r is not None and r > 0:
                        # 使用原始顺序的track点（保持hit点的自然顺序）
                        track_x = track_data['middleX'].values / 100
                        track_y = track_data['middleY'].values / 100
                        
                        # 计算每个点相对于圆心的角度
                        track_angles = np.arctan2(track_y - cy, track_x - cx)
                        
                        # 对hit点角度进行排序 (升序)
                        sorted_angles = np.sort(track_angles)
                        
                        # 计算相邻角度的差值
                        diffs = np.diff(sorted_angles)
                        
                        # 计算首尾角度的差值 (跨越2pi)
                        wrap_diff = 2 * np.pi - (sorted_angles[-1] - sorted_angles[0])
                        
                        # 找到最大间隔的位置
                        # diffs的索引 i 对应 sorted_angles[i] 和 sorted_angles[i+1] 之间的间隔
                        # 如果 wrap_diff 最大，说明没有跨越边界，不需要旋转
                        all_diffs = np.append(diffs, wrap_diff)
                        max_gap_idx = np.argmax(all_diffs)
                        
                        # 重新排列角度，使之连续
                        if max_gap_idx == len(diffs): # wrap_diff is max
                            # 已经是连续的，不需要旋转
                            ordered_angles = sorted_angles
                        else:
                            # 最大间隔在中间，需要旋转
                            # 间隔在 max_gap_idx 和 max_gap_idx+1 之间
                            # 新的起始点应该是 max_gap_idx + 1
                            ordered_angles = np.concatenate((sorted_angles[max_gap_idx+1:], sorted_angles[:max_gap_idx+1]))
                        
                        # 解包角度，使其单调
                        unwrapped_angles = np.unwrap(ordered_angles)
                        
                        # 生成曲线点
                        theta_start = unwrapped_angles[0]
                        theta_end = unwrapped_angles[-1]
                        
                        # 生成插值点
                        total_angle = abs(theta_end - theta_start)
                        num_points = max(20, int(100 * total_angle / (2 * np.pi)))
                        theta_smooth = np.linspace(theta_start, theta_end, num_points)
                        
                        # 计算坐标
                        curve_x = cx + r * np.cos(theta_smooth)
                        curve_y = cy + r * np.sin(theta_smooth)
                        
                        # 过滤超出探测器范围的点
                        # 将出界的点设为NaN，matplotlib会自动断开
                        plot_x = curve_x.copy()
                        plot_y = curve_y.copy()
                        for k in range(len(plot_x)):
                            if np.sqrt(plot_x[k]**2 + plot_y[k]**2) > 0.81:
                                plot_x[k] = np.nan
                                plot_y[k] = np.nan
                        
                        ax.plot(plot_x, plot_y, color=color, linestyle='-', 
                               alpha=0.9, linewidth=0.5, zorder=5)

            # 配置绘图设置
            ax.set(xlim=(-0.81,0.81), ylim=(-0.81,0.81),
                  xlabel='X (m)', ylabel='Y (m)',
                  title=f'Run {int(run_id)}, Event {int(event_id)} ({num_tracks} Central Tracks)')

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
    """分析电离process hit点到好track曲线的距离分布，以及好track自身hit点到曲线的距离分布"""
    from utils import get_ionization_process_list, identify_good_tracks, calculate_track_parameters, calculate_hit_geometry, calculate_distance_to_track
    
    # 读取原始数据
    df_rawData = pd.read_csv(args.input1, index_col=False)
    
    # 按run和event分组
    grouped_rawData = df_rawData.groupby(['run', 'event'])
    
    # 获取电离process列表
    ionization_processes = get_ionization_process_list()
    
    # 存储距离数据
    ionization_distances = [] # 电离hit点到最近好track的距离
    good_track_hit_distances = [] # 好track的hit点到自身track曲线的距离
    
    # 处理事件数限制
    max_possible_events = len(grouped_rawData)
    events_to_process = min(args.events, max_possible_events) if hasattr(args, 'events') else max_possible_events
    
    # print(f"Processing {events_to_process} events for distance-to-tracks analysis...")
    # print(f"Ionization processes: {ionization_processes}")
    
    # 遍历每个事件
    for event_num, ((run_id, event_id), raw_event_data) in enumerate(grouped_rawData):
        if event_num >= events_to_process:
            break
        
        print(f"Processing run: {run_id}, event: {event_id}")
        
        # 第一步：识别好track并计算其自身hit点的距离
        good_tracks = []
        good_tracks_params = [] # 存储 (track_data, params)
        
        # 获取所有非零的径迹索引
        track_indices = sorted(raw_event_data['trackIndex'].unique())
        
        for track_idx in track_indices:
            if track_idx == 0:  # 跳过噪声点
                continue
                
            track_data = raw_event_data[raw_event_data['trackIndex'] == track_idx]
            
            # 检查是否是好track
            if identify_good_tracks(track_data):
                # 计算track参数
                pt, track_radius, sign, track_center_x, track_center_y = calculate_track_parameters(track_data)
                
                if track_center_x is None or track_center_y is None or track_radius is None:
                    continue
                
                track_center = (track_center_x, track_center_y)
                good_tracks.append(track_data)
                good_tracks_params.append((track_data, track_radius, track_center))
                
                # print(f"  Found good track {track_idx}: hits={len(track_data)}, layers={track_data['layer'].nunique()}")
                
                # 计算该track自身hit点到曲线的距离
                for _, hit_row in track_data.iterrows():
                    hit_df = pd.DataFrame([hit_row])
                    hit_center, hit_radius = calculate_hit_geometry(hit_df)
                    
                    if hit_center is None or hit_radius is None:
                        continue
                        
                    dist = calculate_distance_to_track(hit_center, hit_radius, track_center, track_radius)
                    if dist is not None:
                        good_track_hit_distances.append(dist)
        
        # 如果没有好track，跳过这个事件
        if not good_tracks:
            print(f"  No good tracks found in this event")
            continue
        
        # 第二步：获取电离process的hit点
        ionization_hits = raw_event_data[raw_event_data['process'].isin(ionization_processes)]
        
        if not ionization_hits.empty:
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
                for _, track_radius, track_center in good_tracks_params:
                    # 计算距离参数d
                    distance = calculate_distance_to_track(hit_center, hit_radius, track_center, track_radius)
                    
                    if distance is not None:
                        distances_to_tracks.append(distance)
                
                # 只保留最小距离值（到最近track的距离）
                if distances_to_tracks:
                    min_distance = min(distances_to_tracks)
                    ionization_distances.append(min_distance)
                    # print(f"    Minimum distance to nearest track: {min_distance:.3f} m")
    
    # 第四步：绘制距离分布图
    if ionization_distances or good_track_hit_distances:
        # 创建输出文件路径
        output_path = args.output.replace('.pdf', '_distance_analysis.pdf')
        
        with PdfPages(output_path) as pdf:
            # 图1：电离process hit点到好track曲线的距离分布
            if ionization_distances:
                fig1, ax1 = plt.subplots(figsize=(10, 8))
                distances1 = np.array(ionization_distances)
                ax1.hist(distances1, bins=50, edgecolor='black', alpha=0.7, color='skyblue', label='Ionization Hits', range=(0, 0.1))
                ax1.set_xlabel('Distance to Nearest Good Track (m)', fontsize=12)
                ax1.set_ylabel('Count', fontsize=12)
                ax1.set_title(f'Distance Distribution: Ionization Hits to Good Tracks\nTotal Hits: {len(distances1)}', fontsize=14)
                ax1.grid(True, alpha=0.3)
                
                stats_text1 = (f'Statistics:\n'
                             f'Min: {np.min(distances1):.3f} m\n'
                             f'Max: {np.max(distances1):.3f} m\n'
                             f'Mean: {np.mean(distances1):.3f} m\n'
                             f'Std: {np.std(distances1):.3f} m\n'
                             f'Median: {np.median(distances1):.3f} m')
                ax1.text(0.95, 0.95, stats_text1, transform=ax1.transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                pdf.savefig(fig1)
                plt.close(fig1)
            
            # 图2：好track的hit点到好track曲线的距离分布
            if good_track_hit_distances:
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                distances2 = np.array(good_track_hit_distances)
                ax2.hist(distances2, bins=50, edgecolor='black', alpha=0.7, color='lightgreen', label='Good Track Hits', range=(0, 0.1))
                ax2.set_xlabel('Distance to Own Track Curve (m)', fontsize=12)
                ax2.set_ylabel('Count', fontsize=12)
                ax2.set_title(f'Distance Distribution: Good Track Hits to Own Curve\nTotal Hits: {len(distances2)}', fontsize=14)
                ax2.grid(True, alpha=0.3)
                
                stats_text2 = (f'Statistics:\n'
                             f'Min: {np.min(distances2):.3f} m\n'
                             f'Max: {np.max(distances2):.3f} m\n'
                             f'Mean: {np.mean(distances2):.3f} m\n'
                             f'Std: {np.std(distances2):.3f} m\n'
                             f'Median: {np.median(distances2):.3f} m')
                ax2.text(0.95, 0.95, stats_text2, transform=ax2.transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                pdf.savefig(fig2)
                plt.close(fig2)
            
            # 图3：对比图
            if ionization_distances and good_track_hit_distances:
                fig3, ax3 = plt.subplots(figsize=(10, 8))
                
                # 使用固定bins覆盖指定范围（例如 0 到 0.1 m）
                xmin, xmax = 0.0, 0.1
                bins = np.linspace(xmin, xmax, 50)
                
                # 使用相同的bins范围
                # all_data = np.concatenate([distances1, distances2])
                # min_val, max_val = np.min(all_data), np.max(all_data)
                # bins = np.linspace(min_val, max_val, 50)
                
                # 绘制两个直方图，使用density=True进行归一化比较，或者使用alpha叠加
                ax3.hist(distances1, bins=bins, alpha=0.5, color='skyblue', label='Ionization Hits', density=True)
                ax3.hist(distances2, bins=bins, alpha=0.5, color='lightgreen', label='Good Track Hits', density=True)
                
                ax3.set_xlabel('Distance to Track Curve (m)', fontsize=12)
                ax3.set_ylabel('Density', fontsize=12)
                ax3.set_title('Comparison of Distance Distributions (Normalized)', fontsize=14)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig3)
                plt.close(fig3)
                
            print(f"Distance analysis plots saved to: {output_path}")
    else:
        print("No distance data collected for plotting")
    
    return ionization_distances