import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import distinct_colors


def create_detector_cylinder(radius=0.81, height=2.0, num_points=50):
    """创建探测器圆柱体的3D网格"""
    theta = np.linspace(0, 2*np.pi, num_points)
    z = np.linspace(-height/2, height/2, 2)
    
    # 外圆柱面
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    
    return x, y, z_grid


def create_detector_endcaps(radius=0.81, num_points=50):
    """创建探测器两端的圆盘"""
    theta = np.linspace(0, 2*np.pi, num_points)
    r = np.linspace(0, radius, 10)
    
    theta_grid, r_grid = np.meshgrid(theta, r)
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    
    return x, y, theta_grid


def visualize_3d_event(args):
    """3D可视化指定的单个事件"""
    
    # 读取原始数据
    print(f"Reading data from {args.input1}...")
    df_rawData = pd.read_csv(args.input1, index_col=False)
    
    # 筛选指定的run和event
    event_data = df_rawData[(df_rawData['run'] == args.run) & (df_rawData['event'] == args.event)]
    
    if event_data.empty:
        print(f"Error: No data found for run={args.run}, event={args.event}")
        print(f"Available runs: {sorted(df_rawData['run'].unique())}")
        print(f"Available events in run {args.run}: {sorted(df_rawData[df_rawData['run'] == args.run]['event'].unique()) if args.run in df_rawData['run'].values else 'N/A'}")
        return
    
    print(f"Processing run: {args.run}, event: {args.event}")
    print(f"Total hits in this event: {len(event_data)}")
    
    # 创建3D图形
    fig = go.Figure()
    
    # 添加探测器圆柱体（外圆柱）
    x_cyl, y_cyl, z_cyl = create_detector_cylinder(radius=0.81, height=2.0)
    fig.add_trace(go.Surface(
        x=x_cyl, y=y_cyl, z=z_cyl,
        colorscale=[[0, 'lightgray'], [1, 'lightgray']],
        showscale=False,
        opacity=0.1,
        name='Detector Outer Cylinder',
        hoverinfo='skip'
    ))
    
    # 添加内圆柱
    x_cyl_inner, y_cyl_inner, z_cyl_inner = create_detector_cylinder(radius=0.06, height=2.0)
    fig.add_trace(go.Surface(
        x=x_cyl_inner, y=y_cyl_inner, z=z_cyl_inner,
        colorscale=[[0, 'gray'], [1, 'gray']],
        showscale=False,
        opacity=0.2,
        name='Detector Inner Cylinder',
        hoverinfo='skip'
    ))
    
    # 添加探测器中心点
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=5, color='red', symbol='cross'),
        name='Detector Center'
    ))
    
    # 区分噪声和信号
    noise_data = event_data[event_data['trackIndex'] == 0]
    signal_data = event_data[event_data['trackIndex'] != 0]
    
    # 绘制噪声点
    if not noise_data.empty:
        fig.add_trace(go.Scatter3d(
            x=noise_data['posX'] / 100,
            y=noise_data['posY'] / 100,
            z=noise_data['posZ'] / 100,
            mode='markers',
            marker=dict(size=3, color='gray', opacity=0.5),
            name=f'Noise ({len(noise_data)} hits)',
            hovertemplate='<b>Noise</b><br>X: %{x:.3f} m<br>Y: %{y:.3f} m<br>Z: %{z:.3f} m<extra></extra>'
        ))
    
    # 获取所有信号径迹索引
    track_indices = sorted(signal_data['trackIndex'].unique())
    num_tracks = len(track_indices)
    print(f"Found {num_tracks} tracks")
    
    # 绘制每条信号径迹
    for i, track_idx in enumerate(track_indices):
        color = distinct_colors[i % len(distinct_colors)]
        track_data = signal_data[signal_data['trackIndex'] == track_idx]
        
        if track_data.empty:
            continue
        
        # 获取径迹信息
        pid = track_data['PID'].iloc[0] if 'PID' in track_data.columns else 'N/A'
        process = track_data['process'].iloc[0] if 'process' in track_data.columns else 'N/A'
        hit_count = len(track_data)
        unique_layers = track_data['layer'].nunique()
        mother_pid = track_data['motherPID'].iloc[0] if 'motherPID' in track_data.columns else 'N/A'
        
        # 获取turnId信息
        if 'turnId' in track_data.columns:
            turn_ids = sorted(track_data['turnId'].unique())
            turn_id_str = ','.join([str(int(tid)) for tid in turn_ids])
        else:
            turn_id_str = 'N/A'
        
        legend_text = f'Track {track_idx}: PID={pid}, mPID={mother_pid}, turnId={turn_id_str}, hits={hit_count}, layers={unique_layers}'
        
        # 绘制hit点
        fig.add_trace(go.Scatter3d(
            x=track_data['posX'] / 100,
            y=track_data['posY'] / 100,
            z=track_data['posZ'] / 100,
            mode='markers',
            marker=dict(size=4, color=color, opacity=0.8),
            name=legend_text,
            hovertemplate=f'<b>Track {track_idx}</b><br>X: %{{x:.3f}} m<br>Y: %{{y:.3f}} m<br>Z: %{{z:.3f}} m<br>Layer: %{{customdata[0]}}<br>Process: {process}<extra></extra>',
            customdata=track_data[['layer']].values if 'layer' in track_data.columns else None
        ))
        
        # 如果需要，绘制径迹线（连接同一track的hit点）
        if args.show_track_lines:
            # 按scaledFltLen排序以正确连接点
            if 'scaledFltLen' in track_data.columns:
                track_data_sorted = track_data.sort_values('scaledFltLen')
            else:
                # 如果没有scaledFltLen，按z坐标排序
                track_data_sorted = track_data.sort_values('posZ')
            
            fig.add_trace(go.Scatter3d(
                x=track_data_sorted['posX'] / 100,
                y=track_data_sorted['posY'] / 100,
                z=track_data_sorted['posZ'] / 100,
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                name=f'Track {track_idx} path',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # 绘制动量箭头（在起始点）
        if args.show_momentum_arrows and 'momX' in track_data.columns:
            # 按距离中心最近的点作为起始点
            distances = np.sqrt(track_data['posX']**2 + track_data['posY']**2)
            start_idx = distances.idxmin()
            start_point = track_data.loc[start_idx]
            
            start_x = start_point['posX'] / 100
            start_y = start_point['posY'] / 100
            start_z = start_point['posZ'] / 100
            
            mom_x = start_point['momX']
            mom_y = start_point['momY']
            mom_z = start_point['momZ'] if 'momZ' in track_data.columns else 0
            
            mom_magnitude = np.sqrt(mom_x**2 + mom_y**2 + mom_z**2)
            
            if mom_magnitude > 0:
                # 归一化并缩放箭头长度
                scale = 0.15 * (1 + np.log1p(mom_magnitude) / 10)
                arrow_end_x = start_x + (mom_x / mom_magnitude) * scale
                arrow_end_y = start_y + (mom_y / mom_magnitude) * scale
                arrow_end_z = start_z + (mom_z / mom_magnitude) * scale
                
                # 绘制箭头（使用线段和锥体）
                fig.add_trace(go.Scatter3d(
                    x=[start_x, arrow_end_x],
                    y=[start_y, arrow_end_y],
                    z=[start_z, arrow_end_z],
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=f'Track {track_idx} momentum',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # 添加箭头头部（锥体）
                fig.add_trace(go.Cone(
                    x=[arrow_end_x], y=[arrow_end_y], z=[arrow_end_z],
                    u=[mom_x / mom_magnitude], v=[mom_y / mom_magnitude], w=[mom_z / mom_magnitude],
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    sizemode='absolute',
                    sizeref=0.02,
                    name=f'Track {track_idx} momentum direction',
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # 设置布局
    fig.update_layout(
        title=f'3D Visualization - Run {args.run}, Event {args.event} ({num_tracks} Tracks)',
        scene=dict(
            xaxis=dict(title='X (m)', range=[-0.9, 0.9]),
            yaxis=dict(title='Y (m)', range=[-0.9, 0.9]),
            zaxis=dict(title='Z (m)'),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=8)
        ),
        width=1200,
        height=900
    )
    
    # 保存或显示图形
    if args.output:
        print(f"Saving 3D visualization to {args.output}...")
        fig.write_html(args.output)
        print(f"3D visualization saved successfully!")
        print(f"Open {args.output} in a web browser to view the interactive 3D plot.")
    else:
        print("Displaying 3D visualization...")
        fig.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='3D Visualization of Particle Tracks')
    
    parser.add_argument('--input1', type=str, required=True,
                        help='Path to input CSV file with raw data')
    parser.add_argument('--run', type=int, required=True,
                        help='Run number to visualize')
    parser.add_argument('--event', type=int, required=True,
                        help='Event number to visualize')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file path (default: display in browser)')
    parser.add_argument('--show-track-lines', action='store_true',
                        help='Show lines connecting hits in the same track')
    parser.add_argument('--show-momentum-arrows', action='store_true',
                        help='Show momentum direction arrows at track start points')
    
    args = parser.parse_args()
    
    visualize_3d_event(args)


if __name__ == '__main__':
    main()
