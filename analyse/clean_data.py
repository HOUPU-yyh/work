import argparse
import pandas as pd
import numpy as np
import os
import sys

# Add current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    identify_good_tracks, 
    get_ionization_process_list, 
    calculate_track_parameters, 
    calculate_hit_geometry, 
    calculate_distance_to_track,
    calculate_turn_id,
    remove_direction_outlier_hits
)

# ============================================================================
# 处理步骤开关与阈值配置
# ============================================================================
ENABLE_STEP_0_MARK_DECAY = False               # Step 0: 标记衰变过程hits为噪声
ENABLE_STEP_1_RECALCULATE_TURNID = True       # Step 1: 重新计算turnID
ENABLE_STEP_2_REASSIGN_IONIZATION = False      # Step 2: 重新分配电离hits
ENABLE_STEP_3_REMOVE_OUTLIERS = False          # Step 3: 移除连续性离群点（仅圈数=1）
ENABLE_STEP_4_FILTER_NOISE_TRACKS = False      # Step 4: 过滤噪声径迹

# 连续性判断的邻近距离阈值（米），用于剔除与其他hit相距过远的孤立点
OUTLIER_GAP_THRESHOLD_M = 0.20
# ============================================================================

# Clean data function
def clean_data(args):
    print(f"Reading input file: {args.input}")
    df = pd.read_csv(args.input, index_col=False)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Group by run and event
    grouped = df.groupby(['run', 'event'])
    
    processed_rows = []
    
    ionization_processes = get_ionization_process_list()
    
    total_events = len(grouped)
    print(f"Processing {total_events} events...")
    
    # 全局统计变量
    total_decay_count = 0
    total_reassigned_count = 0
    total_outlier_count = 0
    total_turnid_corrected_count = 0
    total_filtered_tracks_count = 0
    total_events_dropped = 0
    
    for i, ((run, event), group) in enumerate(grouped):
        if (i + 1) % 100 == 0:
            print(f"Processing event {i + 1}/{total_events} (Run: {run}, Event: {event})")
            
        # Work on a copy of the group to avoid SettingWithCopy warnings
        event_data = group.copy()
        
        # 统计变量
        decay_count = 0
        reassigned_count = 0
        outlier_count = 0
        turnid_corrected_count = 0
        filtered_tracks_count = 0
        event_dropped = False
        
        # --- Step 0: Mark Decay process hits as noise based on motherPID ---
        if ENABLE_STEP_0_MARK_DECAY:
            # 检查是否有 motherPID 列
            if 'motherPID' in event_data.columns:
                # 获取所有 Decay 过程的 hits
                decay_mask = event_data['process'] == 'Decay'
                decay_hits = event_data[decay_mask]
                
                if len(decay_hits) > 0:
                    # 按 motherPID 分组
                    for mother_pid, mother_group in decay_hits.groupby('motherPID'):
                        if mother_pid == -1:
                            # motherPID == -1: 不做任何动作
                            continue
                        else:
                            # motherPID != -1: 检查 hit 数量
                            hit_count = len(mother_group)
                            
                            if hit_count < 40:
                                # hit 数 < 40: 标记为噪声
                                event_data.loc[mother_group.index, 'trackIndex'] = 0
                                decay_count += hit_count
                            else:
                                # hit 数 >= 40: 丢掉整个 event
                                event_dropped = True
                                break
                    
                    total_decay_count += decay_count
        
        # 如果 event 被丢掉，跳过后续处理
        if event_dropped:
            total_events_dropped += 1
            continue
        
        # --- Step 1: Recalculate turnID for all tracks ---
        if ENABLE_STEP_1_RECALCULATE_TURNID:
            current_track_indices = event_data['trackIndex'].unique()
            for t_idx in current_track_indices:
                if t_idx == 0: continue
                mask = event_data['trackIndex'] == t_idx
                t_rows = event_data[mask]
                if len(t_rows) > 0:
                    old_turn_ids = t_rows['turnId'].copy() if 'turnId' in t_rows.columns else None
                    new_turn_ids = calculate_turn_id(t_rows)
                    event_data.loc[mask, 'turnId'] = new_turn_ids
                    if old_turn_ids is not None:
                        corrected = (old_turn_ids != new_turn_ids).sum()
                        turnid_corrected_count += corrected
            total_turnid_corrected_count += turnid_corrected_count
        
        # --- Step 2: Reassign Ionization Hits ---
        if ENABLE_STEP_2_REASSIGN_IONIZATION:
            # Identify good tracks
            good_tracks = {} # trackIndex -> track_data
            track_indices = event_data['trackIndex'].unique()
            
            for idx in track_indices:
                if idx == 0: continue
                track_rows = event_data[event_data['trackIndex'] == idx]
                if identify_good_tracks(track_rows):
                    good_tracks[idx] = track_rows
            
            # Pre-calculate track parameters for good tracks
            # For multi-turn tracks, calculate parameters for each turn separately
            good_track_params = {}  # trackIndex -> [(pt, r, sign, track_center), ...]
            for t_idx, t_data in good_tracks.items():
                if 'turnId' in t_data.columns and t_data['turnId'].max() > 1:
                    # Multi-turn: calculate parameters for each turn
                    turn_params_list = []
                    for turn_id in sorted(t_data['turnId'].unique()):
                        turn_data = t_data[t_data['turnId'] == turn_id]
                        pt, r, sign, cx, cy = calculate_track_parameters(turn_data, use_turn_hint=False)
                        if cx is not None:
                            turn_params_list.append((pt, r, sign, (cx, cy)))
                    if turn_params_list:
                        good_track_params[t_idx] = turn_params_list
                else:
                    # Single-turn: use overall track parameters
                    pt, r, sign, cx, cy = calculate_track_parameters(t_data)
                    if cx is not None:
                        good_track_params[t_idx] = [(pt, r, sign, (cx, cy))]
            
            # Identify ionization hits
            ionization_mask = event_data['process'].isin(ionization_processes)
            
            # Iterate over ionization hits
            for idx, row in event_data[ionization_mask].iterrows():
                # Calculate geometry for this hit
                hit_df = pd.DataFrame([row])
                hit_center, hit_radius = calculate_hit_geometry(hit_df)
                
                if hit_center is None or hit_radius is None:
                    continue
                    
                best_dist = float('inf')
                best_track_idx = None
                
                # Try all good tracks and all their turns
                for t_idx, params_list in good_track_params.items():
                    for pt, r, sign, track_center in params_list:
                        dist = calculate_distance_to_track(hit_center, hit_radius, track_center, r)
                        
                        if dist is not None and dist < best_dist:
                            best_dist = dist
                            best_track_idx = t_idx
                
                # Threshold 1cm = 0.01m
                if best_track_idx is not None and best_dist < 0.01:
                    # Update the row in event_data
                    event_data.at[idx, 'trackIndex'] = best_track_idx
                    # Update process to good track's process
                    # Get process from good track (take the first one)
                    target_process = good_tracks[best_track_idx]['process'].iloc[0]
                    event_data.at[idx, 'process'] = target_process
                    reassigned_count += 1
                    
            total_reassigned_count += reassigned_count
                
        # --- Step 3: Remove Continuity Outlier Hits (turnId=1 only) ---
        if ENABLE_STEP_3_REMOVE_OUTLIERS:
            # 仅对圈数为1的track执行连续性过滤
                # 基于邻近距离的连续性过滤（仅turnId=1）
            current_track_indices = event_data['trackIndex'].unique()
            
            for t_idx in current_track_indices:
                if t_idx == 0: continue
                
                mask = event_data['trackIndex'] == t_idx
                t_rows = event_data[mask]
                
                if len(t_rows) > 3 and 'turnId' in t_rows.columns and t_rows['turnId'].max() == 1 and t_rows['turnId'].min() == 1:
                    clean_rows = remove_direction_outlier_hits(
                        t_rows,
                        gap_threshold=OUTLIER_GAP_THRESHOLD_M
                    )
                    
                    # 如果有点被移除，更新event_data
                    if len(clean_rows) < len(t_rows):
                        removed_indices = t_rows.index.difference(clean_rows.index)
                        # 将移除的点标记为噪声
                        event_data.loc[removed_indices, 'trackIndex'] = 0
                        outlier_count += len(removed_indices)
                    
            total_outlier_count += outlier_count
        
        # --- Step 4: Filter Noise Tracks ---
        if ENABLE_STEP_4_FILTER_NOISE_TRACKS:
            # Re-evaluate tracks because some might have gained hits
            
            # Get all track indices again (some hits might have moved from one track to another, 
            # or from noise/bad track to good track)
            current_track_indices = event_data['trackIndex'].unique()
            
            for t_idx in current_track_indices:
                if t_idx == 0: continue
                
                # Get hits for this track
                t_rows = event_data[event_data['trackIndex'] == t_idx]
                
                # Check condition
                hit_count = len(t_rows)
                unique_layers = t_rows['layer'].nunique()
                
                if hit_count < 6 or unique_layers < 6:
                    # Mark as noise
                    # Update all rows for this track
                    # Using boolean indexing on event_data
                    mask = event_data['trackIndex'] == t_idx
                    event_data.loc[mask, 'process'] = 0 
                    event_data.loc[mask, 'trackIndex'] = 0
                    filtered_tracks_count += 1
                    
            total_filtered_tracks_count += filtered_tracks_count
                
        processed_rows.append(event_data)
        
    # Combine
    if processed_rows:
        final_df = pd.concat(processed_rows)
    else:
        final_df = pd.DataFrame(columns=df.columns)
    
    # 输出整体统计信息
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Total events processed: {total_events}")
    print(f"Step 0 - Events dropped (Decay hit count >= 40): {total_events_dropped}")
    print(f"Step 0 - Decay hits marked as noise: {total_decay_count}")
    print(f"Step 1 - TurnID hits corrected: {total_turnid_corrected_count}")
    print(f"Step 2 - Ionization hits reassigned: {total_reassigned_count}")
    print(f"Step 3 - Continuity outlier hits removed (turnId=1): {total_outlier_count}")
    print(f"Step 4 - Noise tracks filtered: {total_filtered_tracks_count}")
    print("="*60 + "\n")
    
    # Save
    print(f"Saving to {args.output}")
    final_df.to_csv(args.output, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean track data")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file")
    args = parser.parse_args()
    
    clean_data(args)
