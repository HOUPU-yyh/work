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
    calculate_distance_to_track
)

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
    
    for i, ((run, event), group) in enumerate(grouped):
        if (i + 1) % 100 == 0:
            print(f"Processing event {i + 1}/{total_events} (Run: {run}, Event: {event})")
            
        # Work on a copy of the group to avoid SettingWithCopy warnings
        event_data = group.copy()
        
        # --- Step 1: Reassign Ionization Hits ---
        
        # Identify good tracks
        good_tracks = {} # trackIndex -> track_data
        track_indices = event_data['trackIndex'].unique()
        
        for idx in track_indices:
            if idx == 0: continue
            track_rows = event_data[event_data['trackIndex'] == idx]
            if identify_good_tracks(track_rows):
                good_tracks[idx] = track_rows
        
        # Pre-calculate track parameters for good tracks
        good_track_params = {}
        for t_idx, t_data in good_tracks.items():
            pt, r, sign, cx, cy = calculate_track_parameters(t_data)
            if cx is not None:
                good_track_params[t_idx] = (pt, r, sign, (cx, cy))
        
        # Identify ionization hits
        ionization_mask = event_data['process'].isin(ionization_processes)
        
        # Iterate over ionization hits
        # We use iterrows but we need to update event_data. 
        # Using index from iterrows is safe for updating event_data.
        for idx, row in event_data[ionization_mask].iterrows():
            # Calculate geometry for this hit
            hit_df = pd.DataFrame([row])
            hit_center, hit_radius = calculate_hit_geometry(hit_df)
            
            if hit_center is None or hit_radius is None:
                continue
                
            best_dist = float('inf')
            best_track_idx = None
            
            for t_idx, params in good_track_params.items():
                pt, r, sign, track_center = params
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
                
        # --- Step 2: Filter Noise Tracks ---
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
                
        processed_rows.append(event_data)
        
    # Combine
    if processed_rows:
        final_df = pd.concat(processed_rows)
    else:
        final_df = pd.DataFrame(columns=df.columns)
    
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
