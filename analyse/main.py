#!/usr/bin/env python3
"""
BESIII Track Analysis - Main Entry Point

This is the main entry point for the modular BESIII track analysis system.
It integrates all the modules: utils, track_filter, analysis, and visualization.
"""

import argparse
import sys
import os

# Import the modular functions
from utils import calculate_track_parameters, calculate_momentum_direction, angle_difference, calculate_distance_from_origin, find_closest_point_to_origin, sort_track_points_by_distance
from track_filter import filter_central_tracks, calculate_closest_distances_for_tracks
from analysis import calculate_hit_relationships, analyze_hit_relationships, analyze_closest_distance_distribution
from visualization import visualize_raw_tracks, visualize_rec_tracks, plot_closest_distance_distribution_by_process, plot_hit_relationship_distributions, analyze_distance_to_tracks_distribution


def main():
    """Main function that handles command line arguments and dispatches to appropriate modules"""
    parser = argparse.ArgumentParser(description='BESIII Track Analysis (Modular Version)')
    parser.add_argument('--run-mode', required=True, 
                        choices=['raw', 'rec', 'distance', 'hit-relationships', 'distance-to-tracks'], 
                        help='''Specify which function to run:
                        raw: Run visualize_raw_tracks (needs --input1)
                        rec: Run visualize_rec_tracks (needs --input1 and --input2)
                        distance: Analyze closest hit distance distribution (needs --input1)
                        hit-relationships: Analyze relationships between consecutive hits (needs --input1)
                        distance-to-tracks: Analyze ionization hit distance to good tracks (needs --input1)''')

    parser.add_argument('--input1', help='Input raw data CSV file')
    parser.add_argument('--input2', help='Input predicted hits CSV file')
    parser.add_argument('--output', default="display.pdf", help='Output PDF file path (default: display.pdf)')
    parser.add_argument('--events', type=int, default=10, help='Maximum number of events to display (default: 10)')
    parser.add_argument('--processes', nargs='+', help='List of processes to reassign to tracks based on momentum direction')
    parser.add_argument('--whitelist-processes', nargs='+', default=['eIoni', 'muIoni', 'hIoni'], 
                        help='List of processes that are exempt from cut1 conditions (hit<6 and layer<6)')
    parser.add_argument('--max-distance', type=float, default=80, 
                        help='Maximum distance from origin for central tracks (default: 80 cm)')
    parser.add_argument('--max-layer', type=int, default=80, 
                        help='Maximum layer number for starting point (default: 80)')
    parser.add_argument('--show-start-triangle', action='store_true', 
                        help='Show triangle marker at track start point (default: False)')
    parser.add_argument('--show-track-curve', action='store_true', 
                        help='Show track curve based on calculated parameters (default: False)')

    args = parser.parse_args()

    # Validate input arguments based on run mode
    if args.run_mode == 'raw' and args.input1 is None:
        parser.error("When --run-mode=raw, --input1 (raw data CSV) is required!")
    if args.run_mode == 'rec' and (args.input1 is None or args.input2 is None):
        parser.error("When --run-mode=rec, both --input1 (raw data) and --input2 (predicted hits) are required!")
    if args.run_mode in ['distance', 'hit-relationships', 'distance-to-tracks'] and args.input1 is None:
        parser.error(f"When --run-mode={args.run_mode}, --input1 (raw data CSV) is required!")

    # Process the process list
    process_list = args.processes if args.processes else []
    if process_list:
        print(f"Processes to reassign: {process_list}")

    # Dispatch to appropriate function based on run mode
    if args.run_mode == 'raw':
        print(f"=== Running visualize_raw_tracks ===")
        print(f"Input raw data: {args.input1}")
        print(f"Output PDF: {args.output}")
        print(f"Max events: {args.events}")
        print(f"Max distance for central tracks: {args.max_distance} cm")
        print(f"Max layer for starting point: {args.max_layer}")
        print(f"Show start triangle: {args.show_start_triangle}")
        print(f"Show track curve: {args.show_track_curve}")
        
        visualize_raw_tracks(args, process_list)

    elif args.run_mode == 'rec':
        print(f"=== Running visualize_rec_tracks ===")
        print(f"Input raw data: {args.input1}")
        print(f"Input predicted hits: {args.input2}")
        print(f"Output PDF: {args.output}")
        print(f"Max events: {args.events}")
        
        visualize_rec_tracks(args)

    elif args.run_mode == 'distance':
        print(f"=== Running closest distance analysis ===")
        print(f"Input raw data: {args.input1}")
        print(f"Output PDF: {args.output}")
        print(f"Max events: {args.events}")
        print(f"Track selection criteria: hits ≥ 6 and layers ≥ 6")
        
        analyze_closest_distance_distribution(args, process_list)

    elif args.run_mode == 'hit-relationships':
        print(f"=== Running hit relationship analysis ===")
        print(f"Input raw data: {args.input1}")
        print(f"Output PDF: {args.output}")
        print(f"Max events: {args.events}")
        
        analyze_hit_relationships(args, process_list)

    elif args.run_mode == 'distance-to-tracks':
        print(f"=== Running distance to tracks analysis ===")
        print(f"Input raw data: {args.input1}")
        print(f"Output PDF: {args.output}")
        print(f"Max events: {args.events}")
        print(f"Good track criteria: hits > 6 and layers > 6")
        print(f"Ionization processes: eIoni, hIoni, muIoni")
        
        analyze_distance_to_tracks_distribution(args)


def list_available_functions():
    """List all available functions in the modular system"""
    print("=== Available Functions in BESIII Track Analysis System ===\n")
    
    print("UTILS MODULE (基础工具函数):")
    print("  - calculate_track_parameters(track_data): 计算径迹参数（中心、半径等）")
    print("  - calculate_momentum_direction(momX, momY): 计算动量方向角度")
    print("  - angle_difference(angle1, angle2): 计算两个角度之间的最小差异")
    print("  - calculate_distance_from_origin(x, y): 计算点到原点的距离")
    print("  - find_closest_point_to_origin(track_data): 找到径迹中最接近原点的点")
    print("  - sort_track_points_by_distance(track_data): 按距离排序径迹点\n")
    
    print("TRACK_FILTER MODULE (径迹筛选函数):")
    print("  - filter_central_tracks(track_data, max_distance_from_origin, max_layer_for_origin): 筛选中央径迹")
    print("  - calculate_closest_distances_for_tracks(track_data): 计算径迹间最近距离\n")
    
    print("ANALYSIS MODULE (数据分析函数):")
    print("  - calculate_hit_relationships(track_data): 计算击中点关系")
    print("  - analyze_hit_relationships(args, process_list): 分析击中点关系分布")
    print("  - analyze_closest_distance_distribution(args, process_list): 分析最近距离分布\n")
    
    print("VISUALIZATION MODULE (可视化函数):")
    print("  - visualize_raw_tracks(args, process_list): 可视化原始径迹数据")
    print("  - visualize_rec_tracks(args): 可视化重建径迹")
    print("  - plot_closest_distance_distribution_by_process(args, process_list): 按过程绘制最近距离分布")
    print("  - plot_hit_relationship_distributions(args, process_list): 绘制击中点关系分布")
    print("  - analyze_distance_to_tracks_distribution(args): 分析电离hit点到好track曲线的距离分布\n")


if __name__ == "__main__":
    # Check if the user wants to list functions
    if len(sys.argv) > 1 and sys.argv[1] == '--list-functions':
        list_available_functions()
    else:
        main()