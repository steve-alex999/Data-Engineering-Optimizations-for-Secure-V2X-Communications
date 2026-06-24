#!/usr/bin/env python3
"""
V2X Performance Profiling - Visualization Helper
Generate charts and graphs from profiling results CSV
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
import sys


def load_data(csv_path: str, json_path: str = None):
    """Load CSV and optional JSON summary."""
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    summary = None
    if json_path and Path(json_path).exists():
        print(f"Loading summary: {json_path}")
        with open(json_path) as f:
            summary = json.load(f)
    
    return df, summary


def print_basic_stats(df: pd.DataFrame, summary: dict = None):
    """Print basic statistics."""
    print("\n" + "=" * 80)
    print("PERFORMANCE PROFILING - BASIC STATISTICS")
    print("=" * 80)
    
    # Predictions
    print(f"\n📊 DATASET:")
    print(f"  Total predictions: {len(df):,}")
    print(f"  Unique vehicles: {df['sender_id'].nunique()}")
    print(f"  Max vehicle count: {df['vehicle_count'].max()}")
    print(f"  Test duration (timestamp range): ~{(df['timestamp'].max() - df['timestamp'].min())/60:.1f} minutes")
    
    # Latencies
    print(f"\n⏱️ LATENCY (milliseconds):")
    print(f"  Layer 1 (DoS):        {df['layer1_time_ms'].mean():.4f} ± {df['layer1_time_ms'].std():.4f} ms")
    print(f"  Layer 2 Speed:        {df['layer2_speed_time_ms'].mean():.4f} ± {df['layer2_speed_time_ms'].std():.4f} ms")
    print(f"  Layer 2 Accel:        {df['layer2_accel_time_ms'].mean():.4f} ± {df['layer2_accel_time_ms'].std():.4f} ms")
    print(f"  Layer 2 Heading:      {df['layer2_heading_time_ms'].mean():.4f} ± {df['layer2_heading_time_ms'].std():.4f} ms")
    print(f"  Layer 3 (Aggreg):     {df['layer3_time_ms'].mean():.4f} ± {df['layer3_time_ms'].std():.4f} ms")
    print(f"  {'─'*50}")
    print(f"  Total (End-to-End):   {df['total_inference_time_ms'].mean():.4f} ± {df['total_inference_time_ms'].std():.4f} ms")
    print(f"    - P50 (median): {df['total_inference_time_ms'].median():.4f} ms")
    print(f"    - P95:          {df['total_inference_time_ms'].quantile(0.95):.4f} ms")
    print(f"    - P99:          {df['total_inference_time_ms'].quantile(0.99):.4f} ms")
    print(f"    - Max:          {df['total_inference_time_ms'].max():.4f} ms")
    
    # Memory
    print(f"\n💾 MEMORY:")
    print(f"  Avg buffer/prediction: {df['buffer_memory_kb'].mean():.2f} KB")
    print(f"  Max buffer:            {df['buffer_memory_kb'].max():.2f} KB")
    print(f"  Total buffer (all):    {df['buffer_memory_kb'].sum():.2f} KB")
    
    # Verdicts
    print(f"\n🎯 ATTACK DETECTION:")
    vc = df['verdict'].value_counts()
    total = len(df)
    for verdict, count in vc.items():
        pct = count / total * 100
        print(f"  {verdict:<25} {count:>6,} ({pct:>5.1f}%)")
    
    # Throughput
    if 'timestamp' in df.columns:
        try:
            time_range = pd.to_datetime(df['timestamp']).max() - pd.to_datetime(df['timestamp']).min()
            seconds = time_range.total_seconds()
            if seconds > 0:
                throughput = len(df) / seconds
                print(f"\n⚡ THROUGHPUT:")
                print(f"  {throughput:.0f} predictions/second")
                print(f"  {throughput*1000:.0f} predictions/ms")
        except:
            pass
    
    # JSON summary if available
    if summary:
        print(f"\n🔢 FLOPS (from summary):")
        flops = summary.get('flops', {})
        for model, f in flops.get('per_model', {}).items():
            print(f"  {model:<20} {f:>8,} FLOPs")
        print(f"  {'─'*50}")
        print(f"  Total per prediction: {flops.get('total_per_prediction', 0):>8,} FLOPs")
        print(f"  Total all pred's:     {flops.get('total_all_predictions', 0):>15,} FLOPs")


def print_scaling_analysis(df: pd.DataFrame):
    """Analyze scaling with vehicle count."""
    print("\n" + "=" * 80)
    print("VEHICLE SCALING ANALYSIS")
    print("=" * 80)
    
    # Group by vehicle count
    grouped = df.groupby('vehicle_count').agg({
        'total_inference_time_ms': ['mean', 'std', 'count'],
        'buffer_memory_kb': ['mean', 'max'],
        'layer1_time_ms': 'mean',
        'layer2_total_time_ms': 'mean',
    }).round(4)
    
    print(f"\n{'Vehicles':<10} {'Latency (ms)':<20} {'Std Dev':<12} {'Memory (KB)':<12} {'Predictions':<12}")
    print(f"{'─'*80}")
    
    for vehicles in sorted(df['vehicle_count'].unique()):
        subset = df[df['vehicle_count'] == vehicles]
        latency = subset['total_inference_time_ms'].mean()
        std = subset['total_inference_time_ms'].std()
        memory = subset['buffer_memory_kb'].mean()
        count = len(subset)
        
        print(f"{vehicles:<10} {latency:<20.4f} {std:<12.4f} {memory:<12.2f} {count:<12}")
    
    # Check for linear vs exponential growth
    vehicle_counts = sorted(df['vehicle_count'].unique())
    if len(vehicle_counts) > 2:
        latencies = [df[df['vehicle_count']==v]['total_inference_time_ms'].mean() 
                    for v in vehicle_counts]
        
        # Calculate growth rate
        if latencies[0] > 0:
            growth_rates = [(latencies[i+1] - latencies[i])/latencies[i]*100 
                           for i in range(len(latencies)-1)]
            avg_growth = np.mean(growth_rates)
            
            print(f"\n📈 SCALING PATTERN:")
            print(f"  Average growth per vehicle: {avg_growth:.2f}%")
            
            if avg_growth < 2:
                print(f"  ✅ LINEAR (good) - latency stable with vehicle count")
            elif avg_growth < 5:
                print(f"  ⚠️  MODERATE - some latency increase per vehicle")
            else:
                print(f"  ⚠️  EXPONENTIAL - investigate vehicle state management")


def print_layer_breakdown(df: pd.DataFrame):
    """Analyze time spent in each layer."""
    print("\n" + "=" * 80)
    print("LAYER BREAKDOWN")
    print("=" * 80)
    
    # Only where Layer 2 ran
    layer2_ran = df[df['layer2_ran'] == True]
    
    if len(layer2_ran) > 0:
        print(f"\nFor predictions where Layer 2 ran (Message Fabrication test): {len(layer2_ran):,} packets")
        
        total_avg = layer2_ran['total_inference_time_ms'].mean()
        layer1_avg = layer2_ran['layer1_time_ms'].mean()
        layer2_avg = layer2_ran['layer2_total_time_ms'].mean()
        layer3_avg = layer2_ran['layer3_time_ms'].mean()
        
        l1_pct = layer1_avg / total_avg * 100
        l2_pct = layer2_avg / total_avg * 100
        l3_pct = layer3_avg / total_avg * 100
        
        print(f"\n  Layer 1: {layer1_avg:.4f} ms ({l1_pct:>5.1f}%) {'█' * int(l1_pct/2)}")
        print(f"  Layer 2: {layer2_avg:.4f} ms ({l2_pct:>5.1f}%) {'█' * int(l2_pct/2)}")
        print(f"  Layer 3: {layer3_avg:.4f} ms ({l3_pct:>5.1f}%) {'█' * int(l3_pct/2)}")
        print(f"  {'─'*50}")
        print(f"  Total:   {total_avg:.4f} ms (100%)")
    
    # DOS only
    dos_only = df[df['layer2_ran'] == False]
    if len(dos_only) > 0:
        print(f"\nFor DOS detections (early exit): {len(dos_only):,} packets")
        print(f"  Layer 1: {dos_only['total_inference_time_ms'].mean():.4f} ms (average)")


def generate_python_plot_code(csv_path: str):
    """Generate Python code for plotting."""
    print("\n" + "=" * 80)
    print("PYTHON MATPLOTLIB CODE FOR VISUALIZATION")
    print("=" * 80)
    
    code = f"""
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('{csv_path}')

# 1. Latency distribution by layer
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Latency Distribution by Layer', fontsize=16)

axes[0,0].hist(df['layer1_time_ms'], bins=50, edgecolor='black')
axes[0,0].set_title('Layer 1 (DoS)')
axes[0,0].set_xlabel('Time (ms)')
axes[0,0].set_ylabel('Frequency')

axes[0,1].hist(df['layer2_speed_time_ms'].dropna(), bins=50, edgecolor='black')
axes[0,1].set_title('Layer 2 Speed')
axes[0,1].set_xlabel('Time (ms)')

axes[0,2].hist(df['layer2_accel_time_ms'].dropna(), bins=50, edgecolor='black')
axes[0,2].set_title('Layer 2 Accel')
axes[0,2].set_xlabel('Time (ms)')

axes[1,0].hist(df['layer2_heading_time_ms'].dropna(), bins=50, edgecolor='black')
axes[1,0].set_title('Layer 2 Heading')
axes[1,0].set_xlabel('Time (ms)')

axes[1,1].hist(df['layer3_time_ms'], bins=50, edgecolor='black')
axes[1,1].set_title('Layer 3 (Aggregation)')
axes[1,1].set_xlabel('Time (ms)')

axes[1,2].hist(df['total_inference_time_ms'], bins=50, edgecolor='black', color='red')
axes[1,2].set_title('Total Inference')
axes[1,2].set_xlabel('Time (ms)')

plt.tight_layout()
plt.show()

# 2. Latency over time
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['total_inference_time_ms'].rolling(100).mean(), label='100-pkt rolling avg')
ax.set_xlabel('Prediction #')
ax.set_ylabel('Latency (ms)')
ax.set_title('Inference Latency Over Test Duration')
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()

# 3. Memory vs vehicle count
fig, ax = plt.subplots(figsize=(10, 6))
vehicle_memory = df.groupby('vehicle_count')['buffer_memory_kb'].mean()
vehicle_memory.plot(kind='line', marker='o', ax=ax)
ax.set_xlabel('Number of Vehicles')
ax.set_ylabel('Avg Buffer Memory (KB)')
ax.set_title('Memory Scaling with Vehicle Count')
ax.grid(True, alpha=0.3)
plt.show()

# 4. Verdict distribution
fig, ax = plt.subplots(figsize=(10, 6))
df['verdict'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red', 'orange'])
ax.set_ylabel('Count')
ax.set_xlabel('Attack Type')
ax.set_title('Attack Detection Distribution')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()

# 5. Latency percentiles
fig, ax = plt.subplots(figsize=(10, 6))
percentiles = [10, 25, 50, 75, 90, 95, 99]
values = [df['total_inference_time_ms'].quantile(p/100) for p in percentiles]
ax.bar([f'P{p}' for p in percentiles], values, color='steelblue', edgecolor='black')
ax.set_ylabel('Latency (ms)')
ax.set_title('Latency Percentiles')
ax.grid(True, alpha=0.3, axis='y')
plt.show()
"""
    
    print(code)
    print("\n💡 Copy the code above and run in Jupyter/Python to generate plots")


def main():
    parser = argparse.ArgumentParser(
        description="V2X Performance Profiling - Analysis Tool",
        epilog="""
Examples:
  python analyze_profiling.py profiling_results/performance_metrics.csv
  python analyze_profiling.py -c profiling_results/performance_metrics.csv -s profiling_results/performance_summary.json
  python analyze_profiling.py -c metrics.csv --code  # Generate plotting code
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-c', '--csv', required=True,
                       help='Path to performance_metrics.csv')
    parser.add_argument('-s', '--summary',
                       help='Path to performance_summary.json')
    parser.add_argument('--code', action='store_true',
                       help='Print Python plotting code instead of analysis')
    parser.add_argument('--scaling', action='store_true',
                       help='Show vehicle scaling analysis only')
    parser.add_argument('--layers', action='store_true',
                       help='Show layer breakdown only')
    
    args = parser.parse_args()
    
    # Load data
    try:
        df, summary = load_data(args.csv, args.summary)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Generate output
    if args.code:
        generate_python_plot_code(args.csv)
    elif args.scaling:
        print_scaling_analysis(df)
    elif args.layers:
        print_layer_breakdown(df)
    else:
        # Default: all analysis
        print_basic_stats(df, summary)
        print_layer_breakdown(df)
        print_scaling_analysis(df)
        generate_python_plot_code(args.csv)


if __name__ == '__main__':
    main()
