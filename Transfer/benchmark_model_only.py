import time
import csv
import numpy as np
import psutil
import os
from pathlib import Path
from aggregator_model import AggregatorModel
from collections import defaultdict
import sys

def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def format_bytes(bytes_val):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def estimate_state_memory(aggregator):
    """Estimate memory used by sender state and feature buffers."""
    total_state_memory = 0
    
    # Speed model state
    if hasattr(aggregator, 'speed_model_aggregator'):
        for sender_id, state in aggregator.speed_model_aggregator.sender_state.items():
            state_size = 50 + 140 + 140 + 56
            total_state_memory += state_size
    
    # Heading model state
    if hasattr(aggregator, 'heading_model_aggregator'):
        for sender_id, state in aggregator.heading_model_aggregator.sender_state.items():
            state_size = 50 + 140 + 140 + 56
            total_state_memory += state_size
    
    # Accel model state
    if hasattr(aggregator, 'accel_model_aggregator'):
        for sender_id, state in aggregator.accel_model_aggregator.sender_state.items():
            state_size = 50 + 140 + 140 + 56
            total_state_memory += state_size
    
    return total_state_memory

def benchmark_model_with_memory():
    """Benchmark model inference latency with memory profiling."""
    
    # Record total start time and memory
    benchmark_start_time = time.time()
    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.2f} MB\n")
    
    # Load model
    print("Loading models...")
    model_load_start = time.time()
    models_path = Path("/home/kali/v2x_ids_system/models")
    aggregator = AggregatorModel(models_dir=str(models_path))
    model_load_time = time.time() - model_load_start
    
    memory_after_models = get_memory_usage()
    model_memory = memory_after_models - initial_memory
    
    print(f"Models loaded in {model_load_time:.2f} seconds")
    print(f"Memory after loading: {memory_after_models:.2f} MB (+{model_memory:.2f} MB)\n")
    
    # Load test CSV
    csv_file = "/home/kali/v2x_ids_system/Testing/Speed_processed_test_city2.csv"
    print(f"Loading test data from {csv_file}...")
    csv_load_start = time.time()
    
    latencies = []
    verdicts = defaultdict(int)
    sender_ids = set()
    row_count = 0
    
    # Memory tracking per 1000 packets
    memory_timeline = []
    packet_count_timeline = []
    
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)  # skip header
        
        for row_num, row in enumerate(csv_reader, start=2):
            if len(row) < 10:
                continue
            
            try:
                # Parse CSV
                sender_id = str(row[0])
                heading = float(row[1])
                speed = float(row[2])
                long_accel = float(row[3])
                gen_time = int(float(row[4]))
                elevation = float(row[5])
                latitude = float(row[6])
                longitude = float(row[7])
                bit_len = int(float(row[8]))
                
                # Build BSM (in-memory, no network)
                bsm = {
                    'senderId': sender_id,
                    'heading': heading,
                    'speed': speed,
                    'longAcceleration': long_accel,
                    'generationTime': gen_time,
                    'elevation': elevation,
                    'latitude': latitude,
                    'longitude': longitude,
                    'bitLen': bit_len,
                }
                
                sender_ids.add(sender_id)
                
                # ===== TIME JUST THE MODEL INFERENCE =====
                start_time = time.perf_counter()
                result = aggregator.process_bsm(bsm)  # Direct call, no network
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                verdicts[result['attack_type']] += 1
                row_count += 1
                
                # Track memory every 1000 packets
                if row_count % 1000 == 0:
                    current_memory = get_memory_usage()
                    state_memory = estimate_state_memory(aggregator)
                    memory_timeline.append({
                        'packets': row_count,
                        'total_memory': current_memory,
                        'state_memory': state_memory,
                        'delta_from_initial': current_memory - initial_memory
                    })
                    print(f"  Processed {row_count:,} packets | Memory: {current_memory:.2f} MB (+{current_memory - initial_memory:.2f} MB total, state: {format_bytes(state_memory)})")
            
            except Exception as e:
                print(f"Error on row {row_num}: {e}")
                continue
    
    csv_load_time = time.time() - csv_load_start
    
    # Final memory measurements
    final_memory = get_memory_usage()
    final_state_memory = estimate_state_memory(aggregator)
    total_memory_increase = final_memory - initial_memory
    
    # Record total end time
    benchmark_end_time = time.time()
    total_benchmark_time = benchmark_end_time - benchmark_start_time
    
    # Calculate times
    inference_total_time = sum(latencies) / 1000  # ms to seconds
    
    # ===== STATISTICS =====
    print("\n" + "=" * 100)
    print("ENHANCED MODEL BENCHMARK WITH MEMORY PROFILING")
    print("=" * 100)
    
    print(f"\n📊 DATASET:")
    print(f"  Total predictions: {len(latencies):,}")
    print(f"  Unique vehicles: {len(sender_ids)}")
    
    print(f"\n⏱️  LATENCY (milliseconds):")
    print(f"  Mean:   {np.mean(latencies):.4f} ms")
    print(f"  Median: {np.median(latencies):.4f} ms")
    print(f"  Std:    {np.std(latencies):.4f} ms")
    print(f"  Min:    {np.min(latencies):.4f} ms")
    print(f"  Max:    {np.max(latencies):.4f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.4f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.4f} ms")
    
    # Throughput
    throughput = len(latencies) / inference_total_time
    print(f"\n⚡ THROUGHPUT:")
    print(f"  {throughput:.0f} predictions/second")
    print(f"  {throughput*1000:.0f} predictions/ms")
    print(f"  {throughput/1000:.1f} predictions/microsecond")
    
    print(f"\n💾 MEMORY USAGE:")
    print(f"  Initial memory:          {initial_memory:.2f} MB")
    print(f"  Memory after models:     {memory_after_models:.2f} MB")
    print(f"  Model code size:         {model_memory:.2f} MB")
    print(f"  Final memory:            {final_memory:.2f} MB")
    print(f"  Total increase:          {total_memory_increase:.2f} MB")
    print(f"  Runtime state memory:    {format_bytes(final_state_memory)}")
    print(f"  Per-vehicle avg memory:  {format_bytes(final_state_memory / len(sender_ids) if sender_ids else 0)}")
    
    print(f"\n📈 MEMORY GROWTH OVER TIME:")
    print(f"  {'Packets':<12} {'Total Memory':<16} {'State Memory':<16} {'Delta from Init':<16}")
    print(f"  {'-' * 60}")
    for timeline in memory_timeline:
        print(f"  {timeline['packets']:<12,} {timeline['total_memory']:<16.2f} MB {format_bytes(timeline['state_memory']):<16} {timeline['delta_from_initial']:<16.2f} MB")
    
    print(f"\n🎯 VERDICTS:")
    for verdict, count in verdicts.items():
        pct = count / len(latencies) * 100
        print(f"  {verdict:<25} {count:>6,} ({pct:>5.1f}%)")
    
    print(f"\n⏳ EXECUTION TIME BREAKDOWN:")
    print(f"  Model loading:     {model_load_time:.2f} seconds")
    print(f"  CSV loading:       {csv_load_time:.2f} seconds")
    print(f"  Model inference:   {inference_total_time:.2f} seconds")
    print(f"  ─" * 50)
    print(f"  Total time:        {total_benchmark_time:.2f} seconds")
    print(f"\n  Efficiency breakdown:")
    print(f"    Model inference: {(inference_total_time/total_benchmark_time)*100:.1f}%")
    print(f"    I/O & overhead:  {(1-(inference_total_time/total_benchmark_time))*100:.1f}%")
    
    print(f"\n💡 MEMORY EFFICIENCY:")
    print(f"  Bytes per prediction: {(total_memory_increase * 1024 * 1024) / len(latencies):.2f} bytes")
    print(f"  Bytes per vehicle:    {(final_state_memory) / len(sender_ids) if sender_ids else 0:.2f} bytes")
    print(f"  Predictions per MB:   {len(latencies) / total_memory_increase:.0f}")
    
    print("\n" + "=" * 100)
    
    # Return metrics for further analysis
    return {
        'total_predictions': len(latencies),
        'unique_vehicles': len(sender_ids),
        'latency_mean_ms': np.mean(latencies),
        'latency_median_ms': np.median(latencies),
        'latency_p95_ms': np.percentile(latencies, 95),
        'latency_p99_ms': np.percentile(latencies, 99),
        'throughput_per_sec': throughput,
        'initial_memory_mb': initial_memory,
        'final_memory_mb': final_memory,
        'model_memory_mb': model_memory,
        'runtime_state_memory_bytes': final_state_memory,
        'total_memory_increase_mb': total_memory_increase,
        'model_load_time_sec': model_load_time,
        'inference_total_time_sec': inference_total_time,
        'total_benchmark_time_sec': total_benchmark_time,
    }

if __name__ == '__main__':
    results = benchmark_model_with_memory()

