"""
V2X Aggregator Performance Profiler
Comprehensive metrics collection:
- Per-prediction latency (each layer + total)
- Memory usage (per vehicle state, buffers, models)
- FLOPs estimation (Decision Tree based on depth/leaves)
- Vehicle scaling analysis
"""

import time
import json
import psutil
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np


class PerformanceProfiler:
    """Collect and aggregate performance metrics for all predictions."""
    
    def __init__(self, output_dir: str = "./profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-prediction metrics
        self.predictions = []  # List of dicts with all metrics
        
        # Aggregated statistics
        self.latency_stats = {
            "layer1_time": [],
            "layer2_speed_time": [],
            "layer2_accel_time": [],
            "layer2_heading_time": [],
            "layer3_time": [],
            "total_time": [],
        }
        
        # Memory tracking
        self.memory_stats = {
            "model_loading_mb": 0,
            "per_vehicle_state_bytes": defaultdict(int),
            "peak_process_memory_mb": 0,
            "avg_buffer_memory_kb": [],
        }
        
        # Vehicle tracking
        self.vehicle_ids = set()
        self.vehicle_prediction_count = defaultdict(int)
        
        # FLOPs tracking
        self.flops_per_model = {}
        
        # Process info for memory tracking
        self.process = psutil.Process(os.getpid())
        self.initial_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
    def record_prediction(self, 
                         sender_id: str,
                         timestamp: float,
                         layer1_time_ms: float,
                         layer2_speed_time_ms: float,
                         layer2_accel_time_ms: float,
                         layer2_heading_time_ms: float,
                         layer3_time_ms: float,
                         total_time_ms: float,
                         verdict: dict,
                         layer2_ran: bool = True,
                         memory_footprint_kb: float = 0.0):
        """Record metrics for a single prediction."""
        
        # Track vehicles
        self.vehicle_ids.add(sender_id)
        self.vehicle_prediction_count[sender_id] += 1
        
        # Calculate total layer2 time
        layer2_time_ms = (layer2_speed_time_ms + layer2_accel_time_ms + 
                         layer2_heading_time_ms if layer2_ran else 0)
        
        # Store prediction record
        record = {
            "timestamp": timestamp,
            "sender_id": sender_id,
            "vehicle_count": len(self.vehicle_ids),
            "prediction_sequence": len(self.predictions) + 1,
            # Latencies (milliseconds)
            "layer1_time_ms": layer1_time_ms,
            "layer2_speed_time_ms": layer2_speed_time_ms,
            "layer2_accel_time_ms": layer2_accel_time_ms,
            "layer2_heading_time_ms": layer2_heading_time_ms,
            "layer2_total_time_ms": layer2_time_ms,
            "layer3_time_ms": layer3_time_ms,
            "total_inference_time_ms": total_time_ms,
            "layer2_ran": layer2_ran,
            # Memory (KB)
            "buffer_memory_kb": memory_footprint_kb,
            # Verdict info
            "verdict": verdict.get("attack_type", "UNKNOWN"),
        }
        
        self.predictions.append(record)
        
        # Update statistics
        self.latency_stats["layer1_time"].append(layer1_time_ms)
        if layer2_ran:
            self.latency_stats["layer2_speed_time"].append(layer2_speed_time_ms)
            self.latency_stats["layer2_accel_time"].append(layer2_accel_time_ms)
            self.latency_stats["layer2_heading_time"].append(layer2_heading_time_ms)
        self.latency_stats["layer3_time"].append(layer3_time_ms)
        self.latency_stats["total_time"].append(total_time_ms)
        
        self.memory_stats["avg_buffer_memory_kb"].append(memory_footprint_kb)
    
    def set_model_flops(self, model_name: str, flops: int):
        """Set FLOPs for a model (computed from tree depth/leaves)."""
        self.flops_per_model[model_name] = flops
    
    def set_model_memory(self, model_name: str, size_mb: float):
        """Set memory footprint for loaded model."""
        self.memory_stats["model_loading_mb"] += size_mb
    
    def update_vehicle_memory(self, sender_id: str, state_bytes: int):
        """Update memory used by vehicle state (histories, etc)."""
        self.memory_stats["per_vehicle_state_bytes"][sender_id] = state_bytes
    
    def update_peak_memory(self):
        """Update peak memory usage during test."""
        current_mb = self.process.memory_info().rss / (1024 * 1024)
        if current_mb > self.memory_stats["peak_process_memory_mb"]:
            self.memory_stats["peak_process_memory_mb"] = current_mb
    
    def get_latency_summary(self) -> Dict[str, Any]:
        """Get summary statistics for latencies."""
        summary = {}
        
        for layer, times in self.latency_stats.items():
            if times:
                summary[layer] = {
                    "mean_ms": float(np.mean(times)),
                    "median_ms": float(np.median(times)),
                    "std_ms": float(np.std(times)),
                    "min_ms": float(np.min(times)),
                    "max_ms": float(np.max(times)),
                    "p95_ms": float(np.percentile(times, 95)),
                    "p99_ms": float(np.percentile(times, 99)),
                }
        
        return summary
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary statistics for memory."""
        vehicle_states = list(self.memory_stats["per_vehicle_state_bytes"].values())
        buffer_memory = self.memory_stats["avg_buffer_memory_kb"]
        
        summary = {
            "total_model_size_mb": self.memory_stats["model_loading_mb"],
            "peak_process_memory_mb": self.memory_stats["peak_process_memory_mb"],
            "memory_growth_mb": (self.memory_stats["peak_process_memory_mb"] - 
                                self.initial_memory_mb),
            "num_vehicles_tracked": len(self.vehicle_ids),
        }
        
        if vehicle_states:
            summary["per_vehicle_state"] = {
                "total_bytes": int(sum(vehicle_states)),
                "mean_bytes": float(np.mean(vehicle_states)),
                "median_bytes": float(np.median(vehicle_states)),
                "min_bytes": int(np.min(vehicle_states)),
                "max_bytes": int(np.max(vehicle_states)),
                "std_bytes": float(np.std(vehicle_states)),
            }
        
        if buffer_memory:
            summary["buffer_memory"] = {
                "mean_kb": float(np.mean(buffer_memory)),
                "median_kb": float(np.median(buffer_memory)),
                "max_kb": float(np.max(buffer_memory)),
                "total_kb": float(sum(buffer_memory)),
            }
        
        return summary
    
    def get_flops_summary(self) -> Dict[str, Any]:
        """Get FLOPs summary."""
        return {
            "per_model": self.flops_per_model,
            "total_per_prediction": sum(self.flops_per_model.values()),
            "total_all_predictions": (sum(self.flops_per_model.values()) * 
                                     len(self.predictions)),
        }
    
    def get_vehicle_scaling(self) -> Dict[str, Any]:
        """Analyze how metrics scale with number of vehicles."""
        vehicle_counts = [p["vehicle_count"] for p in self.predictions]
        total_times = [p["total_inference_time_ms"] for p in self.predictions]
        
        if not vehicle_counts or not total_times:
            return {}
        
        # Group by vehicle count
        by_vehicle_count = defaultdict(list)
        for vc, tt in zip(vehicle_counts, total_times):
            by_vehicle_count[vc].append(tt)
        
        # Calculate mean latency per vehicle count
        scaling_data = {}
        for vc in sorted(by_vehicle_count.keys()):
            times = by_vehicle_count[vc]
            scaling_data[str(vc)] = {
                "mean_ms": float(np.mean(times)),
                "std_ms": float(np.std(times)),
                "count": len(times),
            }
        
        return scaling_data
    
    def export_to_csv(self, filename: str = "performance_metrics.csv"):
        """Export all prediction metrics to CSV."""
        if not self.predictions:
            print("⚠️  No predictions recorded")
            return None
        
        csv_path = self.output_dir / filename
        
        # Determine columns
        columns = list(self.predictions[0].keys())
        
        # Write CSV
        with open(csv_path, 'w') as f:
            # Header
            f.write(",".join(columns) + "\n")
            
            # Data rows
            for pred in self.predictions:
                row = []
                for col in columns:
                    value = pred[col]
                    if isinstance(value, dict):
                        value = json.dumps(value)
                    row.append(str(value))
                f.write(",".join(row) + "\n")
        
        print(f"✅ Exported {len(self.predictions)} predictions to {csv_path}")
        return csv_path
    
    def export_summary_json(self, filename: str = "performance_summary.json"):
        """Export comprehensive summary to JSON."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": len(self.predictions),
            "num_unique_vehicles": len(self.vehicle_ids),
            "latency": self.get_latency_summary(),
            "memory": self.get_memory_summary(),
            "flops": self.get_flops_summary(),
            "vehicle_scaling": self.get_vehicle_scaling(),
        }
        
        json_path = self.output_dir / filename
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Exported summary to {json_path}")
        return json_path
    
    def print_report(self):
        """Print formatted report to console."""
        print("\n" + "=" * 100)
        print("V2X AGGREGATOR PERFORMANCE PROFILING REPORT")
        print("=" * 100)
        
        # Predictions
        print(f"\n📊 PREDICTION STATISTICS:")
        print(f"  Total predictions: {len(self.predictions)}")
        print(f"  Unique vehicles: {len(self.vehicle_ids)}")
        
        # Latencies
        latency_summary = self.get_latency_summary()
        if latency_summary:
            print(f"\n⏱️  LATENCY SUMMARY (milliseconds):")
            print(f"  {'Layer':<25} {'Mean':<10} {'Median':<10} {'Std':<10} {'P95':<10} {'P99':<10}")
            print(f"  {'-'*75}")
            for layer, stats in latency_summary.items():
                print(f"  {layer:<25} {stats['mean_ms']:<10.4f} {stats['median_ms']:<10.4f} "
                      f"{stats['std_ms']:<10.4f} {stats['p95_ms']:<10.4f} {stats['p99_ms']:<10.4f}")
        
        # Memory
        memory_summary = self.get_memory_summary()
        if memory_summary:
            print(f"\n💾 MEMORY SUMMARY:")
            print(f"  Total model size: {memory_summary['total_model_size_mb']:.2f} MB")
            print(f"  Peak process memory: {memory_summary['peak_process_memory_mb']:.2f} MB")
            print(f"  Memory growth: {memory_summary['memory_growth_mb']:.2f} MB")
            print(f"  Vehicles tracked: {memory_summary['num_vehicles_tracked']}")
            
            if "per_vehicle_state" in memory_summary:
                pv = memory_summary["per_vehicle_state"]
                print(f"\n  Per-Vehicle State:")
                print(f"    Total: {pv['total_bytes']} bytes")
                print(f"    Mean: {pv['mean_bytes']:.0f} bytes/vehicle")
                print(f"    Median: {pv['median_bytes']:.0f} bytes/vehicle")
                print(f"    Max: {pv['max_bytes']} bytes/vehicle")
            
            if "buffer_memory" in memory_summary:
                bm = memory_summary["buffer_memory"]
                print(f"\n  Buffer Memory (per prediction):")
                print(f"    Mean: {bm['mean_kb']:.2f} KB")
                print(f"    Median: {bm['median_kb']:.2f} KB")
                print(f"    Max: {bm['max_kb']:.2f} KB")
        
        # FLOPs
        flops_summary = self.get_flops_summary()
        if flops_summary and flops_summary["per_model"]:
            print(f"\n🔢 FLOPs ESTIMATION:")
            print(f"  Per-Model FLOPs:")
            for model, flops in flops_summary["per_model"].items():
                print(f"    {model}: {flops:,} FLOPs")
            print(f"  Total per prediction: {flops_summary['total_per_prediction']:,} FLOPs")
            if len(self.predictions) > 0:
                print(f"  Total all predictions: {flops_summary['total_all_predictions']:,} FLOPs")
        
        # Vehicle Scaling
        vehicle_scaling = self.get_vehicle_scaling()
        if vehicle_scaling:
            print(f"\n📈 VEHICLE SCALING ANALYSIS:")
            print(f"  {'Vehicles':<15} {'Mean Latency (ms)':<20} {'Std Dev':<15} {'Count':<10}")
            print(f"  {'-'*60}")
            for vc, stats in vehicle_scaling.items():
                print(f"  {vc:<15} {stats['mean_ms']:<20.4f} {stats['std_ms']:<15.4f} {stats['count']:<10}")
        
        print("\n" + "=" * 100)
