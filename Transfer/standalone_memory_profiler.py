import sys
import os
import time
import csv
import numpy as np
import psutil
from pathlib import Path
from aggregator_model import AggregatorModel
from collections import defaultdict
import gc

class MemoryProfiler:
    """Profile memory usage of aggregator_model directly."""
    
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.aggregator = None
        self.process = psutil.Process(os.getpid())
        self.metrics = {
            'timestamp': [],
            'packets_processed': [],
            'unique_vehicles': [],
            'total_memory_mb': [],
            'state_memory_bytes': [],
            'per_vehicle_memory_bytes': [],
        }
    
    def get_memory_mb(self):
        """Get current process memory in MB (RSS = actual RAM used)."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def calculate_state_memory(self):
        """
        Calculate actual memory used by vehicle state in all models.
        Returns bytes.
        
        FIXED: Use correct attribute names from AggregatorModel
        """
        total_state_bytes = 0
        
        # Speed model IDS (FIXED: was speed_model_aggregator)
        if hasattr(self.aggregator, 'speed_ids'):
            for sender_id, state in self.aggregator.speed_ids.vehicle_states.items():
                total_state_bytes += 386
        
        # Heading model IDS (FIXED: was heading_model_aggregator)
        if hasattr(self.aggregator, 'heading_ids'):
            for sender_id, state in self.aggregator.heading_ids.vehicle_states.items():
                total_state_bytes += 386
        
        # Accel model IDS (FIXED: was accel_model_aggregator)
        if hasattr(self.aggregator, 'accel_ids'):
            for sender_id, state in self.aggregator.accel_ids.vehicle_states.items():
                total_state_bytes += 386
        
        # Time model IDS (added for complete tracking)
        if hasattr(self.aggregator, 'time_ids'):
            for sender_id, state in self.aggregator.time_ids.vehicle_states.items():
                total_state_bytes += 200
        
        return total_state_bytes
    
    def count_unique_vehicles(self):
        """Count unique vehicles across all models. FIXED: correct attribute names"""
        vehicles = set()
        
        if hasattr(self.aggregator, 'speed_ids'):
            vehicles.update(self.aggregator.speed_ids.vehicle_states.keys())
        
        if hasattr(self.aggregator, 'heading_ids'):
            vehicles.update(self.aggregator.heading_ids.vehicle_states.keys())
        
        if hasattr(self.aggregator, 'accel_ids'):
            vehicles.update(self.aggregator.accel_ids.vehicle_states.keys())
        
        if hasattr(self.aggregator, 'time_ids'):
            vehicles.update(self.aggregator.time_ids.vehicle_states.keys())
        
        return len(vehicles)
    
    def load_models(self):
        """Load aggregator models and record memory."""
        print("Loading models...")
        gc.collect()
        initial_mem = self.get_memory_mb()
        
        self.aggregator = AggregatorModel(models_dir=self.models_dir)
        
        gc.collect()
        final_mem = self.get_memory_mb()
        
        print(f"✅ Models loaded: {initial_mem:.2f} MB → {final_mem:.2f} MB (model size: +{final_mem - initial_mem:.2f} MB)")
        return initial_mem, final_mem
    
    def process_packet(self, bsm):
        """Process single BSM and return latency."""
        start_time = time.perf_counter()
        result = self.aggregator.process_bsm(bsm)
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000
    
    def run_benchmark(self, csv_file, sample_interval=1000):
        """
        Run benchmark and collect memory metrics.
        
        Args:
            csv_file: Path to test CSV
            sample_interval: Record memory every N packets (default 1000)
        
        Returns:
            Dictionary of metrics
        """
        print(f"\nProcessing {csv_file}...")
        
        latencies = []
        verdicts = defaultdict(int)
        packets_processed = 0
        
        initial_memory = self.get_memory_mb()
        
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            
            for row_num, row in enumerate(csv_reader, start=2):
                if len(row) < 10:
                    continue
                
                try:
                    sender_id = str(row[0])
                    heading = float(row[1])
                    speed = float(row[2])
                    long_accel = float(row[3])
                    gen_time = int(float(row[4]))
                    elevation = float(row[5])
                    latitude = float(row[6])
                    longitude = float(row[7])
                    bit_len = int(float(row[8]))
                    
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
                    
                    latency_ms = self.process_packet(bsm)
                    latencies.append(latency_ms)
                    verdicts[sender_id] += 1
                    packets_processed += 1
                    
                    if packets_processed % sample_interval == 0:
                        gc.collect()
                        current_memory = self.get_memory_mb()
                        state_memory = self.calculate_state_memory()
                        unique_vehicles = self.count_unique_vehicles()
                        
                        self.metrics['timestamp'].append(packets_processed)
                        self.metrics['packets_processed'].append(packets_processed)
                        self.metrics['unique_vehicles'].append(unique_vehicles)
                        self.metrics['total_memory_mb'].append(current_memory)
                        self.metrics['state_memory_bytes'].append(state_memory)
                        
                        per_vehicle = state_memory / unique_vehicles if unique_vehicles > 0 else 0
                        self.metrics['per_vehicle_memory_bytes'].append(per_vehicle)
                        
                        print(f"  {packets_processed:,} packets | {unique_vehicles} vehicles | "
                              f"Memory: {current_memory:.2f} MB | State: {state_memory/1024:.2f} KB | "
                              f"Per-vehicle: {per_vehicle:.0f} bytes")
                
                except Exception as e:
                    print(f"Error on row {row_num}: {e}")
                    continue
        
        return {
            'packets_processed': packets_processed,
            'unique_vehicles': self.count_unique_vehicles(),
            'latencies': latencies,
            'verdicts': verdicts,
            'initial_memory': initial_memory,
            'final_memory': self.get_memory_mb(),
            'state_memory': self.calculate_state_memory(),
        }
    
    def print_report(self, results):
        """Print comprehensive memory report. FIXED: zero-division check"""
        
        print("\n" + "=" * 100)
        print("MEMORY PROFILING REPORT (Direct from aggregator_model)")
        print("=" * 100)
        
        print(f"\n📊 DATASET:")
        print(f"  Total packets processed: {results['packets_processed']:,}")
        print(f"  Unique vehicles tracked: {results['unique_vehicles']}")
        
        print(f"\n⏱️  LATENCY METRICS (milliseconds):")
        print(f"  Mean:    {np.mean(results['latencies']):.4f} ms")
        print(f"  Median:  {np.median(results['latencies']):.4f} ms")
        print(f"  Std Dev: {np.std(results['latencies']):.4f} ms")
        print(f"  P95:     {np.percentile(results['latencies'], 95):.4f} ms")
        print(f"  P99:     {np.percentile(results['latencies'], 99):.4f} ms")
        
        print(f"\n💾 MEMORY METRICS:")
        print(f"  Initial memory (before benchmark): {results['initial_memory']:.2f} MB")
        print(f"  Final memory (after benchmark):   {results['final_memory']:.2f} MB")
        print(f"  Total increase:                    {results['final_memory'] - results['initial_memory']:.2f} MB")
        print(f"  Runtime state memory:              {results['state_memory'] / 1024:.2f} KB")
        
        # FIXED: Zero-division check
        if results['unique_vehicles'] > 0:
            per_vehicle = results['state_memory'] / results['unique_vehicles']
            print(f"  Per-vehicle state:                 {per_vehicle:.0f} bytes")
        else:
            print(f"  Per-vehicle state:                 N/A (no vehicles tracked)")
        
        print(f"\n📈 MEMORY TIMELINE:")
        print(f"  {'Packets':<12} {'Vehicles':<12} {'Total Memory':<16} {'State Memory':<16} {'Per-Vehicle':<16}")
        print(f"  {'-' * 72}")
        
        for i, packets in enumerate(self.metrics['packets_processed']):
            print(f"  {packets:<12,} {self.metrics['unique_vehicles'][i]:<12} "
                  f"{self.metrics['total_memory_mb'][i]:<16.2f} MB "
                  f"{self.metrics['state_memory_bytes'][i]/1024:<16.2f} KB "
                  f"{self.metrics['per_vehicle_memory_bytes'][i]:<16.0f} bytes")
        
        print(f"\n💡 MEMORY EFFICIENCY:")
        per_prediction = (results['final_memory'] - results['initial_memory']) * 1024 * 1024 / results['packets_processed']
        print(f"  Bytes per prediction: {per_prediction:.2f} bytes")
        
        if results['unique_vehicles'] > 0:
            print(f"  Bytes per vehicle:    {results['state_memory'] / results['unique_vehicles']:.0f} bytes")
        print(f"  Predictions per MB:   {results['packets_processed'] / (results['final_memory'] - results['initial_memory']):.0f}")
        
        print(f"\n🎯 TOP 10 VEHICLES BY PACKET COUNT:")
        for vehicle, count in sorted(results['verdicts'].items(), key=lambda x: -x[1])[:10]:
            pct = count / results['packets_processed'] * 100
            print(f"  Vehicle {vehicle:<20} {count:>6,} packets ({pct:>5.1f}%)")
        
        print("\n" + "=" * 100)
        
        # Scalability estimates
        if results['unique_vehicles'] > 0:
            print(f"\n📊 SCALABILITY ESTIMATES (based on {results['unique_vehicles']} vehicles):")
            state_per_vehicle = results['state_memory'] / results['unique_vehicles']
            
            for n_vehicles in [100, 1000, 10000, 100000]:
                estimated_state = (state_per_vehicle * n_vehicles) / 1024 / 1024
                print(f"  {n_vehicles:>6,} vehicles: ~{estimated_state:.2f} MB state memory")
        
        print("\n" + "=" * 100)

def main():
    """Run standalone memory profiler."""
    
    models_dir = "/home/kali/v2x_ids_system/models"
    csv_file = "/home/kali/v2x_ids_system/Testing/Speed_processed_test_city2.csv"
    
    profiler = MemoryProfiler(models_dir)
    
    # Load models
    profiler.load_models()
    
    # Run benchmark
    results = profiler.run_benchmark(csv_file, sample_interval=1000)
    
    # Print report
    profiler.print_report(results)

if __name__ == '__main__':
    main()

