"""
V2X AGGREGATOR SERVER WITH PROFILING INSTRUMENTATION
Enhanced server that collects performance metrics for every prediction.

Profiling metrics collected:
- Per-layer latency (Layer 1, Layer 2 individual, Layer 3)
- Memory footprint per prediction
- Vehicle state tracking
- FLOPs calculation
"""

import socket
import json
import sys
import argparse
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from aggregator_model import AggregatorModel, AGG_LOGGER, LOG_PATH
from performance_profiler import PerformanceProfiler
import joblib


def ts():
    return datetime.now().strftime("%H:%M:%S")


def calculate_tree_flops(tree_model) -> int:
    """
    Estimate FLOPs for Decision Tree.
    
    FLOPs = (tree_depth * num_leaves) * comparisons_per_node
    Simplified: depth * leaves * 2 (assuming 2 features tested per node avg)
    """
    try:
        tree = tree_model.tree_
        num_leaves = tree.n_node_samples[tree.leaf_mask()].shape[0]
        max_depth = tree.max_depth
        # Each internal node does ~2 comparisons + arithmetic
        flops = max_depth * num_leaves * 2
        return int(flops)
    except:
        return 0


def get_model_size_mb(pkl_path: str) -> float:
    """Get size of pickled model in MB."""
    try:
        size_bytes = Path(pkl_path).stat().st_size
        return size_bytes / (1024 * 1024)
    except:
        return 0.0


class InstrumentedAggregatorModel(AggregatorModel):
    """Enhanced AggregatorModel with latency tracking."""
    
    def __init__(self, models_dir: str, profiler: PerformanceProfiler):
        super().__init__(models_dir)
        self.profiler = profiler
        self.models_path = Path(models_dir)
        
        # Initialize FLOPs for each model
        self._init_flops()
        
        # Initialize model sizes
        self._init_model_sizes()
    
    def _init_flops(self):
        """Calculate and set FLOPs for all models."""
        models = {
            "Time IDS": self.time_ids.model,
            "Speed IDS": self.speed_ids.model,
            "Accel IDS": self.accel_ids.model,
            "Heading IDS": self.heading_ids.model,
        }
        
        for name, model in models.items():
            flops = calculate_tree_flops(model)
            self.profiler.set_model_flops(name, flops)
            if flops > 0:
                print(f"[{ts()}] FLOPs calculated: {name} = {flops:,}")
    
    def _init_model_sizes(self):
        """Load model sizes."""
        model_files = {
            "Time IDS": "bsm_dos_model.pkl",
            "Speed IDS": "bsm_speed_model.pkl",
            "Accel IDS": "bsm_acceleration_model.pkl",
            "Heading IDS": "bsm_heading_model.pkl",
        }
        
        total_size = 0
        for name, filename in model_files.items():
            path = self.models_path / filename
            if path.exists():
                size = get_model_size_mb(str(path))
                self.profiler.set_model_memory(name, size)
                total_size += size
                print(f"[{ts()}] Model size: {name} = {size:.2f} MB")
    
    def process_bsm_instrumented(self, bsm: dict):
        """
        Process BSM with detailed latency and memory tracking.
        
        Returns: (result_dict, metrics_dict)
        """
        sender_id = bsm.get("senderId", "unknown")
        metrics = {
            "layer1_time_ms": 0,
            "layer2_speed_time_ms": 0,
            "layer2_accel_time_ms": 0,
            "layer2_heading_time_ms": 0,
            "layer3_time_ms": 0,
            "total_time_ms": 0,
            "buffer_memory_kb": 0.0,
        }
        
        # Start timing and memory tracking
        start_time = time.time_ns()
        tracemalloc.start()
        initial_mem = tracemalloc.get_traced_memory()[0] / 1024  # KB
        
        # ========== LAYER 1: Time-based IDS ==========
        layer1_start = time.time_ns()
        time_result = self.time_ids.predict(bsm)
        layer1_end = time.time_ns()
        metrics["layer1_time_ms"] = (layer1_end - layer1_start) / 1_000_000
        
        if time_result["is_attack"]:
            # DOS detected - skip Layer 2
            result = {
                "senderId": sender_id,
                "attack_type": "DOS",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "dos_confidence": float(time_result["confidence"]),
                "speed_attack": None,
                "accel_attack": None,
                "heading_attack": None,
                "speed_conf": None,
                "accel_conf": None,
                "heading_conf": None,
            }
            
            # Layer 3 for DOS (just passthrough)
            layer3_start = time.time_ns()
            layer3_end = time.time_ns()
            metrics["layer3_time_ms"] = (layer3_end - layer3_start) / 1_000_000
            
            total_end = time.time_ns()
            metrics["total_time_ms"] = (total_end - start_time) / 1_000_000
            
        else:
            # ========== LAYER 2: Parallel Fabrication Detectors ==========
            layer2_speed_start = time.time_ns()
            speed_result = self.speed_ids.predict(bsm)
            layer2_speed_end = time.time_ns()
            metrics["layer2_speed_time_ms"] = (layer2_speed_end - layer2_speed_start) / 1_000_000
            
            layer2_accel_start = time.time_ns()
            accel_result = self.accel_ids.predict(bsm)
            layer2_accel_end = time.time_ns()
            metrics["layer2_accel_time_ms"] = (layer2_accel_end - layer2_accel_start) / 1_000_000
            
            layer2_heading_start = time.time_ns()
            heading_result = self.heading_ids.predict(bsm)
            layer2_heading_end = time.time_ns()
            metrics["layer2_heading_time_ms"] = (layer2_heading_end - layer2_heading_start) / 1_000_000
            
            # ========== LAYER 3: Consensus Decision ==========
            layer3_start = time.time_ns()
            
            speed_attack = speed_result["is_attack"]
            accel_attack = accel_result["is_attack"]
            heading_attack = heading_result["is_attack"]
            
            if speed_attack and accel_attack and heading_attack:
                attack_type = "MESSAGE_FABRICATION"
            else:
                attack_type = "NORMAL"
            
            layer3_end = time.time_ns()
            metrics["layer3_time_ms"] = (layer3_end - layer3_start) / 1_000_000
            
            result = {
                "senderId": sender_id,
                "attack_type": attack_type,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "dos_confidence": None,
                "speed_attack": bool(speed_attack),
                "accel_attack": bool(accel_attack),
                "heading_attack": bool(heading_attack),
                "speed_conf": float(speed_result["confidence"]),
                "accel_conf": float(accel_result["confidence"]),
                "heading_conf": float(heading_result["confidence"]),
            }
            
            total_end = time.time_ns()
            metrics["total_time_ms"] = (total_end - start_time) / 1_000_000
        
        # Memory tracking
        current_mem = tracemalloc.get_traced_memory()[0] / 1024  # KB
        buffer_mem = max(0, current_mem - initial_mem)
        metrics["buffer_memory_kb"] = buffer_mem
        tracemalloc.stop()
        
        # Track vehicle state memory
        vehicle_state_memory = self._estimate_vehicle_memory(sender_id)
        self.profiler.update_vehicle_memory(sender_id, vehicle_state_memory)
        
        # Update peak memory
        self.profiler.update_peak_memory()
        
        return result, metrics
    
    def _estimate_vehicle_memory(self, sender_id: str) -> int:
        """
        Estimate total memory used by vehicle state histories.
        Sums memory from all micro-IDS vehicle states.
        """
        total_bytes = 0
        
        # Time IDS vehicle state
        if sender_id in self.time_ids.vehicle_states:
            state = self.time_ids.vehicle_states[sender_id]
            total_bytes += sys.getsizeof(state)
        
        # Speed IDS vehicle state
        if sender_id in self.speed_ids.vehicle_states:
            state = self.speed_ids.vehicle_states[sender_id]
            total_bytes += sys.getsizeof(state)
            # History buffers
            if "speed_history" in state:
                total_bytes += sys.getsizeof(state["speed_history"])
                total_bytes += len(state["speed_history"]) * 8  # float64
            if "accel_history" in state:
                total_bytes += sys.getsizeof(state["accel_history"])
                total_bytes += len(state["accel_history"]) * 8
        
        # Heading IDS vehicle state
        if sender_id in self.heading_ids.vehicle_states:
            state = self.heading_ids.vehicle_states[sender_id]
            total_bytes += sys.getsizeof(state)
            if "heading_history" in state:
                total_bytes += sys.getsizeof(state["heading_history"])
                total_bytes += len(state["heading_history"]) * 8
            if "heading_rate_history" in state:
                total_bytes += sys.getsizeof(state["heading_rate_history"])
                total_bytes += len(state["heading_rate_history"]) * 8
        
        # Accel IDS vehicle state
        if sender_id in self.accel_ids.vehicle_states:
            state = self.accel_ids.vehicle_states[sender_id]
            total_bytes += sys.getsizeof(state)
            if "accel_history" in state:
                total_bytes += sys.getsizeof(state["accel_history"])
                total_bytes += len(state["accel_history"]) * 8
            if "speed_history" in state:
                total_bytes += sys.getsizeof(state["speed_history"])
                total_bytes += len(state["speed_history"]) * 8
        
        return total_bytes


class AggregatorServerWithProfiling:
    """Server with built-in performance profiling."""
    
    def __init__(self, models_dir: str, host: str = "0.0.0.0", port: int = 5555,
                 output_dir: str = "./profiling_results"):
        self.host = host
        self.port = port
        self.models_dir = models_dir
        self.server_socket = None
        self.running = False
        
        # Initialize profiler
        self.profiler = PerformanceProfiler(output_dir=output_dir)
        
        # Initialize instrumented model
        self.aggregator = InstrumentedAggregatorModel(models_dir, self.profiler)
        
        self.stats = {"total_packets": 0, "total_alerts": 0}
    
    def start(self):
        """Start the server."""
        print("=" * 100)
        print("V2X 3-LAYER AGGREGATOR SERVER WITH PROFILING")
        print("=" * 100)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Models dir: {self.models_dir}")
        print(f"Profiling output: {self.profiler.output_dir}")
        print()
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True
        
        print(f"[{ts()}] ✓ Listening on {self.host}:{self.port}")
        print(f"[{ts()}] ✓ Ready for client connections...\n")
        
        try:
            while self.running:
                client_socket, addr = self.server_socket.accept()
                print(f"[{ts()}] [CONNECT] Client from {addr[0]}:{addr[1]}")
                self.handle_client(client_socket, addr)
        except KeyboardInterrupt:
            print(f"\n[{ts()}] Server interrupted by user")
        finally:
            self.stop()
    
    def handle_client(self, client_socket, addr):
        """Handle single client connection with profiling."""
        buffer = ""
        packets_count = 0
        alerts_count = 0
        
        try:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                
                buffer += data.decode("utf-8", errors="ignore")
                
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        bsm = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # Process with profiling
                    result, metrics = self.aggregator.process_bsm_instrumented(bsm)
                    packets_count += 1
                    self.stats["total_packets"] += 1
                    
                    # Record metrics
                    self.profiler.record_prediction(
                        sender_id=result.get("senderId"),
                        timestamp=time.time(),
                        layer1_time_ms=metrics["layer1_time_ms"],
                        layer2_speed_time_ms=metrics["layer2_speed_time_ms"],
                        layer2_accel_time_ms=metrics["layer2_accel_time_ms"],
                        layer2_heading_time_ms=metrics["layer2_heading_time_ms"],
                        layer3_time_ms=metrics["layer3_time_ms"],
                        total_time_ms=metrics["total_time_ms"],
                        verdict=result,
                        layer2_ran=(result.get("attack_type") != "DOS"),
                        memory_footprint_kb=metrics["buffer_memory_kb"]
                    )
                    
                    # Send response
                    resp = json.dumps(result) + "\n"
                    client_socket.sendall(resp.encode("utf-8"))
                    
                    # Log alerts
                    if result.get("attack_type") != "NORMAL":
                        alerts_count += 1
                        self.stats["total_alerts"] += 1
                        
                        # Progress update every 100 packets
                        if packets_count % 100 == 0:
                            print(f"[{ts()}] Progress: {packets_count} packets, {alerts_count} alerts")
                
        except Exception as e:
            print(f"[{ts()}] [ERROR] {addr[0]}:{addr[1]} - {e}")
        finally:
            client_socket.close()
            print(f"[{ts()}] [DISCONNECT] {addr[0]}:{addr[1]} "
                  f"({packets_count} packets, {alerts_count} alerts)")
    
    def stop(self):
        """Stop the server and export results."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        
        print(f"\n[{ts()}] Server stopped")
        print(f"[{ts()}] Statistics: {self.stats['total_packets']} packets, "
              f"{self.stats['total_alerts']} alerts")
        
        # Export results
        print(f"\n[{ts()}] Exporting results...")
        self.profiler.export_to_csv("performance_metrics.csv")
        self.profiler.export_summary_json("performance_summary.json")
        self.profiler.print_report()


def main():
    parser = argparse.ArgumentParser(
        description="V2X 3-Layer Aggregator Server with Performance Profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v2x_aggregator_server_profiling.py --models /path/to/models --port 5555
  python v2x_aggregator_server_profiling.py --models ./models --output ./results
        """,
    )
    
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5555, 
                       help="Port to listen (default: 5555)")
    parser.add_argument("--models", required=True,
                       help="Path to models directory")
    parser.add_argument("--output", default="./profiling_results",
                       help="Output directory for profiling results (default: ./profiling_results)")
    
    args = parser.parse_args()
    
    server = AggregatorServerWithProfiling(
        models_dir=args.models,
        host=args.host,
        port=args.port,
        output_dir=args.output,
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()
    except Exception as e:
        print(f"[{ts()}] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
