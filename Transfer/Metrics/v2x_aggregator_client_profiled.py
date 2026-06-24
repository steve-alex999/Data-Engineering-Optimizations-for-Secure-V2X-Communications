#!/usr/bin/env python3
"""
V2X Aggregator Client - CSV Testing with Profiling Server
Compatible with profiling server - sends data same as original client.
No modifications needed - works with existing CSV format.
"""

import socket
import json
import csv
import argparse
import time
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


class AggregatorCsvClientProfiled:
    """TCP client for V2X Aggregator Server - works with profiling instrumentation."""
    
    def __init__(self, host='127.0.0.1', port=5555, csv_file='processed_test_city2.csv', fast_mode=False):
        self.host = host
        self.port = port
        self.csv_file = Path(csv_file)
        self.socket = None
        self.predictions = []
        self.actual_labels = []
        self.confidences = []
        self.fast_mode = fast_mode
        self.start_time = None
    
    def connect(self):
        """Connect to server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            if not self.fast_mode:
                print(f"✅ Connected to {self.host}:{self.port}\n")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def close(self):
        """Close connection"""
        if self.socket:
            self.socket.close()
        if not self.fast_mode:
            print("\n✅ Connection closed")
    
    def send_bsm(self, bsm):
        """Send single BSM and get verdict - OPTIMIZED for speed"""
        try:
            if not self.socket:
                self.connect()
            
            self.socket.sendall(json.dumps(bsm).encode() + b'\n')
            response = self.socket.recv(4096).decode().strip()
            
            if not response:
                # Connection lost, reconnect
                self.socket.close()
                if not self.connect():
                    return None
                self.socket.sendall(json.dumps(bsm).encode() + b'\n')
                response = self.socket.recv(4096).decode().strip()
            
            return json.loads(response)
        
        except (ConnectionResetError, BrokenPipeError, json.JSONDecodeError):
            try:
                if self.socket:
                    self.socket.close()
                if not self.connect():
                    return None
                self.socket.sendall(json.dumps(bsm).encode() + b'\n')
                response = self.socket.recv(4096).decode().strip()
                if not response:
                    return None
                return json.loads(response)
            except Exception:
                return None
    
    def ground_truth_to_label(self, ground_truth: int) -> int:
        """Convert ground truth label to classification code."""
        if ground_truth == 0:
            return 0  # NORMAL
        elif ground_truth == 1:
            return 1  # DOS
        else:  # 2, 3, 4
            return 2  # MESSAGE_FABRICATION
    
    def verdict_to_label(self, verdict: dict) -> int:
        """Convert server verdict to label code"""
        attack_type = verdict.get('attack_type', 'NORMAL')
        if attack_type == 'NORMAL':
            return 0
        elif attack_type == 'DOS':
            return 1
        else:  # MESSAGE_FABRICATION
            return 2
    
    def test_csv_file(self):
        """Load CSV and test all rows"""
        if not self.csv_file.exists():
            print(f"❌ File not found: {self.csv_file}")
            return False
        
        self.start_time = time.time()
        
        if not self.fast_mode:
            print("=" * 100)
            print(f"TESTING V2X AGGREGATOR - CSV: {self.csv_file.name}")
            print("=" * 100)
            print(f"\nColumn Format: senderId, heading, speed, longAcceleration, generationTime,")
            print(f" elevation, latitude, longitude, bitLen, isAttack\n")
        
        try:
            with open(self.csv_file, 'r') as f:
                csv_reader = csv.reader(f)
                header = next(csv_reader)  # Skip header
                
                if not self.fast_mode and len(header) != 10:
                    print(f"⚠️ Expected 10 columns, got {len(header)}")
                
                row_count = 0
                error_count = 0
                
                for row_num, row in enumerate(csv_reader, start=2):
                    if len(row) < 10:
                        error_count += 1
                        continue
                    
                    try:
                        # Parse CSV row - MINIMAL PROCESSING
                        sender_id = str(row[0])
                        heading = float(row[1])
                        speed = float(row[2])
                        long_accel = float(row[3])
                        gen_time = int(float(row[4]))
                        elevation = float(row[5])
                        latitude = float(row[6])
                        longitude = float(row[7])
                        bit_len = int(float(row[8]))
                        ground_truth_code = int(float(row[9]))
                        
                        # Convert ground truth to label
                        ground_truth_label = self.ground_truth_to_label(ground_truth_code)
                        
                        # Build BSM (exclude isAttack)
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
                        
                        # Send to server
                        verdict = self.send_bsm(bsm)
                        
                        if not verdict:
                            error_count += 1
                            continue
                        
                        # Convert verdict to label
                        predicted_label = self.verdict_to_label(verdict)
                        confidence = verdict.get('dos_confidence') or verdict.get('speed_conf') or 0.0
                        
                        # Store results
                        self.predictions.append(predicted_label)
                        self.actual_labels.append(ground_truth_label)
                        self.confidences.append(confidence)
                        
                        row_count += 1
                        
                        # OPTIMIZED: Print progress less frequently
                        if not self.fast_mode:
                            if row_count % 20 == 0:
                                y_true = np.array(self.actual_labels)
                                y_pred = np.array(self.predictions)
                                acc = accuracy_score(y_true, y_pred)
                                
                                if row_count % 100 == 0 and len(set(y_true)) > 1:
                                    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                                    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                                    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                                    elapsed = time.time() - self.start_time
                                    rate = row_count / elapsed if elapsed > 0 else 0
                                    print(f" [{row_count:>5d}/{row_num:>5d}] "
                                          f"Acc: {acc:.2%} | Prec: {precision:.4f} | "
                                          f"Rec: {recall:.4f} | F1: {f1:.4f} | "
                                          f"Rate: {rate:.1f} pkt/s")
                                else:
                                    elapsed = time.time() - self.start_time
                                    rate = row_count / elapsed if elapsed > 0 else 0
                                    print(f" [{row_count:>5d}/{row_num:>5d}] Acc: {acc:.2%} | Rate: {rate:.1f} pkt/s")
                    
                    except (ValueError, IndexError, TypeError) as e:
                        error_count += 1
                        if not self.fast_mode and error_count <= 3:
                            print(f" Row {row_num}: Parse error - {e}")
                
                elapsed = time.time() - self.start_time
                if not self.fast_mode:
                    print(f"\n✅ Testing complete! Processed {row_count} packets in {elapsed:.2f}s")
                    print(f" Rate: {row_count/elapsed:.2f} packets/second\n")
                else:
                    print(f"✅ FAST MODE: {row_count} packets in {elapsed:.2f}s ({row_count/elapsed:.2f} pkt/s)")
                
                return row_count, error_count
        
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return False
    
    def print_metrics(self, total_rows, error_count):
        """Calculate and print evaluation metrics"""
        if len(self.predictions) == 0:
            print("❌ No predictions made!")
            return
        
        print("\n" + "=" * 100)
        print("EVALUATION METRICS - V2X AGGREGATOR (3-Class Classification)")
        print("=" * 100)
        
        y_true = np.array(self.actual_labels)
        y_pred = np.array(self.predictions)
        y_proba = np.array(self.confidences)
        
        # ===== OVERALL PERFORMANCE =====
        print("\n📊 OVERALL PERFORMANCE:")
        print("-" * 100)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f" Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f" Precision: {precision_weighted:.4f}")
        print(f" Recall: {recall_weighted:.4f}")
        print(f" F1-Score: {f1_weighted:.4f}")
        
        # ===== CONFUSION MATRIX =====
        print("\n🔲 CONFUSION MATRIX:")
        print("-" * 100)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        print(f"\n Predicted NORMAL Predicted DOS Predicted MESSAGE_FAB")
        print(f"Actual NORMAL {cm[0][0]:>6d} {cm[0][1]:>6d} {cm[0][2]:>6d}")
        print(f"Actual DOS {cm[1][0]:>6d} {cm[1][1]:>6d} {cm[1][2]:>6d}")
        print(f"Actual MESSAGE_FAB {cm[2][0]:>6d} {cm[2][1]:>6d} {cm[2][2]:>6d}")
        
        # ===== ATTACK DETECTION STATISTICS =====
        print("\n🚨 ATTACK DETECTION STATISTICS:")
        print("-" * 100)
        
        dos_total = (y_true == 1).sum()
        dos_detected = ((y_pred == 1) & (y_true == 1)).sum()
        dos_missed = ((y_pred != 1) & (y_true == 1)).sum()
        dos_fp = ((y_pred == 1) & (y_true != 1)).sum()
        
        print(f"\n DOS ATTACKS:")
        print(f" Total in dataset: {dos_total}")
        print(f" Correctly detected: {dos_detected}")
        print(f" Missed: {dos_missed}")
        print(f" False alarms: {dos_fp}")
        if dos_total > 0:
            print(f" Detection rate: {dos_detected/dos_total*100:.2f}%")
        
        fab_total = (y_true == 2).sum()
        fab_detected = ((y_pred == 2) & (y_true == 2)).sum()
        fab_missed = ((y_pred != 2) & (y_true == 2)).sum()
        fab_fp = ((y_pred == 2) & (y_true != 2)).sum()
        
        print(f"\n MESSAGE_FABRICATION ATTACKS:")
        print(f" Total in dataset: {fab_total}")
        print(f" Correctly detected: {fab_detected}")
        print(f" Missed: {fab_missed}")
        print(f" False alarms: {fab_fp}")
        if fab_total > 0:
            print(f" Detection rate: {fab_detected/fab_total*100:.2f}%")
        
        total_attacks = dos_total + fab_total
        total_detected = dos_detected + fab_detected
        
        print(f"\n TOTAL ATTACKS (DOS + MESSAGE_FAB):")
        print(f" Total in dataset: {total_attacks}")
        print(f" Correctly detected: {total_detected}")
        if total_attacks > 0:
            print(f" Overall detection: {total_detected/total_attacks*100:.2f}%")
        
        # ===== PROCESSING STATISTICS =====
        print("\n⚙️ PROCESSING STATISTICS:")
        print("-" * 100)
        
        elapsed = time.time() - self.start_time
        print(f" Total rows in CSV: {total_rows}")
        print(f" Successfully processed: {len(self.predictions)}")
        print(f" Errors/skipped: {error_count}")
        if total_rows > 0:
            print(f" Success rate: {len(self.predictions)/total_rows*100:.2f}%")
        print(f" Total time: {elapsed:.2f}s")
        print(f" Throughput: {len(self.predictions)/elapsed:.2f} packets/second")


def main():
    parser = argparse.ArgumentParser(
        description='V2X Aggregator Client - CSV Testing (Profiling Compatible)',
        epilog="""
Examples:
  # Normal mode with metrics
  python3 v2x_aggregator_client_profiled.py --csv processed_test_city2.csv
  
  # Fast mode: max speed, minimal output
  python3 v2x_aggregator_client_profiled.py --csv processed_test_city2.csv --fast
  
  # Remote server
  python3 v2x_aggregator_client_profiled.py --host 192.168.1.100 --port 5555 --csv data.csv
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--host', default='127.0.0.1', 
                       help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5555, 
                       help='Server port (default: 5555)')
    parser.add_argument('--csv', required=True, 
                       help='Path to CSV file')
    parser.add_argument('--fast', action='store_true',
                       help='FAST mode: max speed, no intermediate metrics')
    
    args = parser.parse_args()
    
    client = AggregatorCsvClientProfiled(args.host, args.port, args.csv, fast_mode=args.fast)
    
    if not client.connect():
        return
    
    try:
        result = client.test_csv_file()
        if result:
            total_rows, error_count = result
            client.print_metrics(total_rows, error_count)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Testing stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        client.close()
        print("\n" + "=" * 100)
        print("✅ V2X Aggregator Testing Completed!")
        print("=" * 100)


if __name__ == '__main__':
    main()
