#!/usr/bin/env python3
"""
Acceleration Fabrication microIDS Inference Server - WITH THRESHOLD TUNING
Loads trained model and listens for BSM packets on port 9999
Detects: Jerk/Acceleration attacks
UPDATED: Decision threshold = 0.75 to balance precision/recall
"""

import socket
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AccelerationMicroIDS')

class AccelerationMicroIDSServer:
    def __init__(self, model_dir='./', port=9999, decision_threshold=0.75):
        self.model_dir = Path(model_dir)
        self.port = port
        self.decision_threshold = decision_threshold  # ← NEW: Adjustable threshold
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.metadata = None
        self.vehicle_states = defaultdict(self._init_vehicle_state)
        self.stats = {'total': 0, 'attacks': 0}
        
    def _init_vehicle_state(self):
        """Initialize state for new vehicle"""
        return {
            'last_accel': None,
            'last_speed': None,
            'last_time': None,
            'accel_history': [],
            'speed_history': [],
            'recent_verdicts': [],
        }
        
    def load_models(self):
        """Load all model files"""
        logger.info("Loading Acceleration microIDS model files...")
        
        try:
            self.model = joblib.load(self.model_dir / 'bsm_acceleration_model.pkl')
            self.scaler = joblib.load(self.model_dir / 'bsm_acceleration_scaler.pkl')
            self.feature_cols = joblib.load(self.model_dir / 'acceleration_feature_names.pkl')
            self.metadata = joblib.load(self.model_dir / 'acceleration_model_metadata.pkl')
            
            logger.info(f"✅ Acceleration Model loaded successfully!")
            logger.info(f"   Model: {self.metadata['model_type']}")
            logger.info(f"   Attack Type: {self.metadata['attack_type']}")
            logger.info(f"   Tree Nodes: {self.metadata['n_nodes']}")
            logger.info(f"   Features: {self.metadata['n_features']}")
            logger.info(f"   F1-Score: {self.metadata['global_f1']:.4f}")
            logger.info(f"   Accuracy: {self.metadata['global_accuracy']:.4f}")
            logger.info(f"   Training samples: {self.metadata['training_samples']:,}")
            logger.info(f"   Test samples: {self.metadata['test_samples']:,}")
            logger.info(f"   🎯 Decision Threshold: {self.decision_threshold:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def engineer_features(self, bsm):
        """Engineer acceleration-based features from BSM"""
        sender_id = bsm.get('senderId', 'unknown')
        current_accel = float(bsm.get('longAcceleration', 0))
        current_speed = float(bsm.get('speed', 0))
        gen_time = int(float(bsm.get('generationTime', 0)))
        
        state = self.vehicle_states[sender_id]
        
        # Calculate acceleration delta (jerk)
        if state['last_accel'] is not None:
            accel_delta = current_accel - state['last_accel']
        else:
            accel_delta = 0.0
        
        accel_jerk = abs(accel_delta)
        accel_abs = abs(current_accel)
        
        # Calculate expected acceleration from speed delta
        if state['last_speed'] is not None:
            speed_delta = current_speed - state['last_speed']
        else:
            speed_delta = 0.0
        
        # Calculate time gap
        if state['last_time'] is not None:
            time_gap_sec = (gen_time - state['last_time']) / 1e9
            if time_gap_sec <= 0:
                time_gap_sec = 0.1  # Avoid division by zero
        else:
            time_gap_sec = 0.1
        
        # Expected acceleration
        expected_accel = speed_delta / time_gap_sec if time_gap_sec > 0 else 0.0
        accel_mismatch = abs(current_accel - expected_accel)
        accel_mismatch_ratio = accel_mismatch / (abs(expected_accel) + 1e-6)
        
        # Acceleration consistency (rolling window std)
        if state['accel_history']:
            recent_accels = state['accel_history'][-3:] if len(state['accel_history']) >= 3 else state['accel_history']
            accel_std_3 = np.std(recent_accels) if len(recent_accels) > 1 else 0.0
        else:
            accel_std_3 = 0.0
        
        # Anomaly flags (using OPTIMIZED thresholds)
        is_excessive_accel = 1 if accel_abs > 8.0 else 0
        is_high_jerk = 1 if accel_jerk > 12.0 else 0
        is_erratic_accel = 1 if accel_std_3 > 8.0 else 0
        is_accel_mismatch = 1 if accel_mismatch > 2.0 else 0
        
        bitlen = float(bsm.get('bitLen', 98))
        
        # Update state
        state['last_accel'] = current_accel
        state['last_speed'] = current_speed
        state['last_time'] = gen_time
        state['accel_history'].append(current_accel)
        state['speed_history'].append(current_speed)
        
        # Keep only recent history
        if len(state['accel_history']) > 10:
            state['accel_history'].pop(0)
        if len(state['speed_history']) > 10:
            state['speed_history'].pop(0)
        
        # Return feature dict
        return {
            'accel_abs': accel_abs,
            'accel_jerk': accel_jerk,
            'accel_std_3': accel_std_3,
            'accel_mismatch': accel_mismatch,
            'accel_mismatch_ratio': accel_mismatch_ratio,
            'is_excessive_accel': is_excessive_accel,
            'is_high_jerk': is_high_jerk,
            'is_erratic_accel': is_erratic_accel,
            'is_accel_mismatch': is_accel_mismatch,
            'bitLen': bitlen,
        }

    def predict(self, bsm):
        """Make prediction for single BSM"""
        try:
            sender_id = bsm.get('senderId', 'unknown')
            
            # Engineer features
            features = self.engineer_features(bsm)
            
            # Use DataFrame to match training
            X = pd.DataFrame([[features[col] for col in self.feature_cols]],
                             columns=self.feature_cols)
            X_scaled = self.scaler.transform(X)
            
            # ════════════════════════════════════════════════════════════════════
            # 🎯 NEW: Use probability threshold instead of raw prediction
            # ════════════════════════════════════════════════════════════════════
            proba = self.model.predict_proba(X_scaled)[0]
            attack_proba = float(proba[1])  # Probability of attack class
            
            # Only flag as attack if confidence >= threshold
            pred = 1 if attack_proba >= self.decision_threshold else 0
            confidence = attack_proba  # Always report actual probability
            # ════════════════════════════════════════════════════════════════════
            
            # Determine attack type
            attack_type = 'none'
            if features['is_excessive_accel']:
                attack_type = 'excessive_accel'
            elif features['is_high_jerk']:
                attack_type = 'high_jerk'
            elif features['is_erratic_accel']:
                attack_type = 'erratic_accel'
            elif features['is_accel_mismatch']:
                attack_type = 'accel_mismatch'
            
            result = {
                'is_attack': bool(pred),
                'confidence': confidence,
                'accel_abs': round(features['accel_abs'], 4),
                'accel_jerk': round(features['accel_jerk'], 4),
                'accel_mismatch': round(features['accel_mismatch'], 4),
                'attack_type': attack_type,
            }
            
            # Update stats
            self.stats['total'] += 1
            if pred:
                self.stats['attacks'] += 1
                logger.warning(f"🚨 ACCELERATION ATTACK: {sender_id} | "
                             f"Accel: {features['accel_abs']:.2f} m/s² | "
                             f"Jerk: {features['accel_jerk']:.2f} m/s³ | "
                             f"Type: {attack_type} | Conf: {attack_proba:.2%} | "
                             f"Threshold: {self.decision_threshold:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {bsm.get('senderId', '?')}: {e}")
            return {'is_attack': False, 'error': str(e)}

    def run(self, host='0.0.0.0'):
        """Start TCP server"""
        self.load_models()
        
        logger.info(f"✅ Acceleration microIDS server listening on {host}:{self.port}")
        logger.info(f"   Monitoring for acceleration/jerk attacks...")
        
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, self.port))
        server_socket.listen(5)
        
        try:
            while True:
                client_socket, address = server_socket.accept()
                logger.info(f"📨 Client connected: {address[0]}:{address[1]}")
                
                try:
                    while True:
                        data = client_socket.recv(4096)
                        if not data:
                            break
                        
                        try:
                            bsm = json.loads(data.decode())
                            result = self.predict(bsm)
                            response = json.dumps(result).encode()
                            client_socket.sendall(response + b'\n')
                            
                            # Log stats every 200 BSMs
                            if self.stats['total'] % 200 == 0:
                                attack_rate = 100 * self.stats['attacks'] / self.stats['total']
                                logger.info(f"Stats: {self.stats['total']} total | "
                                          f"{self.stats['attacks']} attacks | "
                                          f"{attack_rate:.1f}% attack rate")
                        
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON received: {data}")
                            
                except Exception as e:
                    logger.error(f"Connection error: {e}")
                finally:
                    client_socket.close()
                    logger.info(f"Client disconnected: {address[0]}:{address[1]}")
                    
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        finally:
            server_socket.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Acceleration microIDS Inference Server')
    parser.add_argument('--model-dir', default='./', help='Directory with model files')
    parser.add_argument('--port', type=int, default=9999, help='Port to listen on (default: 9999)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--threshold', type=float, default=0.75, 
                       help='Decision threshold for attack detection (0.5-0.95, default: 0.75)')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not (0.5 <= args.threshold <= 0.95):
        print(f"❌ Invalid threshold: {args.threshold}")
        print(f"   Must be between 0.5 and 0.95")
        exit(1)
    
    server = AccelerationMicroIDSServer(args.model_dir, args.port, args.threshold)
    server.run(args.host)
