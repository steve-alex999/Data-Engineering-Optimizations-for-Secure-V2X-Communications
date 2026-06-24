#!/usr/bin/env python3
"""
Speed Fabrication microIDS Inference Server
Loads trained model and listens for BSM packets on port 9998
UPDATED: Uses DataFrame for scaler to match training
"""

import socket
import json
import logging
import joblib
import numpy as np
import pandas as pd  # ← ADDED
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SpeedMicroIDS')

class SpeedMicroIDSServer:
    def __init__(self, model_dir='./'):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.metadata = None
        self.vehicle_states = defaultdict(self._init_vehicle_state)
        self.stats = {'total': 0, 'attacks': 0}
        
    def _init_vehicle_state(self):
        """Initialize state for new vehicle"""
        return {
            'last_speed': None,
            'last_time': None,
            'speed_history': [],
            'accel_history': [],
            'recent_verdicts': [],
        }
        
    def load_models(self):
        """Load all model files"""
        logger.info("Loading Speed microIDS model files...")
        
        try:
            self.model = joblib.load(self.model_dir / 'bsm_speed_model.pkl')
            self.scaler = joblib.load(self.model_dir / 'bsm_speed_scaler.pkl')
            self.feature_cols = joblib.load(self.model_dir / 'speed_feature_names.pkl')
            self.metadata = joblib.load(self.model_dir / 'speed_model_metadata.pkl')
            
            logger.info(f"✅ Speed Model loaded successfully!")
            logger.info(f"   Model: {self.metadata['model_type']}")
            logger.info(f"   Attack Type: {self.metadata['attack_type']}")
            logger.info(f"   Tree Nodes: {self.metadata['n_nodes']}")
            logger.info(f"   Features: {self.metadata['n_features']}")
            logger.info(f"   F1-Score: {self.metadata['global_f1']:.4f}")
            logger.info(f"   Accuracy: {self.metadata['global_accuracy']:.4f}")
            logger.info(f"   Training samples: {self.metadata['training_samples']:,}")
            logger.info(f"   Test samples: {self.metadata['test_samples']:,}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def engineer_features(self, bsm):
        """Engineer speed-based features from BSM"""
        sender_id = bsm.get('senderId', 'unknown')
        current_speed = float(bsm.get('speed', 0))
        gen_time = int(float(bsm.get('generationTime', 0)))
        
        state = self.vehicle_states[sender_id]
        
        # Calculate speed delta
        if state['last_speed'] is not None:
            speed_delta = current_speed - state['last_speed']
        else:
            speed_delta = 0.0
        
        speed_delta_abs = abs(speed_delta)
        
        # Calculate time gap for acceleration
        if state['last_time'] is not None:
            time_gap_sec = (gen_time - state['last_time']) / 1e9
            if time_gap_sec <= 0:
                time_gap_sec = 0.1  # Avoid division by zero
        else:
            time_gap_sec = 0.1
        
        # Calculate acceleration
        acceleration = speed_delta / time_gap_sec
        acceleration_abs = abs(acceleration)
        
        # Speed ratio
        speed_ratio = current_speed / (state['last_speed'] + 1e-6) if state['last_speed'] is not None else 1.0
        
        # Jerk (rate of change of acceleration)
        if state['accel_history']:
            accel_jerk = abs(acceleration - state['accel_history'][-1])
        else:
            accel_jerk = 0.0
        
        # Anomaly flags (using OPTIMIZED thresholds)
        is_excessive_speed_change = 1 if speed_delta_abs > 2.5 else 0  # ← Optimized threshold
        is_excessive_accel = 1 if acceleration_abs > 5.5 else 0         # ← Optimized threshold
        is_unrealistic_ratio = 1 if (speed_ratio < 0.45 or speed_ratio > 2.2) else 0
        is_high_jerk = 1 if accel_jerk > 3.5 else 0                    # ← Optimized threshold
        
        # Spatial delta (using location data)
        latitude = float(bsm.get('latitude', 0))
        longitude = float(bsm.get('longitude', 0))
        
        if state['last_time'] is not None:
            spatial_delta = np.sqrt(
                (latitude - state.get('last_lat', latitude))**2 + 
                (longitude - state.get('last_lon', longitude))**2
            )
        else:
            spatial_delta = 0.0
        
        bitlen = float(bsm.get('bitLen', 98))
        
        # Update state
        state['last_speed'] = current_speed
        state['last_time'] = gen_time
        state['last_lat'] = latitude
        state['last_lon'] = longitude
        state['speed_history'].append(current_speed)
        state['accel_history'].append(acceleration)
        
        # Keep only recent history
        if len(state['speed_history']) > 10:
            state['speed_history'].pop(0)
        if len(state['accel_history']) > 10:
            state['accel_history'].pop(0)
        
        # Return feature dict
        return {
            'speed_delta_abs': speed_delta_abs,
            'acceleration_abs': acceleration_abs,
            'speed_ratio': speed_ratio,
            'accel_jerk': accel_jerk,
            'is_excessive_speed_change': is_excessive_speed_change,
            'is_excessive_accel': is_excessive_accel,
            'is_unrealistic_ratio': is_unrealistic_ratio,
            'is_high_jerk': is_high_jerk,
            'spatial_delta': spatial_delta,
            'bitLen': bitlen,
        }

    def predict(self, bsm):
        """Make prediction for single BSM"""
        try:
            sender_id = bsm.get('senderId', 'unknown')
            
            # Engineer features
            features = self.engineer_features(bsm)
            
            # ════════════════════════════════════════════════════════════════════
            # UPDATED: Use DataFrame to match training (eliminates warnings)
            # ════════════════════════════════════════════════════════════════════
            X = pd.DataFrame([[features[col] for col in self.feature_cols]],
                             columns=self.feature_cols)
            X_scaled = self.scaler.transform(X)
            # ════════════════════════════════════════════════════════════════════
            
            # Predict
            pred = self.model.predict(X_scaled)[0]
            proba = self.model.predict_proba(X_scaled)[0]
            confidence = float(proba[int(pred)])
            
            # Determine attack type
            attack_type = 'none'
            if features['is_excessive_speed_change']:
                attack_type = 'excessive_speed_change'
            elif features['is_excessive_accel']:
                attack_type = 'excessive_accel'
            elif features['is_unrealistic_ratio']:
                attack_type = 'unrealistic_ratio'
            elif features['is_high_jerk']:
                attack_type = 'high_jerk'
            
            result = {
                'is_attack': bool(pred),
                'confidence': confidence,
                'speed_delta_abs': round(features['speed_delta_abs'], 4),
                'acceleration_abs': round(features['acceleration_abs'], 4),
                'speed_ratio': round(features['speed_ratio'], 4),
                'attack_type': attack_type,
            }
            
            # Update stats
            self.stats['total'] += 1
            if pred:
                self.stats['attacks'] += 1
                logger.warning(f"🚨 SPEED ATTACK: {sender_id} | "
                             f"Speed Δ: {features['speed_delta_abs']:.2f} m/s | "
                             f"Accel: {features['acceleration_abs']:.2f} m/s² | "
                             f"Type: {attack_type} | Conf: {proba[1]:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {bsm.get('senderId', '?')}: {e}")
            return {'is_attack': False, 'error': str(e)}

    def run(self, host='0.0.0.0', port=9998):
        """Start TCP server"""
        self.load_models()
        
        logger.info(f"✅ Speed microIDS server listening on {host}:{port}")
        logger.info(f"   Monitoring for speed fabrication attacks...")
        
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
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
    server = SpeedMicroIDSServer('./')
    server.run(host='0.0.0.0', port=9998)
