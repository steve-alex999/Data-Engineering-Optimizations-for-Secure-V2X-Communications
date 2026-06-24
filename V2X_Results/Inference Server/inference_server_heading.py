#!/usr/bin/env python3
"""
Heading Fabrication microIDS Inference Server
Loads trained model and listens for BSM packets on port 9999
Detects unrealistic heading changes (direction fabrication attacks)
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
logger = logging.getLogger('HeadingMicroIDS')

class HeadingMicroIDSServer:
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
            'last_heading': None,
            'last_time': None,
            'last_lat': None,
            'last_lon': None,
            'heading_history': [],
            'heading_rate_history': [],
            'recent_verdicts': [],
        }
        
    def load_models(self):
        """Load all model files"""
        logger.info("Loading Heading microIDS model files...")
        
        try:
            self.model = joblib.load(self.model_dir / 'bsm_heading_model.pkl')
            self.scaler = joblib.load(self.model_dir / 'bsm_heading_scaler.pkl')
            self.feature_cols = joblib.load(self.model_dir / 'heading_feature_names.pkl')
            self.metadata = joblib.load(self.model_dir / 'heading_model_metadata.pkl')
            
            logger.info(f"✅ Heading Model loaded successfully!")
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

    def normalize_heading_delta(self, delta):
        """Normalize heading change to [-180, 180] range"""
        return ((delta + 180) % 360) - 180

    def engineer_features(self, bsm):
        """Engineer heading-based features from BSM"""
        sender_id = bsm.get('senderId', 'unknown')
        current_heading = float(bsm.get('heading', 0))
        gen_time = int(float(bsm.get('generationTime', 0)))
        
        state = self.vehicle_states[sender_id]
        
        # Calculate heading delta
        if state['last_heading'] is not None:
            heading_delta = current_heading - state['last_heading']
            heading_delta = self.normalize_heading_delta(heading_delta)
        else:
            heading_delta = 0.0
        
        heading_delta_abs = abs(heading_delta)
        
        # Calculate time gap for heading rate
        if state['last_time'] is not None:
            time_gap_sec = (gen_time - state['last_time']) / 1e9
            if time_gap_sec <= 0:
                time_gap_sec = 0.1  # Avoid division by zero
        else:
            time_gap_sec = 0.1
        
        # Calculate heading rate (degrees per second)
        heading_rate = heading_delta / time_gap_sec
        heading_rate_abs = abs(heading_rate)
        
        # Heading acceleration (rate of change of heading rate)
        if state['heading_rate_history']:
            heading_accel = abs(heading_rate - state['heading_rate_history'][-1])
        else:
            heading_accel = 0.0
        
        # Speed consistency with heading change (physics-based check)
        speed = float(bsm.get('speed', 0))
        if state['heading_history']:
            # If heading change but speed stays high, unrealistic
            speed_delta = speed - state.get('last_speed', speed)
        else:
            speed_delta = 0.0
        
        heading_speed_ratio = (heading_delta_abs + 1) / (abs(speed_delta) + 1)
        
        # Anomaly flags (using OPTIMIZED thresholds for Pi 3B+)
        is_excessive_heading_change = 1 if heading_delta_abs > 4.0 else 0  # BSM rule: > 4°
        is_high_heading_rate = 1 if heading_rate_abs > 45.0 else 0         # > 45°/sec
        is_unrealistic_heading_accel = 1 if heading_accel > 90.0 else 0    # > 90°/sec²
        is_inconsistent_heading_speed = 1 if heading_speed_ratio > 3.0 else 0  # High ratio = suspicious
        
        # Spatial delta (using location data)
        latitude = float(bsm.get('latitude', 0))
        longitude = float(bsm.get('longitude', 0))
        
        if state['last_time'] is not None:
            spatial_delta = np.sqrt(
                (latitude - state['last_lat'])**2 + 
                (longitude - state['last_lon'])**2
            )
        else:
            spatial_delta = 0.0
        
        bitlen = float(bsm.get('bitLen', 98))
        
        # Update state
        state['last_heading'] = current_heading
        state['last_time'] = gen_time
        state['last_lat'] = latitude
        state['last_lon'] = longitude
        state['last_speed'] = speed
        state['heading_history'].append(current_heading)
        state['heading_rate_history'].append(heading_rate)
        
        # Keep only recent history
        if len(state['heading_history']) > 10:
            state['heading_history'].pop(0)
        if len(state['heading_rate_history']) > 10:
            state['heading_rate_history'].pop(0)
        
        # Return feature dict
        return {
            'heading_delta_abs': heading_delta_abs,
            'heading_rate_abs': heading_rate_abs,
            'heading_accel': heading_accel,
            'heading_speed_ratio': heading_speed_ratio,
            'is_excessive_heading_change': is_excessive_heading_change,
            'is_high_heading_rate': is_high_heading_rate,
            'is_unrealistic_heading_accel': is_unrealistic_heading_accel,
            'is_inconsistent_heading_speed': is_inconsistent_heading_speed,
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
            # Use DataFrame to match training (eliminates warnings)
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
            if features['is_excessive_heading_change']:
                attack_type = 'excessive_heading_change'
            elif features['is_high_heading_rate']:
                attack_type = 'high_heading_rate'
            elif features['is_unrealistic_heading_accel']:
                attack_type = 'unrealistic_heading_accel'
            elif features['is_inconsistent_heading_speed']:
                attack_type = 'inconsistent_heading_speed'
            
            result = {
                'is_attack': bool(pred),
                'confidence': confidence,
                'heading_delta_abs': round(features['heading_delta_abs'], 4),
                'heading_rate_abs': round(features['heading_rate_abs'], 4),
                'heading_accel': round(features['heading_accel'], 4),
                'heading_speed_ratio': round(features['heading_speed_ratio'], 4),
                'attack_type': attack_type,
            }
            
            # Update stats
            self.stats['total'] += 1
            if pred:
                self.stats['attacks'] += 1
                logger.warning(f"🚨 HEADING ATTACK: {sender_id} | "
                             f"Heading Δ: {features['heading_delta_abs']:.2f}° | "
                             f"Rate: {features['heading_rate_abs']:.2f}°/sec | "
                             f"Type: {attack_type} | Conf: {proba[1]:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {bsm.get('senderId', '?')}: {e}")
            return {'is_attack': False, 'error': str(e)}

    def run(self, host='0.0.0.0', port=9999):
        """Start TCP server"""
        self.load_models()
        
        logger.info(f"✅ Heading microIDS server listening on {host}:{port}")
        logger.info(f"   Monitoring for heading fabrication attacks...")
        
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
    server = HeadingMicroIDSServer('./')
    server.run(host='0.0.0.0', port=9999)

