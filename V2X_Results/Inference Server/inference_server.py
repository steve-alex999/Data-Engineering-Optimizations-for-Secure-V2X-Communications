#!/usr/bin/env python3
"""
Lightweight microIDS Inference Server
Loads trained model and listens for BSM packets on port 9999
"""

import socket
import json
import logging
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('InferenceServer')

class MicroIDSServer:
    def __init__(self, model_dir='./'):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.metadata = None
        self.vehicle_states = {}
        self.stats = {'total': 0, 'attacks': 0}
        
    def load_models(self):
        """Load all model files"""
        logger.info("Loading model files...")
        
        try:
            self.model = joblib.load(self.model_dir / 'bsm_dos_model.pkl')
            self.scaler = joblib.load(self.model_dir / 'bsm_dos_scaler.pkl')
            self.feature_cols = joblib.load(self.model_dir / 'feature_names.pkl')
            self.metadata = joblib.load(self.model_dir / 'model_metadata.pkl')
            
            logger.info(f"✅ Model loaded: {self.metadata['n_nodes']} nodes, "
                       f"{self.metadata['n_features']} features")
            logger.info(f"   F1-Score: {self.metadata['global_f1']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def engineer_features(self, bsm):
        """Engineer features from BSM"""
        sender_id = bsm.get('senderId', 'unknown')
        gen_time = bsm.get('generationTime', 0)
        
        # Get previous time for this vehicle
        prev_time = self.vehicle_states.get(sender_id, {}).get('last_time', gen_time)
        
        # Calculate time gap
        time_gap = (gen_time - prev_time) / 1e9 if prev_time != gen_time else 999
        if prev_time == gen_time:
            time_gap = 999  # First packet
        
        # Feature engineering
        is_too_fast = 1 if time_gap < 0.1 else 0
        is_too_slow = 1 if time_gap > 1.0 else 0
        gap_violation = 1 if (is_too_fast or is_too_slow) else 0
        
        heading = float(bsm.get('heading', 0))
        speed = float(bsm.get('speed', 0))
        heading_rad = abs(np.radians(heading))
        speed_heading_ratio = speed / (heading_rad + 1e-6)
        
        accel = float(bsm.get('longAcceleration', 0))
        accel_anomaly = 1 if abs(accel) > 5.0 else 0
        
        bitlen = float(bsm.get('bitLen', 98))
        
        # Store state
        self.vehicle_states[sender_id] = {
            'last_time': gen_time,
            'last_lat': float(bsm.get('latitude', 0)),
            'last_lon': float(bsm.get('longitude', 0)),
        }
        
        return {
            'time_gap_sec': time_gap,
            'is_too_fast': is_too_fast,
            'is_too_slow': is_too_slow,
            'gap_violation': gap_violation,
            'speed_heading_ratio': speed_heading_ratio,
            'spatial_delta': 0.0,
            'accel_anomaly': accel_anomaly,
            'bitLen': bitlen,
        }

    def predict(self, bsm):
        """Make prediction for single BSM"""
        try:
            # Engineer features
            features = self.engineer_features(bsm)
            
            # Order features
            X = np.array([features[col] for col in self.feature_cols])
            X_scaled = self.scaler.transform([X])
            
            # Predict
            pred = self.model.predict(X_scaled)[0]
            proba = self.model.predict_proba(X_scaled)[0]
            
            # Determine violation type
            violation = 'none'
            if features['is_too_fast']:
                violation = 'too_fast'
            elif features['is_too_slow']:
                violation = 'too_slow'
            
            result = {
                'is_attack': bool(pred),
                'confidence': float(proba[int(pred)]),
                'time_gap_sec': round(features['time_gap_sec'], 4),
                'violation': violation,
            }
            
            # Update stats
            self.stats['total'] += 1
            if pred:
                self.stats['attacks'] += 1
                logger.warning(f"🚨 ATTACK: {bsm.get('senderId', '?')} | "
                             f"Gap: {features['time_gap_sec']:.4f}s | "
                             f"Type: {violation} | Conf: {proba[1]:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'is_attack': False, 'error': str(e)}

    def run(self, host='0.0.0.0', port=9999):
        """Start TCP server"""
        self.load_models()
        
        logger.info(f"✅ Inference server listening on {host}:{port}")
        
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
                            
                            # Log stats every 100 BSMs
                            if self.stats['total'] % 100 == 0:
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
    server = MicroIDSServer('./')
    server.run(host='0.0.0.0', port=9999)
