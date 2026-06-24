"""
V2X 3-LAYER AGGREGATOR MODEL - STRICT CONSENSUS VOTING
FIXED: Feature name warnings from sklearn

Simple 3-layer architecture:
1. Layer 1: Time-based IDS (DoS detection)
   - If attack detected → immediate DOS verdict, skip Layer 2
   
2. Layer 2: Parallel detectors (Speed, Acceleration, Heading)
   - All three run in parallel if Layer 1 passes
   - Each returns: is_attack (True/False)
   
3. Layer 3: Strict Consensus Decision
   - Only if ALL THREE say attack → MESSAGE_FABRICATION
   - If even ONE says normal → NORMAL (no alert)

All attack messages logged to: /home/kali/v2x_ids_system/logs/aggregator.log
"""

import joblib
import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import warnings

# Suppress sklearn feature name warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

LOG_PATH = Path("/home/kali/v2x_ids_system/logs/aggregator.log")


def setup_logger():
    """Setup append-only file logger."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Aggregator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


AGG_LOGGER = setup_logger()


class TimeMicroIDS:
    """Time-interval (DoS) microIDS - Layer 1."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = joblib.load(model_dir / "bsm_dos_model.pkl")
        self.scaler = joblib.load(model_dir / "bsm_dos_scaler.pkl")
        self.feature_cols = joblib.load(model_dir / "feature_names.pkl")
        self.vehicle_states = {}

    def engineer_features(self, bsm: dict):
        sender_id = bsm.get("senderId", "unknown")
        gen_time = int(float(bsm.get("generationTime", 0)))

        prev_state = self.vehicle_states.get(sender_id, {})
        prev_time = prev_state.get("last_time", gen_time)

        if prev_time == gen_time:
            time_gap = 999.0
        else:
            time_gap = (gen_time - prev_time) / 1e9

        is_too_fast = 1 if time_gap < 0.1 else 0
        is_too_slow = 1 if time_gap > 1.0 else 0
        gap_violation = 1 if (is_too_fast or is_too_slow) else 0

        heading = float(bsm.get("heading", 0.0))
        speed = float(bsm.get("speed", 0.0))
        heading_rad = abs(np.radians(heading))
        speed_heading_ratio = speed / (heading_rad + 1e-6)

        accel = float(bsm.get("longAcceleration", 0.0))
        accel_anomaly = 1 if abs(accel) > 5.0 else 0

        bitlen = float(bsm.get("bitLen", 98.0))

        self.vehicle_states[sender_id] = {
            "last_time": gen_time,
            "last_lat": float(bsm.get("latitude", 0.0)),
            "last_lon": float(bsm.get("longitude", 0.0)),
        }

        return {
            "time_gap_sec": time_gap,
            "is_too_fast": is_too_fast,
            "is_too_slow": is_too_slow,
            "gap_violation": gap_violation,
            "speed_heading_ratio": speed_heading_ratio,
            "spatial_delta": 0.0,
            "accel_anomaly": accel_anomaly,
            "bitLen": bitlen,
        }

    def predict(self, bsm: dict):
        """Return (is_attack, confidence)."""
        feats = self.engineer_features(bsm)
        X = pd.DataFrame([[feats[col] for col in self.feature_cols]], columns=self.feature_cols)
        X_scaled = self.scaler.transform(X)
        
        pred = int(self.model.predict(X_scaled)[0])
        proba = self.model.predict_proba(X_scaled)[0]
        confidence = float(np.max(proba))

        violation = "none"
        if feats["is_too_fast"]:
            violation = "too_fast"
        elif feats["is_too_slow"]:
            violation = "too_slow"

        return {
            "is_attack": bool(pred),
            "confidence": confidence,
            "violation": violation,
            "time_gap_sec": feats["time_gap_sec"],
        }


class SpeedMicroIDS:
    """Speed fabrication microIDS - Layer 2."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = joblib.load(model_dir / "bsm_speed_model.pkl")
        self.scaler = joblib.load(model_dir / "bsm_speed_scaler.pkl")
        self.feature_cols = joblib.load(model_dir / "speed_feature_names.pkl")
        self.vehicle_states = defaultdict(self._init_vehicle_state)

    def _init_vehicle_state(self):
        return {
            "last_speed": None,
            "last_time": None,
            "speed_history": [],
            "accel_history": [],
        }

    def engineer_features(self, bsm: dict):
        sender_id = bsm.get("senderId", "unknown")
        speed = float(bsm.get("speed", 0.0))
        gen_time = int(float(bsm.get("generationTime", 0)))
        lat = float(bsm.get("latitude", 0.0))
        lon = float(bsm.get("longitude", 0.0))
        bitlen = float(bsm.get("bitLen", 98.0))

        state = self.vehicle_states[sender_id]

        if state["last_speed"] is not None:
            speed_delta = speed - state["last_speed"]
        else:
            speed_delta = 0.0
        speed_delta_abs = abs(speed_delta)

        if state["last_time"] is not None:
            dt = (gen_time - state["last_time"]) / 1e9
            if dt <= 0:
                dt = 0.1
        else:
            dt = 0.1

        accel = speed_delta / dt
        accel_abs = abs(accel)

        if state["last_speed"] is not None:
            speed_ratio = speed / (state["last_speed"] + 1e-6)
        else:
            speed_ratio = 1.0

        if state["accel_history"]:
            accel_jerk = abs(accel - state["accel_history"][-1])
        else:
            accel_jerk = 0.0

        is_excessive_speed_change = 1 if speed_delta_abs > 2.5 else 0
        is_excessive_accel = 1 if accel_abs > 5.5 else 0
        is_unrealistic_ratio = 1 if (speed_ratio < 0.45 or speed_ratio > 2.2) else 0
        is_high_jerk = 1 if accel_jerk > 3.5 else 0

        if state["last_time"] is not None:
            spatial_delta = np.sqrt(
                (lat - state.get("last_lat", lat)) ** 2
                + (lon - state.get("last_lon", lon)) ** 2
            )
        else:
            spatial_delta = 0.0

        state["last_speed"] = speed
        state["last_time"] = gen_time
        state["last_lat"] = lat
        state["last_lon"] = lon
        state["speed_history"].append(speed)
        state["accel_history"].append(accel)
        if len(state["speed_history"]) > 10:
            state["speed_history"].pop(0)
        if len(state["accel_history"]) > 10:
            state["accel_history"].pop(0)

        return {
            "speed_delta_abs": speed_delta_abs,
            "acceleration_abs": accel_abs,
            "speed_ratio": speed_ratio,
            "accel_jerk": accel_jerk,
            "is_excessive_speed_change": is_excessive_speed_change,
            "is_excessive_accel": is_excessive_accel,
            "is_unrealistic_ratio": is_unrealistic_ratio,
            "is_high_jerk": is_high_jerk,
            "spatial_delta": spatial_delta,
            "bitLen": bitlen,
        }

    def predict(self, bsm: dict):
        """Return (is_attack, confidence)."""
        feats = self.engineer_features(bsm)
        X = pd.DataFrame([[feats[c] for c in self.feature_cols]], columns=self.feature_cols)
        X_scaled = self.scaler.transform(X)
        
        pred = int(self.model.predict(X_scaled)[0])
        proba = self.model.predict_proba(X_scaled)[0]
        confidence = float(np.max(proba))

        return {
            "is_attack": bool(pred),
            "confidence": confidence,
        }


class HeadingMicroIDS:
    """Heading fabrication microIDS - Layer 2."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = joblib.load(model_dir / "bsm_heading_model.pkl")
        self.scaler = joblib.load(model_dir / "bsm_heading_scaler.pkl")
        self.feature_cols = joblib.load(model_dir / "heading_feature_names.pkl")
        self.vehicle_states = defaultdict(self._init_vehicle_state)

    def _init_vehicle_state(self):
        return {
            "last_heading": None,
            "last_time": None,
            "last_lat": None,
            "last_lon": None,
            "heading_history": [],
            "heading_rate_history": [],
            "last_speed": None,
        }

    @staticmethod
    def normalize_heading_delta(delta: float) -> float:
        return ((delta + 180.0) % 360.0) - 180.0

    def engineer_features(self, bsm: dict):
        sender_id = bsm.get("senderId", "unknown")
        heading = float(bsm.get("heading", 0.0))
        speed = float(bsm.get("speed", 0.0))
        gen_time = int(float(bsm.get("generationTime", 0)))
        lat = float(bsm.get("latitude", 0.0))
        lon = float(bsm.get("longitude", 0.0))
        bitlen = float(bsm.get("bitLen", 98.0))

        state = self.vehicle_states[sender_id]

        if state["last_heading"] is not None:
            heading_delta = heading - state["last_heading"]
            heading_delta = self.normalize_heading_delta(heading_delta)
        else:
            heading_delta = 0.0
        heading_delta_abs = abs(heading_delta)

        if state["last_time"] is not None:
            dt = (gen_time - state["last_time"]) / 1e9
            if dt <= 0:
                dt = 0.1
        else:
            dt = 0.1

        heading_rate = heading_delta / dt
        heading_rate_abs = abs(heading_rate)

        if state["heading_rate_history"]:
            heading_accel = abs(heading_rate - state["heading_rate_history"][-1])
        else:
            heading_accel = 0.0

        if state["heading_history"]:
            speed_delta = speed - (state["last_speed"] if state["last_speed"] is not None else speed)
        else:
            speed_delta = 0.0

        heading_speed_ratio = (heading_delta_abs + 1.0) / (abs(speed_delta) + 1.0)

        is_excessive_heading_change = 1 if heading_delta_abs > 4.0 else 0
        is_high_heading_rate = 1 if heading_rate_abs > 45.0 else 0
        is_unrealistic_heading_accel = 1 if heading_accel > 90.0 else 0
        is_inconsistent_heading_speed = 1 if heading_speed_ratio > 3.0 else 0

        if state["last_time"] is not None:
            spatial_delta = np.sqrt(
                (lat - (state["last_lat"] or lat)) ** 2
                + (lon - (state["last_lon"] or lon)) ** 2
            )
        else:
            spatial_delta = 0.0

        state["last_heading"] = heading
        state["last_time"] = gen_time
        state["last_lat"] = lat
        state["last_lon"] = lon
        state["last_speed"] = speed
        state["heading_history"].append(heading)
        state["heading_rate_history"].append(heading_rate)
        if len(state["heading_history"]) > 10:
            state["heading_history"].pop(0)
        if len(state["heading_rate_history"]) > 10:
            state["heading_rate_history"].pop(0)

        return {
            "heading_delta_abs": heading_delta_abs,
            "heading_rate_abs": heading_rate_abs,
            "heading_accel": heading_accel,
            "heading_speed_ratio": heading_speed_ratio,
            "is_excessive_heading_change": is_excessive_heading_change,
            "is_high_heading_rate": is_high_heading_rate,
            "is_unrealistic_heading_accel": is_unrealistic_heading_accel,
            "is_inconsistent_heading_speed": is_inconsistent_heading_speed,
            "spatial_delta": spatial_delta,
            "bitLen": bitlen,
        }

    def predict(self, bsm: dict):
        """Return (is_attack, confidence)."""
        feats = self.engineer_features(bsm)
        X = pd.DataFrame([[feats[c] for c in self.feature_cols]], columns=self.feature_cols)
        X_scaled = self.scaler.transform(X)
        
        pred = int(self.model.predict(X_scaled)[0])
        proba = self.model.predict_proba(X_scaled)[0]
        confidence = float(np.max(proba))

        return {
            "is_attack": bool(pred),
            "confidence": confidence,
        }


class AccelerationMicroIDS:
    """Acceleration fabrication microIDS - Layer 2."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = joblib.load(model_dir / "bsm_acceleration_model.pkl")
        self.scaler = joblib.load(model_dir / "bsm_acceleration_scaler.pkl")
        self.feature_cols = joblib.load(model_dir / "acceleration_feature_names.pkl")
        self.vehicle_states = defaultdict(self._init_vehicle_state)

    def _init_vehicle_state(self):
        return {
            "last_accel": None,
            "last_speed": None,
            "last_time": None,
            "accel_history": [],
            "speed_history": [],
        }

    def engineer_features(self, bsm: dict):
        sender_id = bsm.get("senderId", "unknown")
        accel = float(bsm.get("longAcceleration", 0.0))
        speed = float(bsm.get("speed", 0.0))
        gen_time = int(float(bsm.get("generationTime", 0)))
        bitlen = float(bsm.get("bitLen", 98.0))

        state = self.vehicle_states[sender_id]

        if state["last_accel"] is not None:
            accel_delta = accel - state["last_accel"]
        else:
            accel_delta = 0.0
        accel_jerk = abs(accel_delta)
        accel_abs = abs(accel)

        if state["last_speed"] is not None:
            speed_delta = speed - state["last_speed"]
        else:
            speed_delta = 0.0

        if state["last_time"] is not None:
            dt = (gen_time - state["last_time"]) / 1e9
            if dt <= 0:
                dt = 0.1
        else:
            dt = 0.1

        expected_accel = speed_delta / dt if dt > 0 else 0.0
        accel_mismatch = abs(accel - expected_accel)
        accel_mismatch_ratio = accel_mismatch / (abs(expected_accel) + 1e-6)

        if state["accel_history"]:
            recent = state["accel_history"][-3:] if len(state["accel_history"]) >= 3 else state["accel_history"]
            accel_std_3 = np.std(recent) if len(recent) > 1 else 0.0
        else:
            accel_std_3 = 0.0

        is_excessive_accel = 1 if accel_abs > 8.0 else 0
        is_high_jerk = 1 if accel_jerk > 12.0 else 0
        is_erratic_accel = 1 if accel_std_3 > 8.0 else 0
        is_accel_mismatch = 1 if accel_mismatch > 2.0 else 0

        state["last_accel"] = accel
        state["last_speed"] = speed
        state["last_time"] = gen_time
        state["accel_history"].append(accel)
        state["speed_history"].append(speed)
        if len(state["accel_history"]) > 10:
            state["accel_history"].pop(0)
        if len(state["speed_history"]) > 10:
            state["speed_history"].pop(0)

        return {
            "accel_abs": accel_abs,
            "accel_jerk": accel_jerk,
            "accel_std_3": accel_std_3,
            "accel_mismatch": accel_mismatch,
            "accel_mismatch_ratio": accel_mismatch_ratio,
            "is_excessive_accel": is_excessive_accel,
            "is_high_jerk": is_high_jerk,
            "is_erratic_accel": is_erratic_accel,
            "is_accel_mismatch": is_accel_mismatch,
            "bitLen": bitlen,
        }

    def predict(self, bsm: dict):
        """Return (is_attack, confidence)."""
        feats = self.engineer_features(bsm)
        X = pd.DataFrame([[feats[c] for c in self.feature_cols]], columns=self.feature_cols)
        X_scaled = self.scaler.transform(X)
        
        pred = int(self.model.predict(X_scaled)[0])
        proba = self.model.predict_proba(X_scaled)[0]
        confidence = float(np.max(proba))

        return {
            "is_attack": bool(pred),
            "confidence": confidence,
        }


class AggregatorModel:
    """3-layer ensemble aggregator - STRICT CONSENSUS VOTING."""

    def __init__(self, models_dir: str):
        models_path = Path(models_dir)
        self.time_ids = TimeMicroIDS(models_path)
        self.speed_ids = SpeedMicroIDS(models_path)
        self.heading_ids = HeadingMicroIDS(models_path)
        self.accel_ids = AccelerationMicroIDS(models_path)

    def process_bsm(self, bsm: dict):
        """Process single BSM through 3-layer architecture."""

        sender_id = bsm.get("senderId", "unknown")
        gen_time_ns = int(float(bsm.get("generationTime", 0)))

        # ============================================================================
        # LAYER 1: Time-based IDS (DoS Detection)
        # ============================================================================
        time_result = self.time_ids.predict(bsm)

        if time_result["is_attack"]:
            # DOS DETECTED: Immediate verdict, skip Layer 2
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
            AGG_LOGGER.info(json.dumps({"alert": result}))
            return result

        # ============================================================================
        # LAYER 2: Parallel Fabrication Detectors (only if Layer 1 passes)
        # ============================================================================
        speed_result = self.speed_ids.predict(bsm)
        accel_result = self.accel_ids.predict(bsm)
        heading_result = self.heading_ids.predict(bsm)

        speed_attack = speed_result["is_attack"]
        accel_attack = accel_result["is_attack"]
        heading_attack = heading_result["is_attack"]

        speed_conf = float(speed_result["confidence"])
        accel_conf = float(accel_result["confidence"])
        heading_conf = float(heading_result["confidence"])

        # ============================================================================
        # LAYER 3: Final decision (STRICT CONSENSUS)
        # Rule: If ALL THREE say attack → MESSAGE_FABRICATION
        #       If even ONE says normal → NORMAL (no alert)
        # ============================================================================
        if speed_attack and accel_attack and heading_attack:
            attack_type = "MESSAGE_FABRICATION"
        else:
            attack_type = "NORMAL"

        result = {
            "senderId": sender_id,
            "attack_type": attack_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dos_confidence": None,
            "speed_attack": bool(speed_attack),
            "accel_attack": bool(accel_attack),
            "heading_attack": bool(heading_attack),
            "speed_conf": speed_conf,
            "accel_conf": accel_conf,
            "heading_conf": heading_conf,
        }

        # Log only attack messages
        if attack_type != "NORMAL":
            AGG_LOGGER.info(json.dumps({"alert": result}))

        return result
