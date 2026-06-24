"""
V2X 3-LAYER AGGREGATOR SERVER (RSU) - STRICT CONSENSUS VOTING

TCP server listening for BSM packets and returning verdicts.
- Layer 1: Time-based IDS (DoS detection)
- Layer 2: Parallel Speed, Accel, Heading detectors
- Layer 3: Strict consensus - only alerts if ALL THREE say attack

Server logs only attack messages with:
- Per-IDS attack flags (speed_attack, accel_attack, heading_attack)
- Per-IDS confidence scores (speed_conf, accel_conf, heading_conf)
- Attack type (DOS or MESSAGE_FABRICATION)
"""

import socket
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

from aggregator_model import AggregatorModel, AGG_LOGGER, LOG_PATH


def ts():
    return datetime.now().strftime("%H:%M:%S")


class AggregatorServer:
    def __init__(self, models_dir: str, host: str = "0.0.0.0", port: int = 5555):
        self.host = host
        self.port = port
        self.models_dir = models_dir
        self.server_socket = None
        self.running = False
        self.aggregator = AggregatorModel(models_dir=models_dir)
        self.stats = {"total_packets": 0, "total_alerts": 0}

    def start(self):
        """Start the server."""
        print("=" * 100)
        print("V2X 3-LAYER AGGREGATOR SERVER (RSU) - STRICT CONSENSUS VOTING")
        print("=" * 100)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Models dir: {self.models_dir}")
        print(f"Log file  : {LOG_PATH}")
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
        """Handle single client connection."""
        buffer = ""
        packets_count = 0
        alerts_count = 0

        try:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break

                buffer += data.decode("utf-8", errors="ignore")

                # Process complete JSON lines
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        bsm = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Process through aggregator
                    result = self.aggregator.process_bsm(bsm)
                    packets_count += 1
                    self.stats["total_packets"] += 1

                    # Send response back to client
                    resp = json.dumps(result) + "\n"
                    client_socket.sendall(resp.encode("utf-8"))

                    # ================================================================
                    # ONLY LOG ATTACK MESSAGES
                    # ================================================================
                    if result.get("attack_type") != "NORMAL":
                        alerts_count += 1
                        self.stats["total_alerts"] += 1

                        # Extract attack info
                        sender_id = result.get("senderId")
                        attack_type = result.get("attack_type")
                        timestamp = result.get("timestamp")

                        # Layer 2 attacks: Extract per-IDS predictions and confidences
                        if attack_type == "MESSAGE_FABRICATION":
                            speed_attack = result.get("speed_attack")
                            accel_attack = result.get("accel_attack")
                            heading_attack = result.get("heading_attack")
                            speed_conf = result.get("speed_conf")
                            accel_conf = result.get("accel_conf")
                            heading_conf = result.get("heading_conf")

                            # Console log
                            print(
                                f"[{ts()}] [ALERT] {sender_id} - "
                                f"Type: {attack_type} | "
                                f"Speed(attack={speed_attack}, conf={speed_conf:.4f}) | "
                                f"Accel(attack={accel_attack}, conf={accel_conf:.4f}) | "
                                f"Heading(attack={heading_attack}, conf={heading_conf:.4f})"
                            )

                            # Build structured log entry
                            log_entry = {
                                "senderId": sender_id,
                                "attack_type": attack_type,
                                "timestamp": timestamp,
                                "speed_attack": speed_attack,
                                "accel_attack": accel_attack,
                                "heading_attack": heading_attack,
                                "speed_conf": float(speed_conf) if speed_conf is not None else None,
                                "accel_conf": float(accel_conf) if accel_conf is not None else None,
                                "heading_conf": float(heading_conf) if heading_conf is not None else None,
                            }

                        # Layer 1 DOS attack
                        elif attack_type == "DOS":
                            dos_conf = result.get("dos_confidence")

                            # Console log
                            print(
                                f"[{ts()}] [ALERT] {sender_id} - "
                                f"Type: {attack_type} | "
                                f"Confidence: {dos_conf:.4f}"
                            )

                            # Build structured log entry
                            log_entry = {
                                "senderId": sender_id,
                                "attack_type": attack_type,
                                "timestamp": timestamp,
                                "dos_confidence": float(dos_conf) if dos_conf is not None else None,
                            }

                        # File log: append JSON line to aggregator log
                        AGG_LOGGER.info(json.dumps({"alert": log_entry}))

        except Exception as e:
            print(f"[{ts()}] [ERROR] {addr[0]}:{addr[1]} - {e}")
        finally:
            client_socket.close()
            print(
                f"[{ts()}] [DISCONNECT] {addr[0]}:{addr[1]} "
                f"({packets_count} packets, {alerts_count} alerts)"
            )

    def stop(self):
        """Stop the server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

        print(f"\n[{ts()}] Server stopped")
        print(
            f"[{ts()}] Statistics: {self.stats['total_packets']} packets, "
            f"{self.stats['total_alerts']} alerts"
        )


def main():
    parser = argparse.ArgumentParser(
        description="V2X 3-Layer Aggregator Server (Strict Consensus Voting)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v2x_aggregator_server.py --models /home/kali/v2x_ids_system/models --port 5555
  python v2x_aggregator_server.py --models ./models --host 127.0.0.1 --port 5555
        """,
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=5555, help="Port to listen (default: 5555)"
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Path to models directory (e.g., /home/kali/v2x_ids_system/models)",
    )

    args = parser.parse_args()

    server = AggregatorServer(
        models_dir=args.models,
        host=args.host,
        port=args.port,
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
