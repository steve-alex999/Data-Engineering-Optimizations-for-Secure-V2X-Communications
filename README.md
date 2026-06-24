# Data Engineering Optimizations for Secure V2X Communications

**Johns Hopkins University — Capstone Project**
**Authors:** Seonuk Kim · Harsh Bhaskar · Stephen Guzzaralapudi
**Advisors:** Dr. Lanier Watkins · Dr. Ahmed Abdo (APL) · Ilya Sabnani
**Supported by:** Johns Hopkins University Applied Physics Laboratory

---

## Overview

This project designs and evaluates a **lightweight, fog-deployable data platform** for Vehicle-to-Everything (V2X) communications, and applies it to an **autonomous cybersecurity system** capable of detecting and filtering malicious Basic Safety Messages (BSMs) in real time.

Modern V2X infrastructure relies heavily on cloud and edge computing, creating critical availability risks when WAN links go down. This platform addresses that gap by moving data ingestion, storage, and processing directly onto **Roadside Units (RSUs)** — eliminating dependence on centralized data centers for safety-critical operations.

On top of the data platform, the project introduces a novel **three-tier micro-IDS + aggregator architecture** for autonomic V2X cybersecurity, targeting both Denial-of-Service (DoS) flooding attacks and fabricated motion data attacks at the fog layer.

---

## Key Contributions

- **Lightweight fog-level data platform** combining MQTT (Mosquitto), Apache Kafka, MongoDB, and Apache Flink — deployable on resource-constrained RSU hardware.
- **MQTT broker benchmarking** across Mosquitto, RabbitMQ, HiveMQ, and NanoMQ under urban, suburban, and rural V2X workloads.
- **Autonomous micro-IDS + aggregator architecture** with specialized detectors for timing intervals, speed, position, heading, and acceleration.
- **Baseline Decision Tree IDS** achieving 98.85% accuracy, 86.97% recall, and sub-millisecond inference (0.0009 ms/packet) on BSM-like traffic.
- **Micro-IDS ensemble** achieving 99.48–99.98% per-detector accuracy with ~1.6 ms latency and throughput scalable to ~700 pkt/s with multithreading.
- First known application of MAPE-K autonomic cybersecurity principles to a fog-level V2X IDS pipeline.

---

## Architecture

The platform consists of three layers managed by a Python orchestration framework:

```
Connected Vehicles / RSUs
        │  (MQTT publish)
        ▼
 ┌─────────────────┐
 │  Mosquitto MQTT │  ← Broker: lightweight, lowest latency
 └────────┬────────┘
          │  (MQTT-Kafka bridge)
          ▼
 ┌─────────────────┐
 │  Apache Kafka   │  ← Event streaming, topic-based routing
 └──────┬──────────┘
        │
   ┌────┴─────┐
   ▼          ▼
MongoDB    Apache Flink
(Storage)  (Stream Processing)
             │
      Micro-IDS Pipeline
      ┌──────┼──────┐
   Time    Speed  Position
    IDS     IDS     IDS ...
      └──────┼──────┘
             ▼
         Aggregator
       (Decision Tree)
             │
     Benign / DoS / Fabrication
```

### Component Versions

| Component | Version |
|-----------|---------|
| Python | 3.13.2 |
| Mosquitto (MQTT) | 2.0.12 |
| paho-mqtt | 2.1.0 |
| confluent-kafka | 2.12.2 |
| Confluent (Kafka) | 4.45.1 |
| MongoDB | 7.0 |
| Apache Flink (PyFlink) | 1.18.0 |

---

## Intrusion Detection System

The IDS adopts a **three-tier hybrid architecture** aligned with MAPE-K autonomic cybersecurity principles:

### Tier 1 — Time Interval IDS
Monitors inter-message timing per sender. Abnormal burst rates are forwarded directly to the aggregator as DoS indicators.

### Tier 2 — Motion Micro-IDS Suite
Four specialized detectors check physical consistency of BSM motion fields:
- Speed IDS
- Position IDS
- Heading IDS
- Acceleration IDS

### Tier 3 — Aggregator (Decision Tree)
Fuses verdicts from Tiers 1 and 2 to produce a final, explainable classification: **Benign**, **DoS**, or **Fabrication**.

### IDS Performance Summary

| Metric | Baseline DT | Time IDS | Speed IDS | Heading IDS | Acceleration IDS |
|--------|------------|----------|-----------|-------------|-----------------|
| Accuracy | 98.85% | 99.48% | 99.95% | 99.98% | 99.52% |
| Recall | 86.97% | 99.48% | 99.96% | 99.98% | 99.52% |
| Precision | 73.31% | 99.50% | 99.92% | 99.98% | 99.95% |
| Latency (ms) | 0.0009 | ~1.61 | ~1.61 | ~1.61 | ~1.66 |
| Model Size | 14.4 KB | — | — | — | — |

Current integrated system latency: **1.8–6.5 ms end-to-end**, with expected optimization to ~1.4 ms at ~700 pkt/s.

---

## Data Platform Stress Test Results

The platform was benchmarked against V2X workloads derived from C2C-CC standards across urban, suburban, and rural traffic densities. Mosquitto outperformed RabbitMQ, HiveMQ (Community Edition), and NanoMQ across all test conditions on both MQTT-only and end-to-end (MQTT + Kafka) latency.

Key Kafka producer/consumer configuration highlights:
- `linger.ms = 0` — instantaneous forwarding
- `compression.type = none` — reduced overhead
- `acks = all` — at-least-once delivery guarantee
- `retention.ms = 500` — discards stale messages beyond DOT's 1000 ms threshold
- `num.replica.fetchers = 2` — aligned with Apache Kafka white paper best practices

---

## Threat Model

The IDS targets **BSM-flooding Denial of Service attacks** at the network/transport and application layers:

- **Transport layer TCP flooding** — SYN flood overwhelming RSU safety processing
- **Transport layer UDP flooding** — BSM rate flooding causing latency spikes
- **Application layer flooding** — protocol-compliant but oversized or high-rate BSMs

These attacks have been shown in literature to degrade or completely disable Forward Collision Warning (FCW) and Stop Sign Gap Assist (SSGA) applications.

---

## Dataset

Training and evaluation used the [V2X Security Threats Dataset](https://doi.org/10.5281/zenodo.3968768) (Goncalves et al., 2020), originally structured for CAM simulation but constrained to BSM-equivalent fields. Attack traces include:

- Rapid BSM flooding (DoS)
- Fabricated motion parameters (speed, heading, position, acceleration)

Robustness was validated using **7-fold cross-validation**, with each city map held out as a separate validation set.

---

## Repository Structure

```
.
├── data_platform/
│   ├── mqtt/               # Mosquitto config and MQTT client (paho-mqtt)
│   ├── kafka/              # Kafka producer/consumer, MQTT-Kafka bridge
│   ├── mongodb/            # MongoDB index creation, batch insert utilities
│   ├── flink/              # PyFlink stream processing jobs
│   └── orchestration/      # Python control framework, service registry, REST API
│
├── ids/
│   ├── baseline/           # Baseline Decision Tree IDS
│   ├── micro_ids/          # Time, Speed, Position, Heading, Acceleration detectors
│   └── aggregator/         # Decision Tree aggregator fusing micro-IDS verdicts
│
├── testbed/
│   ├── sensor_dump/        # Raspberry Pi sensor acquisition and preprocessing
│   └── media/              # MediaMTX video streaming integration
│
├── datasets/               # Dataset references and preprocessing scripts
├── results/                # Latency JSON files, benchmark outputs
└── Final_report.pdf        # Full capstone report
```

> **Note:** Code and datasets are available upon request for research purposes. Contact the authors via the repository.

---

## Simulator Integration

The platform is designed to integrate with **VESNOS** (Vehicular Secure Network Open Simulator), which combines:
- **SUMO** — high-fidelity traffic dynamics, controlled via the TraCI API
- **OMNET++/Veins** — V2V and V2I wireless communication simulation
- **PKI security layer** — certificate-based authentication following US-DOT standards

The current implementation is **unidirectional** (RSU receives from vehicles). Bidirectional operation — enabling RSUs to publish alerts, blacklist recommendations, and sanitized data back to vehicles — is identified as the primary next step for full autonomic operation.

---

## Hopkins V2X Testbed

A physical testbed supports real-world data collection and validation:

- **Android smartphone** (SensaGram app) — rich sensor node streaming over UDP
- **Raspberry Pi** (ARM Debian) — onboard data acquisition, preprocessing, VPN relay
- **Red Hat Linux server** — central data platform host (MongoDB + processing)
- **IP Camera app + MediaMTX** — live video streaming

The testbed enables validation of the IDS against real BSM-like traffic, including on-road noise, hardware jitter, and heterogeneous radio conditions — the primary validation milestone for future work.

---

## Future Work

1. **Bidirectional pipeline** — extend from receive-only to full publish/subscribe for self-healing and self-configuring autonomic behavior.
2. **Real V2X testbed validation** — stress the IDS under real on-road conditions and refine detection thresholds.
3. **Raspberry Pi deployment** — validate latency and memory budgets on physical RSU-class hardware.
4. **Additional message types** — extend micro-IDS coverage to SPaT, CPM, and DENM messages.
5. **Online learning** — incorporate operator feedback into the Knowledge component of the MAPE-K loop to adapt to new attack patterns.

---

## References

Key references from the full report:

- Le Lann, G. (2018). Autonomic Vehicular Networks: Safety, Privacy, Cybersecurity and Societal Issues. *IEEE VTC Spring*.
- Rouff, C., Watkins, L., et al. (2021). SoK: Autonomic Cybersecurity. *IEEE CSR*.
- Hugo, A., et al. (2020). Bridging MQTT and Kafka to Support C-ITS. *IEEE MDM*.
- Goncalves, F., et al. (2020). Synthesizing Datasets with Security Threats for VANETs. *IEEE GLOBECOM*.
- Abdo, A., Wu, G., Abu-Ghazaleh, N. (2024). VESNOS. *ACM SIGSIM-PADS*.

See `Final_report.pdf` for the complete reference list (69 citations).

---

## License

This project was developed for academic research purposes at Johns Hopkins University. Code and datasets are available upon request for research use. Please contact the authors via the GitHub repository.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{kim2025v2x,
  title     = {Data Engineering Optimizations for Secure V2X Communications},
  author    = {Kim, Seonuk and Bhaskar, Harsh and Guzzaralapudi, Stephen},
  year      = {2025},
  school    = {Johns Hopkins University},
  note      = {Capstone Project, supported by JHU Applied Physics Laboratory}
}
```
