# AI Telemetry Interpretability Research

**Can operational telemetry reveal what AI models are "thinking" without accessing their internal states?**

This repository contains the code and results from a 7-phase experimental research project that uses the "Alien in a Box" methodology to study AI model behavior through system-level metrics.

## 🔬 Research Question

Can we use operational telemetry (CPU usage, memory patterns, timing, throughput) to understand AI model internal states and reasoning processes without direct access to model weights or activations?

**Answer:** YES — but the signals depend on which abstraction layer you monitor.

## 📊 The Seven Phases

| Phase | Focus | Key Finding |
|-------|-------|-------------|
| **1-2** | Initial telemetry setup | Validated that CPU/memory patterns differ by task type; normalized per-token metrics to fix "length confound" |
| **3** | Temperature manipulation | **"Entropy Signature"** — 73% CPU reduction at temp=0 for factual questions |
| **4** | Base vs instruct models | **"Competence Confound"** — Base models can't follow constraints |
| **5** | Alignment Tax (Llama-3.1-8B) | Constrained prompts used **508% more CPU per token** |
| **6** | Energy measurement | **"Thermodynamic Floor"** — Constant 50W regardless of task difficulty |
| **7** | Tokens-per-second analysis | Format constraints (15.8% TPS reduction) are harder than impossible tasks (8.9%) |

## 🎯 Major Discovery

**"Counting is Harder Than Creativity"** — Format constraints create sustained uncertainty across generation, while content constraints narrow the search space.

## 📁 Key Files

- `phase3_experiment.py` - Temperature experiments
- `phase5_alignment_tax.py` - Alignment tax measurement  
- `phase7_tps_benchmark.py` - Throughput analysis
- Results in `*_results_*.json` files

## 🚀 Quick Start
```bash
python3 -m venv venv
source venv/bin/activate
pip install psutil transformers torch
python telemetry_test_v2.py
```

Research conducted November 22, 2024
