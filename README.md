# Heterogeneous LLM Inference Runtime

A research prototype for disaggregated Prefill-Decode LLM serving across heterogeneous accelerator architectures.

## Current Validation Status

### Node 0: Physical NVIDIA Execution
- Device: NVIDIA GeForce RTX 5060
- Architecture: Blackwell (Compute Capability 12.0)
- Backend: PyTorch CUDA on WSL2
- Workload: FP16 QKV-style prefill GEMM

### Node 1: DaVinci Microarchitectural Model
- Target: Huawei Ascend 910C / DaVinci V300
- Components: Cube, Vector, DMA pipelines, UB hierarchy, HBM 1.6 TB/s
- Purpose: Performance modelling before physical CANN provisioning

*Node 1 is an explicit emulator, not physical CANN execution.*

## Quickstart

Run the hybrid prototype:

```bash
PYTHONPATH=. python3 -m src.orchestrator
```

Run the Blackwell memory profile:

```bash
python3 benchmarks/blackwell_gddr7_profile.py
```

## Roadmap
- Integrate physical Ascend 910B/910C via CANN/ACL over TCP/gRPC
- Add hardware provenance via nvidia-smi and npu-smi
- Poisson arrival workloads with coordinated-omission-safe reporting

## License
MIT

## Author
Joao Felipe de Souza - Senior ML Systems Engineer
GitHub: https://github.com/JohnScheuer
