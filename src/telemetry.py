import time
import json
import os
from dataclasses import dataclass, asdict

@dataclass
class HybridStepMetrics:
    request_id: str
    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    node0_prefill_device: str      # Ex: "NVIDIA RTX 5060 (Blackwell CC 12.0)"
    node1_decode_device: str       # Ex: "Huawei Ascend 910B (CANN 8.0)"
    ttft_ms: float                 # Time To First Token (Prefill no Node 0)
    tpot_ms: float                 # Time Per Output Token (Decode no Node 1)
    kv_sync_latency_ms: float      # Latência de transferência do KV Cache Host Bridge
    e2e_latency_ms: float
    sla_status: str

class TelemetryCollector:
    def __init__(self, output_dir="artifacts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.run_id = f"run_{int(time.time())}"
        self.records = []

    def record_step(self, metric: HybridStepMetrics):
        self.records.append(asdict(metric))
        # Formatação de log de alta autoridade para o console / vídeo
        print(f"\n[HYBRID DISPATCH] Request: {metric.request_id} | Tokens: {metric.prompt_tokens} in -> {metric.generated_tokens} out")
        print(f"  ├─► [NODE 0: NVIDIA SM120]  Prefill Phase  : {metric.ttft_ms:.2f} ms")
        print(f"  ├─► [BRIDGE: HOST SYNC]     KV Sync Latency: {metric.kv_sync_latency_ms:.2f} ms (PagedAttention Blocks)")
        print(f"  ├─► [NODE 1: HUAWEI CANN]   Decode (TPOT)  : {metric.tpot_ms:.2f} ms/tok (Cube/Vector Pipe)")
        print(f"  └─► [E2E LATENCY]           Total: {metric.e2e_latency_ms:.2f} ms | SLA: {metric.sla_status}")

    def save_artifact(self):
        filename = os.path.join(self.output_dir, f"telemetry_{self.run_id}.json")
        with open(filename, "w") as f:
            json.dump({
                "schema_version": "1.0.0",
                "topology": {
                    "node_0": "Local NVIDIA GeForce RTX 5060 (Blackwell CC 12.0)",
                    "node_1": "Remote Huawei Ascend 910B (CANN 8.0)",
                    "protocol": "Disaggregated Prefill-Decode with Async Memory Bridge"
                },
                "metrics": self.records
            }, f, indent=2)
        print(f"\n✅ Telemetry Artifact saved to: {filename}")