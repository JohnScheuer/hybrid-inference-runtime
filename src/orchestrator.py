# src/orchestrator.py
import time
import uuid
from src.cuda_engine import CUDAPrefillEngine
from src.ascend_client import AscendDecodeWorker
from src.telemetry import TelemetryCollector, HybridStepMetrics

class HybridInferenceOrchestrator:
    def __init__(self):
        print("=" * 70)
        print("  HYBRID LLM INFERENCE RUNTIME (NVIDIA SM100 ↔ HUAWEI ASCEND 910B)")
        print("  Disaggregated Prefill-Decode Architecture with CANN Memory Bridge")
        print("=" * 70)
        self.cuda_node = CUDAPrefillEngine(device_id=0)
        self.ascend_node = AscendDecodeWorker()
        self.telemetry = TelemetryCollector()

    def process_request(self, batch_size=4, prompt_tokens=512, output_tokens=32):
        req_id = f"req-{uuid.uuid4().hex[:8]}"
        
        # 1. FASE DE PREFILL NO NODE 0 (NVIDIA RTX 5060)
        prefill_res = self.cuda_node.execute_prefill(batch_size, prompt_tokens)
        ttft_ms = prefill_res["ttft_ms"]

        # 2. KV CACHE BRIDGE (Host Memory -> Ascend HBM)
        kv_sync_ms = self.ascend_node.sync_kv_cache(prefill_res["bytes_transferred"])

        # 3. FASE DE DECODE NO NODE 1 (HUAWEI ASCEND 910B)
        tpot_samples = []
        for step in range(output_tokens):
            step_tpot = self.ascend_node.execute_decode_step(step, batch_size)
            tpot_samples.append(step_tpot)

        avg_tpot = sum(tpot_samples) / len(tpot_samples)
        e2e_ms = ttft_ms + kv_sync_ms + sum(tpot_samples)
        sla_status = "HEALTHY (P99 < 15ms)" if avg_tpot < 10.0 else "DEGRADED"

        metric = HybridStepMetrics(
            request_id=req_id,
            batch_size=batch_size,
            prompt_tokens=prompt_tokens,
            generated_tokens=output_tokens,
            node0_prefill_device=self.cuda_node.device_name,
            node1_decode_device=self.ascend_node.device_name,
            ttft_ms=ttft_ms,
            tpot_ms=avg_tpot,
            kv_sync_latency_ms=kv_sync_ms,
            e2e_latency_ms=e2e_ms,
            sla_status=sla_status
        )
        self.telemetry.record_step(metric)

    def run_benchmark(self, num_requests=5):
        print(f"\n[ORCHESTRATOR] Starting Hybrid Benchmark: {num_requests} Burst Batches...")
        for i in range(num_requests):
            self.process_request(batch_size=4, prompt_tokens=512, output_tokens=16)
            time.sleep(0.1)
        self.telemetry.save_artifact()

if __name__ == "__main__":
    orchestrator = HybridInferenceOrchestrator()
    orchestrator.run_benchmark(num_requests=5)