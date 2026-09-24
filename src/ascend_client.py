# src/ascend_client.py
import time
from src.davinci_emulator import AscendDaVinciEmulator, DaVinciHardwareSpec

class AscendDecodeWorker:
    def __init__(self, endpoint="local-davinci-emulator:v300"):
        self.endpoint = endpoint
        self.spec = DaVinciHardwareSpec()
        self.emulator = AscendDaVinciEmulator(self.spec)
        self.device_name = f"{self.spec.chip_name} [3-Pipe Microarchitectural Model]"
        print(f"[NODE 1 INITIALIZED] Target: {self.device_name}")

    def sync_kv_cache(self, bytes_to_sync: int) -> float:
        """
        Simula a sincronização DMA HBM <-> Host Bridge do PagedAttention
        """
        start = time.perf_counter()
        dma_latency_ms = self.emulator.simulate_dma_transfer(bytes_to_sync)
        time.sleep(dma_latency_ms / 1000.0)
        return (time.perf_counter() - start) * 1000.0

    def execute_decode_step(self, step_idx: int, batch_size: int, hidden_dim: int = 4096) -> float:
        """
        Executa o passo de Decode no modelo micro-arquitetural DaVinci
        """
        return self.emulator.execute_decode_step(step_idx, batch_size, hidden_dim)