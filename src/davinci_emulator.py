# src/davinci_emulator.py
import time
import math
import torch
from dataclasses import dataclass

@dataclass
class DaVinciHardwareSpec:
    chip_name: str = "Huawei Ascend 910C (DaVinci V300)"
    num_ai_cores: int = 32
    hbm_bandwidth_gbs: float = 1600.0       # 1.6 TB/s HBM3 (1600 GB/s)
    cube_peak_tflops_fp16: float = 380.0    # 380 TFLOPS
    vector_peak_tflops_fp16: float = 190.0  # 190 TFLOPS
    unified_buffer_kb_per_core: int = 256   # 256 KB UB por Core
    l0c_buffer_kb_per_core: int = 256       # 256 KB Accumulator por Core
    dma_startup_overhead_us: float = 1.2    # Latência de inicialização de barramento (1.2 microssegundos)

class AscendDaVinciEmulator:
    def __init__(self, spec: DaVinciHardwareSpec = DaVinciHardwareSpec()):
        self.spec = spec
        print(f"[DAVINCI EMULATOR LOADED] Target Silicon: {self.spec.chip_name}")
        print(f"  ├─► Compute Cores : {self.spec.num_ai_cores} AI Cores (Cube + Vector + Scalar)")
        print(f"  ├─► Memory Engine : {self.spec.hbm_bandwidth_gbs} GB/s HBM3 | 256KB On-chip UB/Core")
        print(f"  └─► Peak Compute  : {self.spec.cube_peak_tflops_fp16} TFLOPS Cube / {self.spec.vector_peak_tflops_fp16} TFLOPS Vector")

    def simulate_dma_transfer(self, bytes_to_transfer: int) -> float:
        """
        Calcula o tempo de transferência no DMA Pipe (MTE2/MTE3: HBM <-> UB)
        """
        bandwidth_bytes_per_sec = self.spec.hbm_bandwidth_gbs * (1024**3)
        transfer_time_s = bytes_to_transfer / bandwidth_bytes_per_sec
        total_time_ms = (transfer_time_s * 1000.0) + (self.spec.dma_startup_overhead_us / 1000.0)
        return total_time_ms

    def simulate_cube_gemv(self, batch_size: int, hidden_dim: int, num_layers: int = 32) -> float:
        """
        Calcula o tempo de execução no Cube Unit para um LLM completo (32 camadas).
        Em cada camada do Decode, a NPU lê a matriz de pesos de projeção da HBM3.
        """
        # 1. Volume de Pesos (Weight Volume): peso por camada = hidden * hidden * 2 (FP16) * 4 (Q,K,V,Out projections)
        weight_bytes_per_layer = (hidden_dim * hidden_dim * 2) * 4
        total_weight_bytes = weight_bytes_per_layer * num_layers  # Ex: ~1.07 GB de leitura para 32 camadas
        
        # 2. DMA Latency (Gargalo de Memória)
        bandwidth_bytes_sec = self.spec.hbm_bandwidth_gbs * (1024**3)
        weight_dma_time_s = total_weight_bytes / bandwidth_bytes_sec
        
        # 3. Compute Latency (Capacidade de Processamento)
        # FLOPs = 2 * batch * hidden_dim * hidden_dim * 4 (projeções) * 32 camadas
        flops = 2.0 * batch_size * hidden_dim * hidden_dim * 4 * num_layers
        peak_flops_per_sec = self.spec.cube_peak_tflops_fp16 * (10**12)
        compute_time_s = flops / peak_flops_per_sec
        
        # O Decode é estritamente memory-bound no carregamento de pesos da HBM3 (Roofline)
        cube_step_ms = max(weight_dma_time_s, compute_time_s) * 1000.0
        return cube_step_ms

    def simulate_vector_pipe(self, batch_size: int, hidden_dim: int, num_layers: int = 32) -> float:
        """
        Calcula o tempo no Vector Unit para operações element-wise de ativações,
        RMSNorm, RoPE e Softmax ao longo das 32 camadas.
        """
        # Operações de ativação/norm por token por camada
        vector_ops_per_layer = batch_size * hidden_dim * 12  
        total_vector_ops = vector_ops_per_layer * num_layers
        peak_vector_flops_sec = self.spec.vector_peak_tflops_fp16 * (10**12)
        
        vector_time_ms = (total_vector_ops / peak_vector_flops_sec) * 1000.0
        
        # Piso físico de latência de despacho para 32 chamadas consecutivas de kernel vetorial
        kernel_dispatch_overhead_ms = 0.015 * num_layers  # 15us por dispatch
        return max(vector_time_ms, kernel_dispatch_overhead_ms)

    def execute_decode_step(self, step_idx: int, batch_size: int, hidden_dim: int = 4096, num_layers: int = 32) -> float:
        """
        Executa a emulação de passo de geração de token (Decode) em 32 camadas,
        modelando o pipeline assíncrono com flags SetFlag/WaitFlag.
        """
        # Medições individuais
        t_dma_kv = self.simulate_dma_transfer(batch_size * hidden_dim * 2 * num_layers) # Leitura do KV cache
        t_cube = self.simulate_cube_gemv(batch_size, hidden_dim, num_layers)            # GEMVs de Projeção
        t_vector = self.simulate_vector_pipe(batch_size, hidden_dim, num_layers)        # SwiGLU / RMSNorm / RoPE
        
        # Sincronização e overlapping de pipeline via hardware flags (SetFlag/WaitFlag)
        # O hardware consegue sobrepor o carregamento do KV Cache (MTE2) com o processamento do Cube Unit.
        overlap_latency_ms = max(t_dma_kv, t_cube)
        
        # O Vector Unit opera em série sobre o output do Cube Unit
        bubble_overhead_ms = 0.002 * num_layers  # Overhead de sincronização inter-cores (2us por camada)
        step_latency_ms = overlap_latency_ms + t_vector + bubble_overhead_ms
        
        # Pausa real controlada pelo relógio físico para simular o clock do acelerador remoto
        time.sleep(step_latency_ms / 1000.0)
        return step_latency_ms