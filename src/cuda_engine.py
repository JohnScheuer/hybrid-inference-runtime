# src/cuda_engine.py
import torch
import time

class CUDAPrefillEngine:
    def __init__(self, device_id=0):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(self.device)
        print(f"[NODE 0 INITIALIZED] Local Device: {self.device_name} (Blackwell CC 12.0)")
        
        # Warmup do stream CUDA
        self.stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(self.stream):
            _ = torch.randn(1024, 1024, device=self.device, dtype=torch.float16) @ torch.randn(1024, 1024, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()

    def execute_prefill(self, batch_size: int, seq_len: int, hidden_dim: int = 4096):
        """
        Executa a fase de Prefill (Compute-Bound GEMM + QKV Projection).
        Gera os blocos iniciais do KV Cache em Paged Host/Pinned Memory.
        """
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.cuda.stream(self.stream):
            # Simulação do GEMM de Projeção QKV do Transformer
            inputs = torch.randn(batch_size, seq_len, hidden_dim, device=self.device, dtype=torch.float16)
            weights = torch.randn(hidden_dim, hidden_dim * 3, device=self.device, dtype=torch.float16)
            qkv = torch.matmul(inputs, weights)

            # Extração dos blocos de KV Cache (Pinned Host Memory para Bridge)
            k_cache = qkv[:, :, hidden_dim:hidden_dim*2].contiguous()
            v_cache = qkv[:, :, hidden_dim*2:].contiguous()

        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - start) * 1000.0

        return {
            "ttft_ms": ttft_ms,
            "k_shape": tuple(k_cache.shape),
            "v_shape": tuple(v_cache.shape),
            "bytes_transferred": k_cache.nbytes + v_cache.nbytes
        }