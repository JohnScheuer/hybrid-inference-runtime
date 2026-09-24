# benchmarks/blackwell_gddr7_profile.py
import torch
import time

def benchmark_gddr7_bandwidth():
    print("=" * 65)
    print("  BLACKWELL RTX 5060 GDDR7 MEMORY BANDWIDTH & DECODE PROFILE")
    print("=" * 65)
    
    device = torch.device("cuda:0")
    # Aloca 2GB de tensores na memória GDDR7
    num_elements = 1024 * 1024 * 1024 // 2  # 512M elementos FP16 = 1GB
    tensor_a = torch.randn(num_elements, dtype=torch.float16, device=device)
    tensor_b = torch.empty_like(tensor_a)
    
    # Warmup
    for _ in range(10):
        tensor_b.copy_(tensor_a)
    torch.cuda.synchronize()
    
    # Medição de Bandwidth de Memória Pura (Copy Engine / VRAM Saturation)
    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        tensor_b.copy_(tensor_a)
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    
    # 2 GB lidos + 2 GB escritos = 2GB transferidos por iteração
    total_bytes = 2 * (tensor_a.nbytes) * iterations
    bandwidth_gbs = (total_bytes / duration) / (1024**3)
    
    print(f"Device Name        : {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability : {torch.cuda.get_device_capability(0)}")
    print(f"Total VRAM Allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    print(f"Effective Bandwidth: {bandwidth_gbs:.2f} GB/s (GDDR7 Saturation)")
    print("-" * 65)

def benchmark_decode_gemv_throughput():
    device = torch.device("cuda:0")
    hidden_dim = 4096
    batch_sizes = [1, 4, 8, 16, 32]
    
    print("Batch Size | Decode Time (ms/step) | Token Throughput (tok/s)")
    print("-" * 65)
    
    for b in batch_sizes:
        weight = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device=device)
        x = torch.randn(b, 1, hidden_dim, dtype=torch.float16, device=device)
        
        # Warmup
        for _ in range(20):
            _ = torch.matmul(x, weight)
        torch.cuda.synchronize()
        
        iters = 200
        start = time.perf_counter()
        for _ in range(iters):
            _ = torch.matmul(x, weight)
        torch.cuda.synchronize()
        step_time_ms = ((time.perf_counter() - start) / iters) * 1000.0
        tokens_per_sec = (b / (step_time_ms / 1000.0))
        
        print(f"   b={b:<2}    |        {step_time_ms:.3f} ms        |       {tokens_per_sec:.1f} tok/s")
    print("=" * 65)

if __name__ == "__main__":
    benchmark_gddr7_bandwidth()
    benchmark_decode_gemv_throughput()