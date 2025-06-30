import torch
import time
import sys

def run_benchmark(device_name, matrix_size, iterations):
    """
    Executa um benchmark de multiplicação de matrizes em um dispositivo específico.

    Args:
        device_name (str): O dispositivo para executar o teste ('cpu' ou 'cuda').
        matrix_size (int): A dimensão das matrizes quadradas (matrix_size x matrix_size).
        iterations (int): O número de multiplicações a serem executadas.
    """
    try:
        print(f"\n--- Iniciando Benchmark no dispositivo: {device_name.upper()} ---")
        device = torch.device(device_name)
        
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(device)
            print(f"GPU: {gpu_name}")
        
        print("Alocando tensores na memória...")
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        print("Tensores alocados com sucesso.")

        print("Realizando aquecimento...")
        for _ in range(10):
            c = torch.matmul(a, b)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        print("Aquecimento concluído.")

        print(f"Executando {iterations} iterações de multiplicação de matrizes...")
        start_time = time.time()

        for i in range(iterations):
            c = torch.matmul(a, b)
            sys.stdout.write(f"\rProgresso: {i + 1}/{iterations}")
            sys.stdout.flush()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nBenchmark em {device_name.upper()} concluído em {total_time:.4f} segundos.")
        return total_time

    except torch.cuda.OutOfMemoryError:
        print("\nERRO: Memória da GPU insuficiente (CUDA out of memory). Tente reduzir o 'matrix_size'.")
        return float('inf')
    except Exception as e:
        print(f"\nOcorreu um erro inesperado no dispositivo {device_name}: {e}")
        return float('inf')

if __name__ == "__main__":
    # Parâmetros do teste. AVISO: A CPU pode ser muito lenta com valores altos.
    MATRIX_SIZE = 2048  # Reduzido para um teste mais rápido na CPU
    ITERATIONS = 1000

    print("--- Comparativo de Performance: CPU vs GPU ---")
    print(f"Tamanho da Matriz: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"Número de Iterações: {ITERATIONS}")

    # --- Teste na CPU ---
    cpu_time = run_benchmark('cpu', MATRIX_SIZE, ITERATIONS)
    
    # --- Teste na GPU ---
    gpu_time = float('inf')
    if torch.cuda.is_available():
        gpu_time = run_benchmark('cuda', MATRIX_SIZE, ITERATIONS)
    else:
        print("\n--- Dispositivo CUDA não encontrado. Pulando teste da GPU. ---")

    # --- Resultados Finais ---
    print("\n\n" + "="*30)
    print("      RESULTADO FINAL")
    print("="*30)
    print(f"Tempo de execução na CPU: {cpu_time:.4f} segundos.")
    
    if gpu_time != float('inf'):
        print(f"Tempo de execução na GPU: {gpu_time:.4f} segundos.")
        if cpu_time > 0 and gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"\nConclusão: A GPU foi {speedup:.2f}x mais rápida que a CPU para esta tarefa.")
    else:
        print("Não foi possível calcular a performance da GPU.")

