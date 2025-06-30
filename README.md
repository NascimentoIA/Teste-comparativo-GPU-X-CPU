# Teste-comparativo-GPU-X-CPU
Comparar hardwares diferentes.  Otimizar aplicações que dependam de álgebra linear intensiva.
Descrição do Código: Benchmark de Multiplicação de Matrizes (CPU vs GPU)
Este script compara o desempenho entre CPU e GPU em operações de multiplicação de matrizes usando PyTorch. É útil para avaliar ganhos de performance em hardware específico.

Função Principal: run_benchmark()
Executa o teste de performance em um dispositivo (CPU/GPU) com as seguintes etapas:

Inicialização do Dispositivo

python
device = torch.device(device_name)
Aceita 'cpu' ou 'cuda' como entrada.

Alocação de Memória

python
a = torch.randn(matrix_size, matrix_size, device=device)
b = torch.randn(matrix_size, matrix_size, device=device)
Cria duas matrizes quadradas de tamanho matrix_size x matrix_size no dispositivo especificado.

Fase de Aquecimento

python
for _ in range(10):
    torch.matmul(a, b)
Pré-executa 10 multiplicações para estabilizar o hardware.

Benchmark Principal

python
for i in range(iterations):
    c = torch.matmul(a, b)
Executa a multiplicação iterations vezes, mostrando progresso em tempo real.

Tratamento de Erros

Detecta OutOfMemoryError da GPU e outros erros genéricos.

Parâmetros de Teste (Seção __main__)
python
MATRIX_SIZE = 2048  # Tamanho das matrizes (2048x2048)
ITERATIONS = 1000   # Número de multiplicações
Aviso: Valores altos podem causar lentidão extrema na CPU.

Fluxo de Execução
Teste na CPU

python
cpu_time = run_benchmark('cpu', MATRIX_SIZE, ITERATIONS)
Teste na GPU (se disponível)

python
if torch.cuda.is_available():
    gpu_time = run_benchmark('cuda', MATRIX_SIZE, ITERATIONS)
Resultados Finais

Exibe tempos de execução e calcula o speedup da GPU vs CPU:

python
speedup = cpu_time / gpu_time
print(f"GPU foi {speedup:.2f}x mais rápida que a CPU")
Saída de Exemplo
text
--- Comparativo de Performance: CPU vs GPU ---
Tamanho da Matriz: 2048x2048
Número de Iterações: 1000

--- Iniciando Benchmark no dispositivo: CPU ---
Tempo de execução: 45.72 segundos.

--- Iniciando Benchmark no dispositivo: CUDA ---
GPU: NVIDIA RTX 3090
Tempo de execução: 1.84 segundos.

Conclusão: A GPU foi 24.85x mais rápida que a CPU.
Requisitos e Observações
Dependências

PyTorch instalado com suporte a CUDA (para GPU).

Gerenciamento de Memória

Reduza MATRIX_SIZE se ocorrer CUDA out of memory.

Sincronização

torch.cuda.synchronize() garante medições precisas na GPU.

Como Executar
bash
python benchmark_matrix_mul.py
