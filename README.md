# 🔬 DETECTOR DE CÂNCER DE PELE - IA

## 📋 Descrição

Sistema de Inteligência Artificial para detecção de câncer de pele usando Deep Learning. O projeto inclui um modelo CNN treinado com ResNet18 e uma aplicação desktop com interface gráfica para análise de imagens médicas.

## 🚀 Características

- **Modelo CNN**: ResNet18 + arquitetura personalizada
- **Acurácia**: 90.62% em validação
- **Interface**: Aplicativo desktop com Tkinter
- **GPU**: Suporte completo a CUDA/NVIDIA
- **Processamento**: Análise em tempo real de imagens

## 📁 Estrutura do Projeto

```
models/
├── app_detector_cancer.py      # Aplicativo principal
├── best_model_optimized.pth    # Modelo treinado (113MB)
├── README.md                   # Este arquivo
├── README_APLICATIVO.md        # Documentação do aplicativo
├── requirements.txt            # Dependências Python
├── .gitignore                 # Arquivos ignorados pelo Git
└── LICENSE                    # Licença do projeto
```

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **PyTorch** - Framework de Deep Learning
- **TorchVision** - Transformações e modelos pré-treinados
- **PIL/Pillow** - Processamento de imagens
- **Tkinter** - Interface gráfica
- **NumPy** - Computação numérica
- **CUDA** - Aceleração GPU

## 📦 Instalação

### 1. Pré-requisitos

- Python 3.8 ou superior
- NVIDIA GPU com suporte CUDA (recomendado)
- 4GB+ RAM
- 200MB+ espaço em disco

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Verificar GPU (Opcional)

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🎯 Como Usar

### Executar o Aplicativo

```bash
python app_detector_cancer.py
```

### Interface do Usuário

1. **Selecionar Imagem**: Clique em "Selecionar Imagem"
2. **Analisar**: Clique em "Analisar Imagem"
3. **Resultado**: Visualize a probabilidade de melanoma
4. **Histórico**: Veja análises anteriores

## 🧠 Sobre o Modelo

### Arquitetura
- **Backbone**: ResNet18 pré-treinado
- **Head**: Camadas personalizadas com regularização
- **Input**: Imagens 224x224 pixels
- **Output**: Probabilidade binária (melanoma/não-melanoma)

### Performance
- **Acurácia**: 90.62%
- **Tamanho**: 113MB
- **Velocidade**: ~1-2 segundos por imagem
- **GPU**: NVIDIA GeForce GTX 1650+ recomendado

### Treinamento
- **Dataset**: 10.015 imagens médicas reais
- **Classes**: Melanoma vs. Não-Melanoma
- **Augmentação**: Rotação, flip, color jitter
- **Regularização**: Dropout, BatchNorm, Weight Decay

## 📊 Resultados

### Métricas de Validação
- **Acurácia**: 90.62%
- **Precisão**: Alta para detecção de melanoma
- **Recall**: Bom para casos positivos
- **F1-Score**: Balanceado entre precisão e recall

### Casos de Uso
- **Triagem médica** - Primeira avaliação
- **Educação médica** - Treinamento de profissionais
- **Pesquisa** - Estudos de dermatologia
- **Telemedicina** - Consultas remotas

## 🔧 Configuração Avançada

### Variáveis de Ambiente
```bash
export CUDA_VISIBLE_DEVICES=0  # Usar GPU específica
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Otimizar memória
```

### Parâmetros do Modelo
- **Threshold**: 0.5 (configurável no código)
- **Batch Size**: 1 (otimizado para aplicação)
- **Image Size**: 224x224 pixels
- **Normalization**: ImageNet padrão

## 🚨 Limitações e Avisos

### Limitações Técnicas
- **Resolução**: Imagens muito pequenas podem afetar precisão
- **Formato**: Suporte a JPG, PNG, JPEG
- **Tamanho**: Imagens muito grandes são redimensionadas

### Avisos Médicos
- **Não substitui diagnóstico médico profissional**
- **Use apenas para triagem inicial**
- **Sempre consulte um dermatologista**
- **Resultados são probabilísticos**

## 🐛 Solução de Problemas

### Erro: "CUDA não disponível"
```bash
# Verificar instalação PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Erro: "Modelo não carregado"
```bash
# Verificar se o arquivo existe
ls -la best_model_optimized.pth
```

### Erro: "Out of Memory"
```bash
# Reduzir batch size ou usar CPU
export CUDA_VISIBLE_DEVICES=""
```

## 📈 Roadmap

### Versão Atual (v1.0)
- ✅ Modelo CNN treinado
- ✅ Aplicativo desktop
- ✅ Suporte GPU
- ✅ Interface amigável

### Próximas Versões
- 🔄 Aplicativo mobile (Android/iOS)
- 🔄 API REST para integração
- 🔄 Múltiplas classes de câncer
- 🔄 Interface web
- 🔄 Integração com sistemas médicos

## 🤝 Contribuição

### Como Contribuir
1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

### Áreas de Contribuição
- **UI/UX**: Melhorar interface
- **Modelo**: Otimizar arquitetura
- **Documentação**: Expandir docs
- **Testes**: Adicionar testes unitários
- **Deploy**: Automatizar instalação

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍⚕️ Aviso Médico

**IMPORTANTE**: Este software é destinado apenas para fins educacionais e de triagem. Não substitui o diagnóstico médico profissional. Sempre consulte um dermatologista para avaliação médica adequada.

## 📞 Suporte

### Issues
- Abra uma issue no GitHub para bugs
- Use o template de issue fornecido
- Inclua logs e screenshots quando possível

### Contato
- **GitHub**: [@NascimentoIA](https://github.com/NascimentoIA)
- **Email**: [Seu email]
- **Discord**: [Seu servidor]

## 🙏 Agradecimentos

- **Dataset**: ISIC, HAM10000
- **Framework**: PyTorch Team
- **Modelos**: Facebook Research (ResNet)
- **Comunidade**: Contribuidores open source

---

**⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!**

**🔬 Desenvolvido com ❤️ para a comunidade médica**
