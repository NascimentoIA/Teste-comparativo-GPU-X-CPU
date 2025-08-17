# 🔬 APLICATIVO DESKTOP - DETECTOR DE CÂNCER DE PELE

## 📋 Descrição

Aplicativo desktop com interface gráfica para detecção de câncer de pele usando Inteligência Artificial. O aplicativo utiliza um modelo CNN treinado com mais de 10.000 imagens médicas reais.

## ✨ Características

- **🎯 Interface Intuitiva**: Interface gráfica moderna e fácil de usar
- **🔍 Análise em Tempo Real**: Processamento rápido de imagens
- **📊 Resultados Detalhados**: Probabilidades e classificações precisas
- **🖼️ Suporte a Múltiplos Formatos**: JPG, PNG, BMP, etc.
- **💾 Modelo Pré-treinado**: Usa modelo com 90.62% de acurácia
- **🚀 Processamento Otimizado**: Threading para interface responsiva

## 🚀 Instalação

### 1. Pré-requisitos
- Python 3.8 ou superior
- Modelo treinado (`best_model_optimized.pth`)
- Dependências Python

### 2. Instalar Dependências
```bash
python instalar_app.py
```

### 3. Verificar Modelo
Certifique-se de que o arquivo `best_model_optimized.pth` existe no diretório.

## 🎯 Como Usar

### 1. Executar Aplicativo
```bash
python app_detector_cancer.py
```

### 2. Interface do Usuário
- **📁 Selecionar Imagem**: Escolha uma foto do banco de imagens
- **🔍 Analisar**: Processa a imagem usando IA
- **📊 Resultados**: Visualiza probabilidades e classificação

### 3. Interpretar Resultados
- **🔴 Melanoma**: Probabilidade > 50% indica possível melanoma
- **🟢 Não-Melanoma**: Probabilidade < 50% indica lesão benigna
- **📊 Probabilidades**: Percentuais detalhados para cada classe

## 📁 Estrutura de Arquivos

```
models/
├── app_detector_cancer.py      # Aplicativo principal
├── instalar_app.py             # Script de instalação
├── requirements_app.txt         # Dependências
├── best_model_optimized.pth    # Modelo treinado
└── README_APLICATIVO.md        # Este arquivo
```

## 🔧 Dependências

- **PyTorch**: Framework de deep learning
- **TorchVision**: Visão computacional
- **Pillow**: Processamento de imagens
- **NumPy**: Computação numérica
- **tkinter**: Interface gráfica (incluído no Python)

## 📱 Próximos Passos

### Versão Mobile
- [ ] Desenvolvimento para Android/iOS
- [ ] Interface touch otimizada
- [ ] Câmera integrada
- [ ] Armazenamento local de resultados

### Melhorias
- [ ] Histórico de análises
- [ ] Exportação de relatórios
- [ ] Múltiplas imagens simultâneas
- [ ] Integração com sistemas médicos

## ⚠️ Importante

- **Este é um sistema de apoio diagnóstico**
- **NÃO substitui avaliação médica profissional**
- **Use apenas para triagem inicial**
- **Sempre consulte um dermatologista**

## 🎯 Performance

- **Acurácia**: 90.62% (validação)
- **Tempo de Processamento**: ~2-5 segundos por imagem
- **Tamanho do Modelo**: ~35 MB
- **Suporte**: Imagens de 224x224 pixels (redimensionadas automaticamente)

## 🆘 Suporte

Para problemas ou dúvidas:
1. Verifique se todas as dependências estão instaladas
2. Confirme se o modelo existe no diretório
3. Use imagens em formatos suportados
4. Verifique logs no terminal

## 📄 Licença

Este projeto é para uso educacional e de pesquisa. Consulte a licença para detalhes de uso comercial.

---

**🔬 Desenvolvido com Inteligência Artificial para Saúde Pública**
