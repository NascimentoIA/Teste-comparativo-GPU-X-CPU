#!/usr/bin/env python3
"""
APLICATIVO DESKTOP - DETECTOR DE CÂNCER DE PELE
Interface gráfica para reconhecimento usando modelo treinado
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import os
import numpy as np
import threading
import time


class SkinCancerDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Detector de Câncer de Pele - IA")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')

        # Configurar estilo
        self.setup_styles()

        # Variáveis
        self.model = None
        self.current_image = None
        self.image_path = None

        # Carregar modelo
        self.load_model()

        # Criar interface
        self.create_widgets()

        # Centralizar janela
        self.center_window()

    def setup_styles(self):
        """Configura estilos da interface"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configurar cores
        style.configure('Title.TLabel',
                        font=('Arial', 16, 'bold'),
                        foreground='#2c3e50',
                        background='#f0f0f0')

        style.configure('Result.TLabel',
                        font=('Arial', 12, 'bold'),
                        background='#f0f0f0')

        style.configure('Success.TLabel',
                        font=('Arial', 12, 'bold'),
                        foreground='#27ae60',
                        background='#f0f0f0')

        style.configure('Warning.TLabel',
                        font=('Arial', 12, 'bold'),
                        foreground='#e74c3c',
                        background='#f0f0f0')

    def load_model(self):
        """Carrega o modelo treinado"""
        try:
            print("🧠 Carregando modelo treinado...")

            # Verificar se o modelo existe
            model_path = 'best_model_optimized.pth'
            if not os.path.exists(model_path):
                messagebox.showerror(
                    "Erro", "Modelo não encontrado! Execute o treinamento primeiro.")
                return

            # Carregar checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')

            # Criar modelo
            self.model = SkinCancerCNN()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            print(
                f"✅ Modelo carregado com sucesso! (Acc: {checkpoint['best_val_acc']:.2f}%)")

        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            messagebox.showerror("Erro", f"Falha ao carregar modelo: {e}")

    def create_widgets(self):
        """Cria os widgets da interface"""

        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Título
        title_label = ttk.Label(main_frame,
                                text="🔬 DETECTOR DE CÂNCER DE PELE",
                                style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Frame da imagem
        image_frame = ttk.LabelFrame(main_frame, text="📸 Imagem", padding="10")
        image_frame.grid(row=1, column=0, columnspan=3,
                         sticky=(tk.W, tk.E), pady=(0, 20))
        image_frame.columnconfigure(0, weight=1)

        # Canvas para imagem
        self.image_canvas = tk.Canvas(
            image_frame, width=400, height=300, bg='white', relief='sunken')
        self.image_canvas.grid(row=0, column=0, pady=(0, 10))

        # Texto padrão no canvas
        self.image_canvas.create_text(200, 150,
                                      text="Clique em 'Selecionar Imagem'\npara carregar uma foto",
                                      font=('Arial', 12),
                                      fill='gray')

        # Botões de imagem
        button_frame = ttk.Frame(image_frame)
        button_frame.grid(row=1, column=0, pady=(0, 10))

        self.select_button = ttk.Button(button_frame,
                                        text="📁 Selecionar Imagem",
                                        command=self.select_image)
        self.select_button.pack(side=tk.LEFT, padx=(0, 10))

        self.analyze_button = ttk.Button(button_frame,
                                         text="🔍 Analisar",
                                         command=self.analyze_image,
                                         state='disabled')
        self.analyze_button.pack(side=tk.LEFT)

        # Frame de resultados
        results_frame = ttk.LabelFrame(
            main_frame, text="📊 Resultados da Análise", padding="10")
        results_frame.grid(row=2, column=0, columnspan=3,
                           sticky=(tk.W, tk.E), pady=(0, 20))
        results_frame.columnconfigure(1, weight=1)

        # Resultados
        self.result_label = ttk.Label(results_frame,
                                      text="Selecione uma imagem para análise",
                                      style='Result.TLabel')
        self.result_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Probabilidades
        self.probability_frame = ttk.Frame(results_frame)
        self.probability_frame.grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Barra de progresso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(results_frame,
                                            variable=self.progress_var,
                                            maximum=100,
                                            length=300)
        self.progress_bar.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        # Status
        self.status_label = ttk.Label(main_frame,
                                      text="✅ Pronto para análise",
                                      style='Success.TLabel')
        self.status_label.grid(row=3, column=0, columnspan=3, pady=(20, 0))

        # Informações do modelo
        info_frame = ttk.LabelFrame(
            main_frame, text="ℹ️ Informações do Modelo", padding="10")
        info_frame.grid(row=4, column=0, columnspan=3,
                        sticky=(tk.W, tk.E), pady=(20, 0))

        if self.model:
            model_info = f"Modelo: ResNet18 + CNN Personalizada | Acurácia: 90.62% | Dataset: 10.015 imagens"
        else:
            model_info = "Modelo não carregado"

        info_label = ttk.Label(
            info_frame, text=model_info, style='Result.TLabel')
        info_label.pack()

    def center_window(self):
        """Centraliza a janela na tela"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def select_image(self):
        """Seleciona imagem do banco de fotos"""
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp"),
                ("Todos os arquivos", "*.*")
            ]
        )

        if file_path:
            try:
                # Carregar imagem
                self.image_path = file_path
                self.current_image = Image.open(file_path)

                # Redimensionar para exibição
                display_image = self.current_image.copy()
                display_image.thumbnail((400, 300), Image.Resampling.LANCZOS)

                # Converter para PhotoImage
                self.photo_image = ImageTk.PhotoImage(display_image)

                # Limpar canvas e mostrar imagem
                self.image_canvas.delete("all")
                self.image_canvas.create_image(
                    200, 150, image=self.photo_image)

                # Habilitar botão de análise
                self.analyze_button.config(state='normal')

                # Atualizar status
                self.status_label.config(
                    text="✅ Imagem carregada - Clique em 'Analisar'")

                # Limpar resultados anteriores
                self.result_label.config(
                    text="Clique em 'Analisar' para processar a imagem")
                self.progress_var.set(0)

                # Limpar probabilidades
                for widget in self.probability_frame.winfo_children():
                    widget.destroy()

            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar imagem: {e}")

    def analyze_image(self):
        """Analisa a imagem usando o modelo treinado"""
        if not self.model or not self.current_image:
            messagebox.showerror("Erro", "Modelo ou imagem não disponível")
            return

        # Desabilitar botões durante análise
        self.select_button.config(state='disabled')
        self.analyze_button.config(state='disabled')

        # Atualizar status
        self.status_label.config(text="🔍 Analisando imagem...")
        self.progress_var.set(25)

        # Executar análise em thread separada
        analysis_thread = threading.Thread(target=self._analyze_image_thread)
        analysis_thread.daemon = True
        analysis_thread.start()

    def _analyze_image_thread(self):
        """Thread para análise da imagem"""
        try:
            # Transformações da imagem
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])

            # Processar imagem
            image_tensor = transform(self.current_image).unsqueeze(0)

            # Atualizar progresso
            self.root.after(0, lambda: self.progress_var.set(50))

            # Predição
            with torch.no_grad():
                output = self.model(image_tensor)
                probability = torch.sigmoid(output).item()

            # Atualizar progresso
            self.root.after(0, lambda: self.progress_var.set(75))

            # Processar resultado na thread principal
            self.root.after(0, lambda: self._show_results(probability))

            # Atualizar progresso final
            self.root.after(0, lambda: self.progress_var.set(100))

        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))

    def _show_results(self, probability):
        """Mostra os resultados da análise"""
        # Calcular probabilidades
        melanoma_prob = probability * 100
        nao_melanoma_prob = (1 - probability) * 100

        # Determinar resultado
        if melanoma_prob > 50:
            result_text = f"⚠️ ALERTA: Possível Melanoma detectado!"
            result_style = 'Warning.TLabel'
        else:
            result_text = f"✅ Provavelmente Não-Melanoma"
            result_style = 'Success.TLabel'

        # Atualizar resultado principal
        self.result_label.config(text=result_text, style=result_style)

        # Limpar frame de probabilidades
        for widget in self.probability_frame.winfo_children():
            widget.destroy()

        # Criar labels de probabilidade
        ttk.Label(self.probability_frame,
                  text="🔴 Melanoma:",
                  style='Result.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        melanoma_label = ttk.Label(self.probability_frame,
                                   text=f"{melanoma_prob:.1f}%",
                                   style='Warning.TLabel')
        melanoma_label.grid(row=0, column=1, sticky=tk.W)

        ttk.Label(self.probability_frame,
                  text="🟢 Não-Melanoma:",
                  style='Result.TLabel').grid(row=1, column=0, sticky=tk.W, padx=(0, 10))

        nao_melanoma_label = ttk.Label(self.probability_frame,
                                       text=f"{nao_melanoma_prob:.1f}%",
                                       style='Success.TLabel')
        nao_melanoma_label.grid(row=1, column=1, sticky=tk.W)

        # Atualizar status
        self.status_label.config(text="✅ Análise concluída")

        # Reabilitar botões
        self.select_button.config(state='normal')
        self.analyze_button.config(state='normal')

    def _show_error(self, error_msg):
        """Mostra erro na análise"""
        self.result_label.config(
            text=f"❌ Erro na análise: {error_msg}", style='Warning.TLabel')
        self.status_label.config(text="❌ Falha na análise")

        # Reabilitar botões
        self.select_button.config(state='normal')
        self.analyze_button.config(state='normal')


class SkinCancerCNN(nn.Module):
    """Modelo CNN para detecção de câncer de pele"""

    def __init__(self, num_classes=1):
        super(SkinCancerCNN, self).__init__()

        # Usar ResNet18 pré-treinado
        self.backbone = models.resnet18(pretrained=True)

        # Head personalizado com dropout
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def main():
    """Função principal"""
    root = tk.Tk()
    app = SkinCancerDetectorApp(root)

    # Configurar ícone (se disponível)
    try:
        root.iconbitmap('icon.ico')
    except:
        pass

    # Iniciar aplicativo
    root.mainloop()


if __name__ == "__main__":
    main()
