import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth):
        super(SimpleNN, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NonLinVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Intro To Non-Linearity of SHL NN")

        self.model = SimpleNN(1, 64, 1, 1)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.is_playing = False
        self.play_pause_button = None
        self.create_widgets()
        self.setup_plot()
        self.create_model()

    def create_model(self):
        width = int(self.width.get()) if self.width.get() else 64
        depth = int(self.depth.get()) if self.depth.get() else 1
        self.model = SimpleNN(1, width, 1, depth)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def toggle_play_pause(self):
        self.is_playing = not self.is_playing
        self.play_pause_button.config(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self.update_model()

    def update_model(self):
        if self.is_playing:
            self.grad_step()
            self.master.after(1, self.update_model)

    def create_widgets(self):
        # Model Params
        param_frame = ttk.LabelFrame(self.master, text="Model Parameters")
        param_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')

        ## Model Width
        ttk.Label(param_frame, text="Model Width").grid(row=0, column=0, padx=5, pady=5)
        self.width = ttk.Combobox(param_frame, width=10)
        self.width['values'] = (8, 16, 32, 64, 128)
        self.width.grid(row=0, column=1)

        ## Model Depth
        ttk.Label(param_frame, text="Model Depth").grid(row=1, column=0, padx=5, pady=5)
        self.depth = ttk.Combobox(param_frame, width=10)
        self.depth['values'] = (1, 2, 3)
        self.depth.grid(row=1, column=1)

        # Function to approximate
        fn_frame = ttk.LabelFrame(self.master, text="Function to Approximate")
        fn_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')

        ## fn dropdown
        ttk.Label(fn_frame, text="Function").grid(row=0, column=0, padx=5, pady=5)
        self.function = ttk.Combobox(fn_frame, width=10)
        self.function['values'] = ('sin', 'cos', 'exp')  # Example functions
        self.function.grid(row=0, column=1)

        self.play_pause_button = ttk.Button(self.master, text="Play", command=self.toggle_play_pause)
        self.play_pause_button.grid(row=2, column=0, pady=5)

        self.width.bind('<<ComboboxSelected>>', lambda e: self.create_model())
        self.depth.bind('<<ComboboxSelected>>', lambda e: self.create_model())
        self.function.bind('<<ComboboxSelected>>', lambda e: self.update_plot())


    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=3, sticky='nsew')
        self.update_plot()

    def update_plot(self, event=None):
        x = np.linspace(-10, 10, 100)
        y_true = self.get_function()(x)

        # Convert to PyTorch tensors
        x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
        y_tensor = torch.tensor(y_true.reshape(-1, 1), dtype=torch.float32)

        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x_tensor).numpy()

        # Plotting
        self.ax.clear()
        self.ax.plot(x, y_true, label='True Function')
        self.ax.plot(x, y_pred, label='NN Approximation')
        self.ax.legend()
        self.canvas.draw()

    def get_function(self):
        selected_fn = self.function.get()
        fns = {'sin': np.sin, 'cos': np.cos, 'exp': np.exp, '2x+1': lambda x: 2 * x + 1, '': lambda x: x}
        return fns[selected_fn]

    def grad_step(self):
        x = np.linspace(-10, 10, 1000)
        y_true = self.get_function()(x)

        # Convert to PyTorch tensors
        x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
        y_tensor = torch.tensor(y_true.reshape(-1, 1), dtype=torch.float32)

        # Training step
        self.model.train()
        y_pred = self.model(x_tensor)
        loss = self.loss_fn(y_pred, y_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_plot()


if __name__ == "__main__":
    root = tk.Tk()
    app = NonLinVisualizer(root)
    root.mainloop()
