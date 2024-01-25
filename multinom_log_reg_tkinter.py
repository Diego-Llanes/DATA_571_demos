import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MLRVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Multinomial Logistic Regression Visualization")

        self.weights = np.zeros((2, 3))
        self.biases = np.zeros(3)

        self.create_widgets()
        self.setup_plot()

    def create_widgets(self):
        weight_frame = ttk.LabelFrame(self.master, text="Weights")
        weight_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')
        self.weight_entries = []
        for i in range(2):
            row_entries = []
            for j in range(3):
                entry = ttk.Entry(weight_frame, width=10)
                entry.insert(0, '0')
                entry.grid(row=i, column=j, padx=5, pady=5)
                entry.bind("<Return>", self.update_plot)
                row_entries.append(entry)
            self.weight_entries.append(row_entries)
        bias_frame = ttk.LabelFrame(self.master, text="Biases")
        bias_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')

        self.bias_entries = []
        for i in range(3):
            entry = ttk.Entry(bias_frame, width=10)
            entry.insert(0, '0')
            entry.grid(row=0, column=i, padx=5, pady=5)
            entry.bind("<Return>", self.update_plot)
            self.bias_entries.append(entry)

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=3, sticky='nsew')
        self.update_plot()

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0))
        return exp_z / exp_z.sum(axis=0)

    def update_plot(self, event=None):
        weights = np.array([[float(entry.get()) for entry in row] for row in self.weight_entries])
        biases = np.array([float(entry.get()) for entry in self.bias_entries])
        self.ax.clear()
        self.plot_decision_boundaries(weights, biases)
        self.canvas.draw()

    def plot_decision_boundaries(self, weights, biases):
        x1, x2 = np.meshgrid(np.linspace(-10, 10, 300), np.linspace(-10, 10, 300))
        grid = np.c_[x1.ravel(), x2.ravel()]
        z = np.dot(grid, weights) + biases
        probabilities = self.softmax(z.T).T
        rgb_values = np.clip(probabilities * 255, 0, 255).astype(int)
        rgb_image = rgb_values.reshape(x1.shape[0], x1.shape[1], 3)
        self.ax.clear()
        self.ax.imshow(rgb_image, extent=(-10, 10, -10, 10), origin='lower')

        cols = {
            0: 'r',
            1: 'g',
            2: 'b'
        }
        for i in range(3):
            if weights[1, i] != 0:
                y_vals = (-weights[0, i] * np.linspace(-10, 10, 100) - biases[i]) / weights[1, i]
                self.ax.plot(np.linspace(-10, 10, 100), y_vals, '--', lw=2, color=cols[i])
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel('X1')
        self.ax.set_ylabel('X2')
        self.ax.set_title('Softmax Probabilities with Hyperplanes')


if __name__ == "__main__":
    root = tk.Tk()
    app = MLRVisualizer(root)
    root.mainloop()
