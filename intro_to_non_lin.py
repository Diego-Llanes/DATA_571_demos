import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MLRVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Intro To Non-Linearity of SHL NN")

        self.create_widgets()
        self.setup_plot()

    def create_widgets(self):

        # Model Params
        param_frame = ttk.LabelFrame(self.master, text="Model Parameters")
        param_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')

        ## Model Width
        ttk.Label(param_frame, text="Model Width", row=0, padx=5, pady=5)
        self.width = ttk.Combobox(param_frame, width=10)
        self.width['values'] = (8, 16, 32, 64, 128)
        self.width.grid(column=0, row=0)

        ## Model Depth
        ttk.Label(param_frame, text="Model Depth", row=1, padx=5, pady=5)
        self.depth = ttk.Combobox(param_frame, width=10)
        self.depth['values'] = (1, 2, 3)
        self.depth.grid(column=0, row=1)

        # Function to approximate
        fn_frame = ttk.LabelFrame(self.master, text="Function to Approximate")
        fn_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')

        ## fn dropdown
        ttk.Label(fn_frame, text="Function", row=0, padx=5, pady=5)
        self.depth = ttk.Combobox(fn_frame, width=10)
        self.depth['values'] = (1, 2, 3)
        self.depth.grid(column=0, row=1)


    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=3, sticky='nsew')
        self.update_plot()

    def update_plot(self, event=None):
        ...


if __name__ == "__main__":
    root = tk.Tk()
    app = MLRVisualizer(root)
    root.mainloop()
