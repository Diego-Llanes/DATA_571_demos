import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Initial weights and biases
initial_weights = np.zeros((2, 3))  # 2x3 matrix
initial_biases = np.zeros(3)  # 1x3 vector

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot setup for the right subplot
axs[1].set_xlim(-10, 10)
axs[1].set_ylim(-10, 10)
axs[1].set_title('Decision Boundaries')
axs[1].grid(True)

# Add sliders for weights
weight_sliders = []
for i in range(2):
    for j in range(3):
        ax_weight = plt.axes([0.1 + j*0.15, 0.05 + i*0.05, 0.1, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_weight, f'W{i+1}{j+1}', -5, 5, valinit=0)
        weight_sliders.append(slider)

# Add sliders for biases
bias_sliders = []
for j in range(3):
    ax_bias = plt.axes([0.55 + j*0.15, 0.05, 0.1, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_bias, f'B{j+1}', -10, 10, valinit=0)
    bias_sliders.append(slider)

def update(val=None):
    weights = np.array([[slider.val for slider in weight_sliders[:3]],
                        [slider.val for slider in weight_sliders[3:]]])
    biases = np.array([slider.val for slider in bias_sliders])
    
    # Clear previous lines
    axs[1].clear()
    axs[1].set_xlim(-10, 10)
    axs[1].set_ylim(-10, 10)
    axs[1].grid(True)
    
    # Plot decision boundaries
    x = np.linspace(-10, 10, 100)
    for i in range(3):
        if weights[1, i] != 0:  # Avoid division by zero
            y = (-weights[0, i] * x - biases[i]) / weights[1, i]
            axs[1].plot(x, y, label=f'Class {i+1}')
    
    axs[1].legend()

# Connect the update function to each slider
for slider in weight_sliders + bias_sliders:
    slider.on_changed(update)

update()
plt.show()
