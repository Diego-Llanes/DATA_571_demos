# Illustrate linear regression with one variable
#
# Authors: Diego Llanes (primary), Brian Hutchinson
# Date: Winter 2024

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse

def main(args):
    # Generate data
    true_m = np.random.uniform(-1.8, 1.8)
    true_b = np.random.uniform(-0.8, 0.8)

    true_x = np.random.uniform(-1, 1, args.N)
    true_y = true_x * true_m + true_b + np.random.uniform(-args.noise, args.noise, args.N) 

    def R(m, b):
        if args.loss == 'squared_error':
            return (1/args.N) * np.sum((true_x * m + b - true_y) ** 2)
        elif args.loss == 'absolute_error':
            return (1/args.N) * np.sum(np.abs(true_x * m + b - true_y))
        else:
            raise ValueError('Invalid loss function: {}'.format(args.loss))

    # optimal might not be equal to the true
    if args.loss == 'squared_error':
        optimal = np.polyfit(true_x, true_y, 1)
        optimal_m = optimal[0]
        optimal_b = optimal[1]
    elif args.loss == 'absolute_error':
        A = np.vstack([true_x, np.ones_like(true_x)]).T
        coef, _, _, _ = np.linalg.lstsq(A, true_y, rcond=None)
        optimal_m = coef[0]
        optimal_b = coef[1]

    m_bound = np.max(np.abs([true_m, optimal_m])) * 1.1
    b_bound = np.max(np.abs([true_b, optimal_b])) * 1.1

    m_range = np.arange(-m_bound, m_bound, 0.02)
    b_range = np.arange(-b_bound, b_bound, 0.01)
    M, B = np.meshgrid(m_range, b_range)
    Z = np.array([[R(m, b) for m, b in zip(mr, br)] for mr, br in zip(M, B)])


    pred_range = np.arange(-2, 2.01, 0.01)

    def update(val):
        m = s_m.val
        b = s_b.val

        line_pred.set_ydata(pred_range * m + b)

        star._offsets3d = ([m], [b], [R(m, b)])
        fig.canvas.draw_idle()


    initial_m = 0.0
    initial_b = 0.0

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')

    ax1.plot_surface(M, B, Z, edgecolor='k',facecolor='none', alpha=0.1)
    star = ax1.scatter([initial_m], [initial_b], [R(initial_m, initial_b)], color='b', s=50)
    true_star = ax1.scatter([true_m], [true_b], [R(true_m, true_b)], color='r', s=50)
    opt_star = ax1.scatter([optimal_m], [optimal_b], [R(optimal_m, optimal_b)], color='m', s=50)
    ax1.set_xlabel('m')
    ax1.set_ylabel('b')
    ax1.set_zlabel('Risk')
    ax1.set_title('3D Risk Surface')

    ax2 = fig.add_subplot(122)

    line_pred, = ax2.plot(pred_range, pred_range * initial_m + initial_b, '-k')
    if args.plot_true:
        ax2.plot(pred_range, pred_range * true_m + true_b, '--r')
    if args.plot_optimal:
        ax2.plot(pred_range, pred_range * optimal_m + optimal_b, '--m')
    ax2.plot(true_x, true_y, '*')
    ax2.set_title('Prediction Function')

    axcolor = 'lightgoldenrodyellow'
    ax_m = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)
    ax_b = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    s_m = Slider(ax_m, 'm', -2.0, 2.0, valinit=initial_m)
    s_b = Slider(ax_b, 'b', -1.0, 1.0, valinit=initial_b)

    s_m.on_changed(update)
    s_b.on_changed(update)

    plt.tight_layout()
    plt.show()

def parse_all_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='squared_error', help='loss function to use in {squared_error, absolute_error} [default: squared_error]')
    parser.add_argument('--N', type=int, default=5, help='number of training set points [default: 5]')
    parser.add_argument('--noise', type=float, default=0.2, help='amount of noise to add to the data [default: 0.2]')
    parser.add_argument('--plot_true', action='store_true', help='plot true line')
    parser.add_argument('--plot_optimal', action='store_true', help='plot optimal fit line')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_all_args()
    main(args)
