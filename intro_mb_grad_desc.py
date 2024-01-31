# Illustrate Gradient Descent
#
# Authors: Diego Llanes (primary), Brian Hutchinson
# Date: Winter 2024

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import argparse


def main(args):
    # Generate data
    true_m = np.random.uniform(-1.8, 1.8)
    true_b = np.random.uniform(-0.8, 0.8)

    true_x = np.random.uniform(-1, 1, args.N)
    true_y = true_x * true_m + true_b + np.random.normal(0, args.noise, args.N)

    def R(m, b):
        if args.loss == 'squared_error':
            return (1/args.N) * np.sum((true_x * m + b - true_y) ** 2)
        elif args.loss == 'absolute_error':
            return (1/args.N) * np.sum(np.abs(true_x * m + b - true_y))
        else:
            raise ValueError('Invalid loss function: {}'.format(args.loss))

    # optimal might not be equal to the true
    optimal = np.polyfit(true_x, true_y, 1)
    optimal_m = optimal[0]
    optimal_b = optimal[1]

    m_bound = np.max(np.abs([true_m, optimal_m])) * 1.1
    b_bound = np.max(np.abs([true_b, optimal_b])) * 1.1

    m_range = np.arange(-m_bound * 2, m_bound * 2, 0.05)
    b_range = np.arange(-b_bound * 2, b_bound * 2, 0.05)

    M, B = np.meshgrid(m_range, b_range)
    Z = np.array([[R(m, b) for m, b in zip(mr, br)] for mr, br in zip(M, B)])

    pred_range = np.arange(-2, 2.01, 0.01)

    initial_m = 0.0
    initial_b = 0.0

    m = initial_m
    b = initial_b

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

    def update(val):
        nonlocal m, b, star, M, B, opt_star, true_star
        lr = s_lr.val
        batch_indices = np.random.choice(args.N, args.mb, replace=False)
        batch_x = true_x[batch_indices]
        batch_y = true_y[batch_indices]

        def R_batch(m, b):
            return (1/args.mb) * np.sum((batch_x * m + b - batch_y) ** 2)

        if args.loss == 'squared_error':
            grad_m = (2/args.mb) * np.sum((batch_x * m + b - batch_y) * batch_x)
            grad_b = (2/args.mb) * np.sum(batch_x * m + b - batch_y)
        elif args.loss == 'absolute_error':
            raise NotImplementedError()

        m = m - lr * grad_m
        b = b - lr * grad_b

        line_pred.set_ydata(pred_range * m + b)
        star.remove()
        star = ax1.scatter([m], [b], [R(m, b)], color='b', s=50)

        # Reduce the number of polygons for performance
        M_coarse, B_coarse = np.meshgrid(np.linspace(-m_bound * 2, m_bound * 2, 20), np.linspace(-b_bound * 2, b_bound * 2, 20))
        
        Z_full = np.array([[R(m_, b_) for m_, b_ in zip(mr, br)] for mr, br in zip(M_coarse, B_coarse)])
        Z_batch = np.array([[R_batch(m_, b_) for m_, b_ in zip(mr, br)] for mr, br in zip(M_coarse, B_coarse)])
        
        ax1.clear()

        true_star = ax1.scatter([true_m], [true_b], [R(true_m, true_b)], color='r', s=50)
        opt_star = ax1.scatter([optimal_m], [optimal_b], [R(optimal_m, optimal_b)], color='m', s=50)

        opt_mb = np.polyfit(batch_x, batch_y, 1)
        opt_mb_m = opt_mb[0]
        opt_mb_b = opt_mb[1]
        opt_star = ax1.scatter([opt_mb_m], [opt_mb_b], [R(opt_mb_m, opt_mb_b)], color='g', s=50)

        ax1.plot_surface(M_coarse, B_coarse, Z_full, color='lightblue', edgecolor='none', alpha=0.5)
        ax1.plot_surface(M_coarse, B_coarse, Z_batch, color='lightgreen', edgecolor='none', alpha=0.5)
        star = ax1.scatter([m], [b], [R(m, b)], color='b', s=50)

        ax1.set_zlim(Z.min(), Z.max())  # Set fixed Z-axis range

        fig.canvas.draw_idle()

    def reset(val=None):
        nonlocal m, b, true_x, true_y, star, true_star, opt_star, line_pred, true_m, true_b, optimal_b, optimal_m
        true_m = np.random.uniform(-1.8, 1.8)
        true_b = np.random.uniform(-0.8, 0.8)
        true_x = np.random.uniform(-1, 1, args.N)

        true_y = true_x * true_m + true_b + np.random.normal(0, args.noise, args.N)

        optimal = np.polyfit(true_x, true_y, 1)
        optimal_m = optimal[0]
        optimal_b = optimal[1]

        m, b = initial_m, initial_b

        s_lr.set_val(0.01)

        ax2.clear()
        line_pred, = ax2.plot(pred_range, pred_range * m + b, '-k', label='Prediction')
        ax2.plot(pred_range, pred_range * true_m + true_b, '--r', label='True')
        ax2.plot(pred_range, pred_range * optimal_m + optimal_b, '--m', label='Optimal')
        ax2.plot(true_x, true_y, '*', label='Data')
        ax2.legend()
        ax2.set_title('Prediction Function')

        Z = np.array([[R(m_, b_) for m_, b_ in zip(mr, br)] for mr, br in zip(M, B)])
        ax1.clear()
        ax1.set_xlim(-m_bound * 2, m_bound * 2)
        ax1.set_ylim(-b_bound * 2, b_bound * 2)
        ax1.plot_surface(M, B, Z, edgecolor='none', alpha=0.5)
        star = ax1.scatter([m], [b], [R(m, b)], color='b', s=50, label='Current')
        true_star = ax1.scatter([true_m], [true_b], [R(true_m, true_b)], color='r', s=50, label='True')
        opt_star = ax1.scatter([optimal_m], [optimal_b], [R(optimal_m, optimal_b)], color='m', s=50, label='Optimal')
        ax1.legend()

        ax1.set_xlabel('m')
        ax1.set_ylabel('b')
        ax1.set_zlabel('Risk')
        ax1.set_title('3D Risk Surface')

        fig.canvas.draw_idle()


    axcolor = 'lightgoldenrodyellow'
    ax_grad = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor=axcolor)
    ax_lr =   plt.axes([0.2, 0.06, 0.65, 0.03], facecolor=axcolor)

    s_lr = Slider(ax_lr, 'Learning Rate', 0.0001, 2, valinit=0.01)
    b_grad = Button(ax_grad, 'Grad Step')
    ax_reset = plt.axes([0.09, 0.02, 0.1, 0.04], facecolor=axcolor)
    b_reset = Button(ax_reset, 'Reset')
    b_reset.on_clicked(reset)

    b_grad.on_clicked(update)

    plt.tight_layout()
    reset()
    plt.show()


def parse_all_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='squared_error', help='loss function to use in {squared_error, absolute_error} [default: squared_error]')
    parser.add_argument('--N', type=int, default=10, help='number of training set points [default: 5]')
    parser.add_argument('--noise', type=float, default=0.2, help='amount of noise to add to the data [default: 0.2]')
    parser.add_argument('--plot_true', action='store_true', help='plot true line')
    parser.add_argument('--plot_optimal', action='store_true', help='plot optimal fit line')
    parser.add_argument('--mb', type=int, default=4, help='size of each minibatch [default: 2]')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_all_args()
    main(args)
