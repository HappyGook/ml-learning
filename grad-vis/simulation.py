import numpy as np
import optimizers as opt
import functions as f
import render as r
from manim import tempconfig
import matplotlib.pyplot as plt


# Additional matplotlib render
def render_paths(function, classic, momentum):

    x,y = np.meshgrid(np.linspace(-2,2,300),np.linspace(-2,2,300))
    z = np.vectorize(lambda u, v: function([u, v]))(x, y)

    fig, ax = plt.subplots(figsize=(6,6))

    # Filled contour
    contour = ax.contourf(x, y, z, levels=40, cmap='RdYlGn_r')
    fig.colorbar(contour, ax=ax, label='Loss')
    ax.contour(x,y,z, levels=15, colors='white', linewidths=0.4, alpha=0.4)

    # Plot the paths
    ax.plot(
        classic[:, 0], classic[:, 1],
        color='dodgerblue', marker='o', markersize=3,
        linewidth=1.5, label='Gradient descent', zorder=5
    )
    ax.plot(
        momentum[:, 0], momentum[:, 1],
        color='tomato', marker='o', markersize=3,
        linewidth=1.5, label='Momentum', zorder=5
    )

    # Mark start and end
    ax.scatter(*classic[0], color='white', s=60, zorder=6, edgecolors='dodgerblue', linewidths=1.5)
    ax.scatter(*classic[-1], color='dodgerblue', s=80, marker='*', zorder=6)
    ax.scatter(*momentum[-1], color='tomato', s=80, marker='*', zorder=6)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Optimization paths (2D view)')
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('paths_2d.png', dpi=150)
    plt.show()

# Run of the project
# Pre-defined configs, to avoid random exploding by user inputs
configs = {
    1: {
        "f": f.bowl, "grad": f.grad_bowl,
        "step": 0.08, "iters": 40, "alpha": 0.4,
        "x0": np.array([-1.5, 1.5]),
        "domain": (-2, 2),
    },
    2: {
        "f": f.valley, "grad": f.grad_valley,
        "step": 0.04, "iters": 60, "alpha": 0.4,
        "x0": np.array([-1.5, 0.5]),   # off-axis for better zigzags
        "domain": (-2, 2),
    },
    3: {
        "f": f.rosenbrock, "grad": f.grad_rb,
        "step": 0.001, "iters": 60, "alpha": 0.4,
        "x0": np.array([-1.5, -2.0]),
        "domain": (-2, 2),
    },
    4: {
        "f": f.saddle, "grad": f.grad_saddle,
        "step": 0.08, "iters": 40, "alpha": 0.4,
        "x0": np.array([0.5, 0.2]),    # asymetry for saddle dynamics
        "domain": (-2, 2),
    },
}

choice = 0
while not 1<=choice<=4:
    choice = int(input("Choose which surface function do you want to compute a descent for:\n"
                   "1 - convex bowl (f(x,y) = x^2 + y^2)\n"
                   "2 - elliptical valley  (f(x,y) = x^2 - 10*y^2)\n"
                   "3 - Rosenbrock valley (f(x,y) = (1 - x)^2 + 100(y-x^2)^2)\n"
                   "4 - Saddle (f(x,y) = x^2 - y^2)\n"))

"""
step_size = float(input("Step size: "))
iterations = int(input("Number of iterations:"))
alpha = float(input("Choose an alpha for momentum descent: "))
x0 = np.array([
    float(input("Initial position (X): ")),
    float(input("Initial position (Y): "))
])
"""

# Load config values
cfg = configs[choice]
step_size = cfg['step']
iterations = cfg['iters']
alpha = cfg['alpha']
x0 = cfg['x0']
gradient = cfg["grad"]
func = cfg["f"]

classic_path = opt.classic_descent(
    x0,
    step_size=step_size,
    iters=iterations,
    gradient=gradient
)

momentum_path = opt.momentum_descent(
        x0,
        step_size=step_size,
        iters=iterations,
        gradient=gradient,
        alpha=alpha
    )

print(f"Computed paths: "
      f"\n Classic descent: {classic_path}"
      f"\n Momentum descent: {momentum_path}")

render_paths(func, classic_path, momentum_path)

with tempconfig({
    "quality": "medium_quality",
    "preview": True,
    "output_file": "gradient_descent"
}):
    scene = r.GradientScene(
        func=func,
        classic_path=classic_path,
        momentum_path=momentum_path
    )
    scene.render()