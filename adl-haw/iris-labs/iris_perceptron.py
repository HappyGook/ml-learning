from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

iris_data    = np.loadtxt('data/iris_data.txt')
species_list = np.loadtxt('data/iris_species.txt', dtype=str).tolist()
columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
df = pd.DataFrame(iris_data, columns=columns)
df['Species'] = species_list
mask = df['Species'].isin(["setosa", "versicolor"])
df2  = df[mask].reset_index(drop=True)
X_iris = df2[["PetalLength", "PetalWidth"]].values                                 # shape (100, 2)
y_iris = (df2['Species'].values == "versicolor").astype(int)             # 0 = setosa, 1 = versicolor
N = X_iris.shape[0]


def train_perceptron(initial_weights, epochs, learning_rate):
    trained_weights = np.copy(initial_weights)
    x_bias = np.hstack([np.ones((N, 1)), X_iris])
    xx_i, yy_i = np.meshgrid(np.linspace(0.5, 5.5, 300),
                             np.linspace(-0.2, 2.0, 300))

    for i in range(epochs):

        # get the activation function for given weights
        a_vec = x_bias @ trained_weights
        y_pred = (a_vec >=0).astype(int)

        # check the accuracy
        accuracy = float(np.mean(y_pred == y_iris))
        error = 1.0 - accuracy
        n_misses = Counter(y_pred!=y_iris)[1]

        # Grid to display
        a_grid = trained_weights[0] + trained_weights[1] * xx_i + trained_weights[2] * yy_i
        y_grid = (a_grid >= 0).astype(int)

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap_bg = ListedColormap(['#ffcccc', '#ccccff'])
        ax.contourf(xx_i, yy_i, y_grid, levels=[-0.5, 0.5, 1.5], cmap=cmap_bg, alpha=0.5)
        ax.contour(xx_i, yy_i, a_grid, levels=[0], colors='black', linewidths=2)

        # Highlight misclassified samples
        mis = y_pred != y_iris
        ax.scatter(X_iris[(y_iris == 0) & ~mis, 0], X_iris[(y_iris == 0) & ~mis, 1], c='red', marker='x', s=60,
                   label='Setosa (0)', zorder=3)
        ax.scatter(X_iris[(y_iris == 1) & ~mis, 0], X_iris[(y_iris == 1) & ~mis, 1], c='blue', marker='o', s=60,
                   label='Versicolor (1)', zorder=3)
        if np.any(mis):
            ax.scatter(X_iris[mis, 0], X_iris[mis, 1], facecolors='none', edgecolors='black', s=160, linewidths=2,
                       label='misclassified', zorder=4)

        ax.set_xlabel('Petal length $x_1$ (cm)')
        ax.set_ylabel('Petal width $x_2$ (cm)')
        ax.set_title(
            f'Performance at the epoch {i}\n Misclassified {n_misses} samples; Error rate of {error*100}%')
        ax.legend(loc='upper left')
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(-0.2, 2.0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # update the weights
        trained_weights += learning_rate * (x_bias.T @ (y_iris - y_pred)) / N

if __name__ == '__main__':
    train_perceptron(initial_weights=np.array([-1,1,0.8]),epochs=10, learning_rate=0.5)