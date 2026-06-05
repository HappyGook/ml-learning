import numpy as np
from matplotlib import pyplot as plt, cm
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist


def sigmoid(x):
    # clip to survive overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_output):
    # assumes sigmoid already computed
    return sigmoid_output * (1 - sigmoid_output)

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)

class AutoEncoder:
    def __init__(self, learning_rate = 0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.out = None
        self.a6 = None
        self.h5 = None
        self.a5 = None
        self.h4 = None
        self.a4 = None
        self.z = None
        self.a3 = None
        self.h2 = None
        self.a2 = None
        self.h1 = None
        self.a1 = None
        self.x = None
        self.lr = learning_rate

        # initialize weights
        self.W1 = np.random.randn(256, 784) * np.sqrt(2 / 784)
        self.W2 = np.random.randn(64, 256) * np.sqrt(2 / 256)
        self.W3 = np.random.randn(16, 64) * np.sqrt(2 / 64)
        self.W4 = np.random.randn(64, 16) * np.sqrt(2 / 16)
        self.W5 = np.random.randn(256, 64) * np.sqrt(2 / 64)
        self.W6 = np.random.randn(784, 256) * np.sqrt(2 / 256)

        # initialize biases
        self.b1 = np.zeros(256)
        self.b2 = np.zeros(64)
        self.b3 = np.zeros(16)

        self.b4 = np.zeros(64)
        self.b5 = np.zeros(256)
        self.b6 = np.zeros(784)

        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.t = 0  # timestep

        # first and second moment buffers for every parameter
        self._init_adam_buffers()

    def _init_adam_buffers(self):
        shapes = {
            'W1': (256, 784), 'W2': (64, 256), 'W3': (16, 64),
            'W4': (64, 16), 'W5': (256, 64), 'W6': (784, 256),
            'b1': (256,), 'b2': (64,), 'b3': (16,),
            'b4': (64,), 'b5': (256,), 'b6': (784,),
        }
        self.m = {k: np.zeros(s) for k, s in shapes.items()}
        self.v = {k: np.zeros(s) for k, s in shapes.items()}

    def encode(self, X):
        self.x = X.flatten()
        # layer 1
        self.a1 = self.W1 @ self.x + self.b1
        self.h1 = relu(self.a1) # (256,)

        #layer 2
        self.a2 = self.W2 @ self.h1 + self.b2
        self.h2 = relu(self.a2) # (64,)

        # layer 3 - latent space
        self.a3 = self.W3 @ self.h2 + self.b3
        self.z = self.a3 # latent space of (16,)
        return self.z

    def decode(self, z):
        # layer 1
        self.a4 = self.W4 @ z + self.b4
        self.h4 = relu(self.a4)

        # layer 2
        self.a5 = self.W5 @ self.h4 + self.b5
        self.h5 = relu(self.a5)

        # output layer (3)
        self.a6 = self.W6 @ self.h5 + self.b6
        self.out = sigmoid(self.a6)

        return self.out.reshape((28, 28))

    def forward_pass(self, X):
        z = self.encode(X)
        decoded = self.decode(z)
        return decoded

    def mse_loss(self, X, decoded):
        flat_x = X.flatten()
        flat_decoded = decoded.flatten()
        return np.mean((flat_x - flat_decoded) ** 2)

    def _adam_step(self, param_name, grad):
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * grad ** 2
        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def backprop(self):
        self.t += 1
        n = self.x.size

        # output layer
        d_loss_out = (2/n) * (self.out - self.x)
        delta6 = d_loss_out * sigmoid_derivative(self.out)
        dW6 = np.outer(delta6, self.h5)
        db6 = delta6

        # decode layer 2 (layer 5)
        delta5 = (self.W6.T @ delta6) * relu_derivative(self.a5)
        dW5 = np.outer(delta5, self.h4)
        db5 = delta5

        # decode layer 1 (layer 4)
        delta4 = (self.W5.T @ delta5) * relu_derivative(self.a4)
        dW4 = np.outer(delta4, self.z)
        db4 = delta4

        # latent layer
        delta3 = (self.W4.T @ delta4) * 1.0
        dW3 = np.outer(delta3, self.h2)
        db3 = delta3

        # layer 2
        delta2 = (self.W3.T @ delta3) * relu_derivative(self.a2)
        dW2 = np.outer(delta2, self.h1)
        db2 = delta2

        # layer 2
        delta1 = (self.W2.T @ delta2) * relu_derivative(self.a1)
        dW1 = np.outer(delta1, self.x)
        db1 = delta1

        # Adam updates
        for name, grad, attr in [
            ('W6', dW6, 'W6'), ('b6', db6, 'b6'),
            ('W5', dW5, 'W5'), ('b5', db5, 'b5'),
            ('W4', dW4, 'W4'), ('b4', db4, 'b4'),
            ('W3', dW3, 'W3'), ('b3', db3, 'b3'),
            ('W2', dW2, 'W2'), ('b2', db2, 'b2'),
            ('W1', dW1, 'W1'), ('b1', db1, 'b1'),
        ]: setattr(self, attr, getattr(self, attr) - self._adam_step(name, grad))

    def fit(self, dataset, epochs, batch_size=32):
        for epoch in range(epochs):
            epoch_loss = 0
            indices = np.random.permutation(len(dataset))
            for start in range(0, len(dataset), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_loss = 0
                for i in batch_idx:
                    reconstructed = self.forward_pass(dataset[i])
                    batch_loss += self.mse_loss(dataset[i], reconstructed)
                    self.backprop()
                epoch_loss += batch_loss
            print(f"Epoch {epoch + 1}: avg loss {epoch_loss / len(dataset):.5f}")

    def save(self, path):
        np.savez(
            path,
            W1=self.W1, W2=self.W2, W3=self.W3,
            W4=self.W4, W5=self.W5, W6=self.W6,
            b1=self.b1, b2=self.b2, b3=self.b3,
            b4=self.b4, b5=self.b5, b6=self.b6,
            # first moments
            m_W1=self.m['W1'], m_W2=self.m['W2'], m_W3=self.m['W3'],
            m_W4=self.m['W4'], m_W5=self.m['W5'], m_W6=self.m['W6'],
            m_b1=self.m['b1'], m_b2=self.m['b2'], m_b3=self.m['b3'],
            m_b4=self.m['b4'], m_b5=self.m['b5'], m_b6=self.m['b6'],
            # second moments
            v_W1=self.v['W1'], v_W2=self.v['W2'], v_W3=self.v['W3'],
            v_W4=self.v['W4'], v_W5=self.v['W5'], v_W6=self.v['W6'],
            v_b1=self.v['b1'], v_b2=self.v['b2'], v_b3=self.v['b3'],
            v_b4=self.v['b4'], v_b5=self.v['b5'], v_b6=self.v['b6'],
            # timestamp
            t=np.array(self.t),
        )

    def load(self, path):
        data = np.load(path)
        self.W1, self.W2, self.W3 = data['W1'], data['W2'], data['W3']
        self.W4, self.W5, self.W6 = data['W4'], data['W5'], data['W6']
        self.b1, self.b2, self.b3 = data['b1'], data['b2'], data['b3']
        self.b4, self.b5, self.b6 = data['b4'], data['b5'], data['b6']

        try:
            self.m = {
                'W1': data['m_W1'], 'W2': data['m_W2'], 'W3': data['m_W3'],
                'W4': data['m_W4'], 'W5': data['m_W5'], 'W6': data['m_W6'],
                'b1': data['m_b1'], 'b2': data['m_b2'], 'b3': data['m_b3'],
                'b4': data['m_b4'], 'b5': data['m_b5'], 'b6': data['m_b6'],
            }
            self.v = {
                'W1': data['v_W1'], 'W2': data['v_W2'], 'W3': data['v_W3'],
                'W4': data['v_W4'], 'W5': data['v_W5'], 'W6': data['v_W6'],
                'b1': data['v_b1'], 'b2': data['v_b2'], 'b3': data['v_b3'],
                'b4': data['v_b4'], 'b5': data['v_b5'], 'b6': data['v_b6'],
            }
            self.t = int(data['t'])
        except Exception as e:
            print(e)


def encode_dataset(encoder, dataset, labels, max_samples=None):
    if max_samples is not None:
        idx = np.random.choice(len(dataset), max_samples, replace=False)
        dataset, labels = dataset[idx], labels[idx]

    embeddings = []
    for image in dataset:
        z = encoder.encode(image)
        embeddings.append(z.copy())  # .copy() since encode() stores state in-place

    return np.array(embeddings), np.array(labels)


def plot_latent_pca(embeddings, labels, title="PCA projection (2D)"):
    pca = PCA(n_components=2)
    projected = pca.fit_transform(embeddings)       # shape: (N, 2)
    var_explained = pca.explained_variance_ratio_
    print(f"PCA explained variance: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f} "
          f"(total={sum(var_explained):.3f})")

    scatter(projected[:, 0], projected[:, 1], labels,
             xlabel=f"PC1 ({var_explained[0]*100:.1f}% var)",
             ylabel=f"PC2 ({var_explained[1]*100:.1f}% var)",
             title=title)


def scatter(x, y, labels, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = cm.get_cmap("tab10", 10)

    for digit in range(10):
        mask = labels == digit
        ax.scatter(x[mask], y[mask],
                   c=[colors(digit)],
                   label=str(digit),
                   s=6,
                   alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Digit", markerscale=3, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    ae = AutoEncoder()
    choice = input("Would you like to train an autoencoder? (y/n): ")
    if choice == "y":
        ae.load("mnist_adam_autoencoder.npz")
        ae.fit(X_train, 5)
        ae.save("mnist_adam_autoencoder.npz")

    else:
        ae.load("mnist_adam_autoencoder.npz")

        random1 = np.random.randint(0, len(X_test))
        random2 = np.random.randint(0, len(X_test))

        decoded = ae.decode(ae.encode(X_test[random1]))
        decoded2 = ae.decode(ae.encode(X_test[random2]))

        fig, axes = plt.subplots(2, 2, figsize=(5, 5))

        axes[0, 0].imshow(X_test[random1], cmap="gray")
        axes[0, 0].set_title(f"Original: {y_test[random1]}")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(decoded, cmap="gray")
        axes[0, 1].set_title(f"Reconstructed: {y_test[random1]}")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(X_test[random2], cmap="gray")
        axes[1, 0].set_title(f"Original: {y_test[random2]}")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(decoded2, cmap="gray")
        axes[1, 1].set_title(f"Reconstructed: {y_test[random2]}")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.show()

        embeddings, labels = encode_dataset(ae, X_test, y_test, max_samples=2000)
        plot_latent_pca(embeddings, labels)
