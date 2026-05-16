import numpy as np
from matplotlib import pyplot as plt
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
    def __init__(self, learning_rate = 0.001):
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

    def backprop(self):
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


        # Gradient descent
        self.W6 -= self.lr * dW6
        self.b6 -= self.lr * db6

        self.W5 -= self.lr * dW5
        self.b5 -= self.lr * db5

        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

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
            b4=self.b4, b5=self.b5, b6=self.b6
        )

    def load(self, path):
        data = np.load(path)

        self.W1 = data['W1']
        self.W2 = data['W2']
        self.W3 = data['W3']
        self.W4 = data['W4']
        self.W5 = data['W5']
        self.W6 = data['W6']

        self.b1 = data['b1']
        self.b2 = data['b2']
        self.b3 = data['b3']
        self.b4 = data['b4']
        self.b5 = data['b5']
        self.b6 = data['b6']


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    ae = AutoEncoder()
    choice = input("Would you like to train an autoencoder? (y/n): ")
    if choice == "y":
        ae.fit(X_train, 10)
        ae.save("mnist_autoencoder.npz")

    else:
        ae.load("mnist_autoencoder.npz")

        encoded = ae.encode(X_test[16])
        encoded2 = ae.encode(X_test[15])
        print(encoded)
        print(encoded2)
        decoded = ae.decode(encoded)
        decoded2 = ae.decode(encoded2)
        plt.subplot(1, 10, 1)
        plt.imshow(decoded2, cmap="gray")
        plt.title(f"Autoencoder output of the number {y_test[15]}")
        plt.axis("off")
        plt.show()
