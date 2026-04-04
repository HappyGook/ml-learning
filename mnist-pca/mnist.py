from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pcas

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

# reshape for pca
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print(X_train.shape, X_test.shape)

for i in range(1,10):
    z,b,m,eigvals = pcas.pca(X_train,i)

    x = pcas.generate_from_pca(b,m,eigvals,i)

    plt.subplot(1, 9, i)
    plt.imshow(x.reshape(28,28), cmap="gray")
    plt.title(f"k={i}")
    plt.axis("off")

plt.tight_layout()
plt.show()