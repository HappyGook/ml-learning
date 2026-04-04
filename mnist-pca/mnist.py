from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pcas

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

digit = 5
mask = (y_train == digit)
X_digit = X_train[mask]
y_digit = y_train[mask]

# reshape for pca
X_digit = X_digit.reshape(X_digit.shape[0], -1)

for i in range(1,10):
    z,b,m,eigvals = pcas.pca(X_digit,i)

    x = pcas.generate_from_pca(b,m,eigvals,i)

    plt.subplot(1, 9, i)
    plt.imshow(x.reshape(28,28), cmap="gray")
    plt.title(f"k={i}")
    plt.axis("off")

plt.tight_layout()
plt.show()