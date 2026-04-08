from tensorflow.keras.datasets import mnist
import pcas
import model
import twoDplot
import threeDplot

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

digit = 7
mask = (y_train == digit)
X_digit = X_train[mask]
y_digit = y_train[mask]

# reshape for pca
X_digit = X_digit.reshape(X_digit.shape[0], -1)

dimensions = 3

z,b,m,eigvals = pcas.pca(X_digit,dimensions)

mixture_components = 4
model = model.Model(mixture_components, z)

# run EM for multiple iterations
n_iters = 20
for _ in range(n_iters):
    model.e_step()
    model.m_step()

# visualize in space
if dimensions ==2:
    twoDplot.plot_gmm_2d(z,model,digit)
elif dimensions ==3:
    threeDplot.plot_gmm_3d(z,model,digit)
else:
    print("Too many dimensions :(")