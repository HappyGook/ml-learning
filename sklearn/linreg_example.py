import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
x = data[["GDP per capita (USD)"]].values
y = data["Life satisfaction"].values

data.plot(kind="scatter", grid = True,
          x = "GDP per capita (USD)", y = "Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])

# simple a*x + b - type linear regression
linreg_model = LinearRegression()
linreg_model.fit(x, y)
lr_line = linreg_model.coef_ * x + linreg_model.intercept_

# K-neighbors regressor
kn_model = KNeighborsRegressor(n_neighbors=3)
kn_model.fit(x, y)

# linear regression with a*x^2 + b*x + c type
deg2_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
deg2_model.fit(x, y)
lin = deg2_model.named_steps['linearregression']
a = lin.coef_[1]   # coefficient for x^2
b = lin.coef_[0]   # coefficient for x
c = lin.intercept_

plt.plot(x, lr_line, color="red")
plt.plot(x, deg2_model, color="blue")
plt.show()


new_sample = [[16_000.5]] # Somewhat Colombia, lifesat approx 6.3

print("Current models generalise poorly for new data\n "
      "The actual colombian Life satisfaction lies close to 6.3\n"
      f"Linear regression model predicts Life satisfaction coefficient of {linreg_model.predict(new_sample)}\n"
      f"K-neighbors regressor (n=3) predicts Life satisfaction coefficient of {kn_model.predict(new_sample)}\n")