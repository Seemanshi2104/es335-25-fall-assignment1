# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from tree.base import DecisionTree
# from metrics import *

# np.random.seed(42)

# # Reading the data
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# data = pd.read_csv(url, delim_whitespace=True, header=None,
#                  names=["mpg", "cylinders", "displacement", "horsepower", "weight",
#                         "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tree.base import DecisionTree
from metrics import rmse, mae

np.random.seed(42)

# --------------------------
# Load dataset
# --------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
data = pd.read_csv(
    url,
    sep=r"\s+",
    header=None,
    names=["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model year", "origin", "car name"]
)

# --------------------------
# Clean dataset
# --------------------------
# Drop car name
data = data.drop(columns=["car name"])

# Replace '?' in horsepower with NaN
data["horsepower"] = data["horsepower"].replace("?", np.nan)

data["horsepower"] = data["horsepower"].astype(float)

# Drop rows with missing values
data = data.dropna()

# --------------------------
# Split features/target
# --------------------------
X = data.drop(columns=["mpg"])
y = data["mpg"]

# Train-test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------
# Part (a) Our Decision Tree
# --------------------------
tree = DecisionTree(criterion="mse",max_depth=10)
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)

print("\n=== Part (a): Custom Decision Tree (Regression) ===")
print("RMSE:", rmse(y_hat, y_test))
print("MAE :", mae(y_hat, y_test))

# --------------------------
# Part (b) Compare with sklearn
# --------------------------
sk_tree = DecisionTreeRegressor(criterion="squared_error", max_depth=10, random_state=42)
sk_tree.fit(X_train, y_train)
y_hat_sk = sk_tree.predict(X_test)

print("\n=== Part (b): Sklearn Decision Tree Regressor ===")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_hat_sk)))
print("MAE :", mean_absolute_error(y_test, y_hat_sk))

# --------------------------
# (Optional) Visualize Predictions
# --------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_hat, label="Custom Tree", alpha=0.7)
plt.scatter(y_test, y_hat_sk, label="Sklearn Tree", alpha=0.7, marker="x")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
plt.xlabel("True MPG")
plt.ylabel("Predicted MPG")
plt.legend()
plt.title("Custom vs Sklearn Decision Tree Predictions")
plt.show()

depths = range(1, 11)   # depths from 1 to 10
rmse_custom, rmse_sklearn = [], []

for d in depths:
    # Custom Tree
    tree = DecisionTree(criterion="mse", max_depth=d)
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    rmse_custom.append(rmse(y_hat, y_test))
    
    # Sklearn Tree
    sk_tree = DecisionTreeRegressor(criterion="squared_error", max_depth=d, random_state=42)
    sk_tree.fit(X_train, y_train)
    y_hat_sk = sk_tree.predict(X_test)
    rmse_sklearn.append(np.sqrt(mean_squared_error(y_test, y_hat_sk)))

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(depths, rmse_custom, label="Custom Tree RMSE", marker="o")
plt.plot(depths, rmse_sklearn, label="Sklearn Tree RMSE", marker="x")
plt.xlabel("Tree Depth")
plt.ylabel("RMSE")
plt.title("RMSE vs Tree Depth")
plt.legend()
plt.grid(True)
plt.show()