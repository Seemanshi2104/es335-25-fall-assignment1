import os
import numpy as np
import pandas as pd

# Path to the HAR dataset root (the folder containing features.txt, activity_labels.txt, train/, test/)
HAR_ROOT = r"C:\Users\ginis\Downloads\ML\UCI HAR Dataset including combined folder\UCI HAR Dataset\UCI HAR Dataset"

# Load feature names (561 columns)
with open(os.path.join(HAR_ROOT, 'features.txt'), 'r') as f:
    lines = f.readlines()

features = []
exists = {}
for line in lines:
    # Take the feature name, replace '-' with '_'
    new_line = line.strip().split(" ", 1)[1].replace('-', '_')
    if new_line in exists:
        features.append(new_line + "_" + str(exists[new_line]))
        exists[new_line] += 1
    else:
        features.append(new_line)
        exists[new_line] = 1

# ---- Load train and test data ----
X_train = np.loadtxt(os.path.join(HAR_ROOT, "train", "X_train.txt"))
y_train = np.loadtxt(os.path.join(HAR_ROOT, "train", "y_train.txt"), dtype=int) - 1  # 0-based labels
X_test  = np.loadtxt(os.path.join(HAR_ROOT, "test",  "X_test.txt"))
y_test  = np.loadtxt(os.path.join(HAR_ROOT, "test",  "y_test.txt"), dtype=int) - 1

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# make DataFrames
X_train_df = pd.DataFrame(X_train, columns=features)
X_test_df  = pd.DataFrame(X_test,  columns=features)

if __name__ == "__main__":
    print("\nData loaded successfully!")
    print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Unique activities in y_train: {np.unique(y_train)}")

def load_har_data():
    return X_train, y_train, X_test, y_test
