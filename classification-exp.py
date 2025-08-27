
# Write the code for Q2 a) and b) below. Show your results.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold

from tree.base import DecisionTree
from metrics import accuracy, precision, recall

np.random.seed(42)
# Code given in the question
X, y = make_classification(
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=2,
    class_sep=0.5
)
# For plotting

plt.figure(figsize=(6,5))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Generated Classification Dataset", fontsize=14)
plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.show()


# Shuffle once
shuffle_idx = np.random.permutation(y.size)
X, y = X[shuffle_idx], y[shuffle_idx]

# Convert to DataFrame + Series (for compatibility with your DT)
X = pd.DataFrame(X, columns=["x1", "x2"])
y = pd.Series(y, dtype="category")

# Train/Test Split (70/30, sequential after shuffle)

total_size = len(X)
train_size = int(total_size * 0.7)

X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

# Train the Decision Tree

tree = DecisionTree(criterion="information_gain")
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)

# Report metrics

print("\n=== Part (a): Results on Test Data (70/30 split) ===")
print("Accuracy:", accuracy(y_hat, y_test))
for cls in y_test.unique():
    print(f"Class {cls}: Precision={precision(y_hat, y_test, cls):.3f}, Recall={recall(y_hat, y_test, cls):.3f}")


from collections import Counter

# Part (b) Nested CV for depth selection

kf = KFold(n_splits=5, shuffle=True, random_state=42)
depth_candidates = [1,2,3,4,5,6,7,8]
outer_scores = []
chosen_depths = []

print("\n=== Nested Cross-Validation Results ===")
print("{:<8} {:<12} {:<15} {:<10}".format("Fold", "Best Depth", "Outer Accuracy", "Val Accuracies"))

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Inner loop: model selection
    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    avg_scores = {d: [] for d in depth_candidates}

    for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
        X_inner_train, X_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
        y_inner_train, y_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]

        for d in depth_candidates:
            tree = DecisionTree(criterion="information_gain", max_depth=d)
            tree.fit(X_inner_train, y_inner_train)
            y_val_hat = tree.predict(X_val)
            avg_scores[d].append(accuracy(y_val_hat, y_val))

    # Average validation accuracy across folds
    mean_scores = {d: np.mean(scores) for d, scores in avg_scores.items()}

    # Pick best depth
    best_depth = max(mean_scores, key=mean_scores.get)
    chosen_depths.append(best_depth)

    # Evaluate only best depth on the outer test set
    best_tree = DecisionTree(criterion="information_gain", max_depth=best_depth)
    best_tree.fit(X_train, y_train)
    y_test_hat = best_tree.predict(X_test)
    best_score = accuracy(y_test_hat, y_test)
    outer_scores.append(best_score)

    # Print fold summary
    val_acc_str = ", ".join([f"{d}:{mean_scores[d]:.2f}" for d in depth_candidates])
    print("{:<8} {:<12} {:<15.3f} {}".format(fold, best_depth, best_score, val_acc_str))

# Final results
print("\n=== Overall Summary ===")
print("Average outer test accuracy:", np.mean(outer_scores))
print("Depths chosen in each fold:", chosen_depths)
print("Most frequently chosen depth:", Counter(chosen_depths).most_common(1)[0][0])
