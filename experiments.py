import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from tqdm import tqdm
import os

np.random.seed(42)
num_average_time = 100  # Number of runs to average timings
N_values = [10, 30, 50, 70, 100]
M_values = [5, 10, 15, 20, 25]

# Ensure output folder exists
os.makedirs("time_complexity_plots", exist_ok=True)

# Data Generation
def get_data(type, N, M):
    np.random.seed(42)
    if type == 'real_input_real_output':
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
    elif type == 'real_input_discrete_output':
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(0, 2, N))
    elif type == 'discrete_input_real_output':
        X = pd.DataFrame(np.random.randint(0, 2, (N, M)))
        y = pd.Series(np.random.randn(N))
    elif type == 'discrete_input_discrete_output':
        X = pd.DataFrame(np.random.randint(0, 2, (N, M)))
        y = pd.Series(np.random.randint(0, 2, N))
    else:
        raise ValueError("Invalid type")
    return X, y

# Timing Function
def get_decision_tree_time(N, M, type):
    X, y = get_data(type, N, M)
    training_times, testing_times = [], []

    for _ in tqdm(range(num_average_time), desc=f"{type} N={N}, M={M}", leave=False):
        tree = DecisionTree(criterion='information_gain', max_depth=10)

        # Training time
        start = time.process_time()
        tree.fit(X, y)
        end = time.process_time()
        training_times.append(end - start)

        # Testing time
        start = time.process_time()
        tree.predict(X)
        end = time.process_time()
        testing_times.append(end - start)

    # Compute mean and std
    mean_train = np.mean(training_times)
    std_train = np.std(training_times)
    mean_test = np.mean(testing_times)
    std_test = np.std(testing_times)

    print(f"[{type}] N={N}, M={M} | Train: {mean_train:.5f} ± {std_train:.5f}, Test: {mean_test:.5f} ± {std_test:.5f}")
    return mean_train, mean_test, std_train, std_test

# Plotting Functions
def plot_twin_axis_graph(x, y1, y2, y1_std, y2_std, title, xlabel):
    # Training Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title(title + " Training")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Time (s)", color="tab:red")
    ax1.plot(x, y1, color="tab:red")
    ax1.errorbar(x, y1, yerr=y1_std, fmt='o', color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend(["Training Time ± 1σ"])
    fig.savefig(f"time_complexity_plots/{title} Training.png")

    # Testing Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title(title + " Testing")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Time (s)", color="tab:blue")
    ax1.plot(x, y2, color="tab:blue")
    ax1.errorbar(x, y2, yerr=y2_std, fmt='o', color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(["Testing Time ± 1σ"])
    fig.savefig(f"time_complexity_plots/{title} Testing.png")

# Graph Function
def plot_graph(N_vals, M_vals, fn, type):
    print(f"\n=== Plotting graphs for {type} ===")

    # Vary N (fix M=5)
    training_times, testing_times, training_time_stds, testing_time_stds = [], [], [], []
    for N in N_vals:
        t_train, t_test, std_train, std_test = fn(N, 5, type)
        training_times.append(t_train)
        testing_times.append(t_test)
        training_time_stds.append(std_train)
        testing_time_stds.append(std_test)
    plot_twin_axis_graph(N_vals, training_times, testing_times, training_time_stds, testing_time_stds, type + " wrt N", "N")

    # Vary M (fix N=20)
    training_times, testing_times, training_time_stds, testing_time_stds = [], [], [], []
    for M in M_vals:
        t_train, t_test, std_train, std_test = fn(20, M, type)
        training_times.append(t_train)
        testing_times.append(t_test)
        training_time_stds.append(std_train)
        testing_time_stds.append(std_test)
    plot_twin_axis_graph(M_vals, training_times, testing_times, training_time_stds, testing_time_stds, type + " wrt M", "M")

# Run Experiments
if __name__ == "__main__":
    plot_graph(N_values, M_values, get_decision_tree_time, "real_input_real_output")
    plot_graph(N_values, M_values, get_decision_tree_time, "real_input_discrete_output")
    plot_graph(N_values, M_values, get_decision_tree_time, "discrete_input_real_output")
    plot_graph(N_values, M_values, get_decision_tree_time, "discrete_input_discrete_output")
