# Samuel Mouradian
# Project 3 - CS 422.622.1001
# Professor - Dr. Emily Hand
# Assingment Due Date - 11.18.2025



# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt



# Binary Classification
# FUNCTION 1: svm_train_brute(training_data)
def svm_train_brute(training_data):
    X = training_data[:, :2]
    Y = training_data[:, 2]
    pos = X[Y == 1]
    neg = X[Y == -1]

    margin = -1
    w = None
    b = None
    S = None
    for i in pos:
        for j in neg:
            temp_w = i - j
            midpoint = (i + j) / 2
            temp_b = -np.dot(temp_w, midpoint)

            valid = True
            for row in training_data:
                xi = row[:2]
                yi = row[2]
                if (yi * (np.dot(temp_w, xi) + temp_b) <= 0):
                    boolVal = False
                    break
                else:
                    continue
            temp_margin = compute_margin(training_data, temp_w, temp_b)
            if (temp_margin > margin):
                margin = temp_margin
                w = temp_w
                b = temp_b
                S = np.array([i, j])
            else:
                break
    return w, b, S

# FUNCTION 2: distance_point_to_hyperplane(pt, w, b)
def distance_point_to_hyperplane(pt, w, b):
    num = abs(np.dot(w, pt) + b)
    denom = np.linalg.norm(w)
    quotient = (num / denom)
    return quotient

# FUNCTION 3: compute_margin(data, w, b)
def compute_margin(data, w, b):
    distance = []
    for row in data:
        pt = row[:2]
        distance.append(distance_point_to_hyperplane(pt, w, b))
    return min(distance)

# FUNCTION 4: svm_test_brute(w, b, x)
def svm_test_brute(w, b, x):
    value = np.dot(w, x) + b
    if value > 0:
        return 1
    else:
        return -1

# FUNCTION 5: plot_and_boundary(data, w, b)
def plot_data_and_boundary(data, w, b):
    for row in data:
        x1, x2, label = row
        if label == 1:
            color = 'r'
        else:
            color = 'b'
        plt.scatter(x1, x2, color=color)
    
    x_vals = np.linspace(min(data[:, 0]) - 1, max(data[:, 0]) + 1, 200)
    if (abs(w[1]) > 1e-6):
      y_vals = (-w[0] * x_vals - b) / w[1]
      plt.plot(x_vals, y_vals, 'k--')
    else:
      x_boundary = -b / w[0]
      plt.axvline(x_boundary, color = 'k', linestyle = '--')
    
    plt.title("Data + Decision Boundary")
    plt.show()



# Multi-Class Classification:
# FUNCTION 1: svm_train_multiclass(training_data)
def svm_train_multiclass(training_data):
    X = training_data[:, :2]
    Y = training_data[:, 2]
    classes = np.unique(Y)
    C = len(classes)
    W = []
    B = []

    for i in classes:
        Y_binary = np.array([1 if y == i else -1 for y in Y])
        data_binary = np.column_stack([X, Y_binary])
        w, b, _ = svm_train_brute(data_binary)
        W.append(w)
        B.append(b)

    return np.array(W), np.array(B)

# FUNCTION 2: svm_test_multiclass(W, B, x)
def svm_test_multiclass(W, B, x):
    scores = [np.dot(W[i], x) + B[i] for i in range(len(W))]
    max_score = max(scores)
    if (max_score <= 0):
        return -1
    
    best_class = np.argmax(scores)
    return (best_class + 1)

# FUNCITON 3: plot_data_and_boundaries(data, W, B)
def plot_data_and_boundaries(data, W, B):
  colors = ['r', 'b', 'g', 'm', 'c', 'y'] # Changed 'n' to 'm'
  for row in data:
    x1, x2, y = row
    plt.scatter(x1, x2, color = colors[int(y) - 1])

    x_vals = np.linspace(min(data[:, 0]) - 1, max(data[:, 0]) + 1, 300)
    for i in range(len(W)):
      w = W[i]
      b = B[i]

      if abs(w[1]) > 1e-6:
        y_vals = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals, '--', label = f"class {i + 1} boundary")
      else:
          x_line = -b / w[0]
          plt.axvline(x_line, linestyle = '--', label = f"Class {i + 1} boundary")

  plt.title("Multi-Class SVM: One-vs-All Boundaries")
  plt.show()




# TESTING:
from helpers import generate_training_data_binary, plot_training_data_binary, generate_training_data_multi, plot_training_data_multi

# BINARY
data = generate_training_data_binary(1)
plot_training_data_binary(data)
[w, b, S] = svm_train_brute(data)
dist = distance_point_to_hyperplane((1, 1), w ,b)
margin = compute_margin(data, w, b)
y = svm_test_brute(w, b, (1, 1))
plot_data_and_boundary(data, w, b)

# MULTI
[data, Y] = generate_training_data_multi(1)
plot_training_data_multi(data)
[W, B] = svm_train_multiclass(data)
y = svm_test_multiclass(W, B, (1, 1))
plot_data_and_boundaries(data, W, B)