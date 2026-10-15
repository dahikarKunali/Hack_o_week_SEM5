"""
Neural Network from Scratch (NumPy only) + Model Deployment Basics
---------------------------------------------------------------------
No PyTorch, no TensorFlow, no sklearn.neural_network. Every piece -
perceptrons, activations, forward pass, backward pass, two optimizers -
is written by hand so the math underneath a neural net is actually visible.

Dataset: "two moons" - two interleaving crescent shapes that a straight
line can't separate. A plain logistic regression fails on this; a small
neural net with one hidden layer solves it easily. That contrast is the
whole point of using a nonlinear model in the first place.

Second half of the script covers deployment basics: saving the trained
model with both pickle and joblib, then reloading it and checking the
predictions still match.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import pickle
import joblib
import os


# ---------------------------------------------------------------------
# Part 1: Data
# ---------------------------------------------------------------------

def get_data(n_samples=500, noise=0.2, seed=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    y = y.reshape(-1, 1)
    return X, y


# ---------------------------------------------------------------------
# Part 2: Activation functions and their derivatives
#
# A perceptron on its own is just a weighted sum: z = w.x + b. Stacking
# plain perceptrons still collapses to one linear function, no matter
# how many layers you add. The activation function is what breaks that -
# it's the only thing that lets a network learn curves instead of lines.
# ---------------------------------------------------------------------

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow on large negative z
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


# ---------------------------------------------------------------------
# Part 3: The network itself
#
# Architecture: input(2) -> hidden(8, ReLU) -> output(1, sigmoid)
# Two inputs because the moons dataset is 2D. One sigmoid output because
# this is binary classification (which moon does the point belong to).
# ---------------------------------------------------------------------

class NeuralNetwork:
    def __init__(self, input_size=2, hidden_size=8, output_size=1, seed=1):
        rng = np.random.default_rng(seed)
        # He initialization - scaled for ReLU layers, keeps early gradients
        # from exploding or vanishing before training even starts.
        self.W1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

        # Adam needs a running average of gradients and their squares,
        # one pair per parameter. Zero-initialized, updated every step.
        self.m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.t = 0  # Adam time step, increments every update

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    # --- Forward propagation ---
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    # --- Loss: binary cross-entropy ---
    @staticmethod
    def compute_loss(y_true, y_pred):
        eps = 1e-8  # keeps log() from hitting exactly 0
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # --- Backward propagation ---
    # Every gradient here is the chain rule applied one layer at a time:
    # dL/dW2 needs dL/da2, da2/dz2, dz2/dW2, multiplied together.
    # Working backward from the loss is exactly where "backprop" gets its name.
    def backward(self, X, y_true):
        n = X.shape[0]

        dz2 = self.a2 - y_true  # derivative of BCE+sigmoid combined, simplifies nicely
        dW2 = self.a1.T @ dz2 / n
        db2 = np.sum(dz2, axis=0, keepdims=True) / n

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = X.T @ dz1 / n
        db1 = np.sum(dz1, axis=0, keepdims=True) / n

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # --- Optimizer: plain SGD ---
    def sgd_step(self, grads, lr):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    # --- Optimizer: Adam ---
    # Adam keeps a moving average of the gradient (m) and of the squared
    # gradient (v), then scales each parameter's step by its own history.
    # That's why it usually converges faster than plain SGD on messy loss surfaces.
    def adam_step(self, grads, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        params = self._params()
        for key in params:
            g = grads[key]
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * g
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (g ** 2)

            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)

            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            params[key] -= update

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


# ---------------------------------------------------------------------
# Part 4: Training loop
# ---------------------------------------------------------------------

def train(net, X, y, epochs=1000, lr=0.05, optimizer="adam", verbose_every=100):
    loss_history = []
    for epoch in range(epochs):
        y_pred = net.forward(X)
        loss = net.compute_loss(y, y_pred)
        loss_history.append(loss)

        grads = net.backward(X, y)
        if optimizer == "sgd":
            net.sgd_step(grads, lr)
        elif optimizer == "adam":
            net.adam_step(grads, lr=lr)
        else:
            raise ValueError("optimizer must be 'sgd' or 'adam'")

        if epoch % verbose_every == 0:
            acc = np.mean(net.predict(X) == y)
            print(f"epoch {epoch:4d} | loss {loss:.4f} | accuracy {acc:.3f}")

    return loss_history


# ---------------------------------------------------------------------
# Part 5: Visualization
# ---------------------------------------------------------------------

def plot_decision_boundary(net, X, y, title, ax):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = net.forward(grid).reshape(xx.shape)

    ax.contourf(xx, yy, preds, levels=50, cmap="RdBu", alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="RdBu", edgecolors="k", s=25)
    ax.set_title(title)


# ---------------------------------------------------------------------
# Part 6: Model deployment - saving and reloading
# ---------------------------------------------------------------------

def save_model(net, folder="saved_models"):
    os.makedirs(folder, exist_ok=True)

    pickle_path = os.path.join(folder, "model.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(net, f)

    joblib_path = os.path.join(folder, "model.joblib")
    joblib.dump(net, joblib_path)

    print(f"Saved to {pickle_path} and {joblib_path}")
    return pickle_path, joblib_path


def load_and_verify(pickle_path, joblib_path, X_test):
    with open(pickle_path, "rb") as f:
        net_from_pickle = pickle.load(f)

    net_from_joblib = joblib.load(joblib_path)

    pred_pickle = net_from_pickle.predict(X_test)
    pred_joblib = net_from_joblib.predict(X_test)

    match = np.array_equal(pred_pickle, pred_joblib)
    print(f"Pickle and joblib predictions match: {match}")
    return net_from_pickle, net_from_joblib


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    X, y = get_data()

    net_sgd = NeuralNetwork(seed=1)
    net_adam = NeuralNetwork(seed=1)  # same init, so the comparison is fair

    print("Training with plain SGD")
    loss_sgd = train(net_sgd, X, y, epochs=1000, lr=0.5, optimizer="sgd")

    print("\nTraining with Adam")
    loss_adam = train(net_adam, X, y, epochs=1000, lr=0.05, optimizer="adam")

    # --- Plot 1: loss curves, SGD vs Adam ---
    plt.figure(figsize=(7, 5))
    plt.plot(loss_sgd, label="SGD")
    plt.plot(loss_adam, label="Adam")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("SGD vs Adam - same network, same data, same starting weights")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_comparison.png", dpi=120)
    plt.show()

    # --- Plot 2: decision boundaries side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_decision_boundary(net_sgd, X, y, "Decision boundary - SGD", axes[0])
    plot_decision_boundary(net_adam, X, y, "Decision boundary - Adam", axes[1])
    plt.tight_layout()
    plt.savefig("decision_boundaries.png", dpi=120)
    plt.show()

    # --- Deployment: save the Adam-trained model, reload it, check it still works ---
    print("\nSaving and reloading the trained model")
    pkl_path, jl_path = save_model(net_adam)
    load_and_verify(pkl_path, jl_path, X)