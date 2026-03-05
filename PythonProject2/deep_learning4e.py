"""
deep_learning4e.py  –  Lightweight stub for Week 9 Workshop
============================================================
This replaces the full deep_learning4e.py from the AIMA
repository, which requires TensorFlow/Keras.

TensorFlow does NOT support Python 3.12 or 3.13, and is not
needed for this workshop. This stub provides every class and
function that learning4e.py and test_learning4e.py actually
import, implemented using only NumPy.

Place this file in the SAME folder as learning4e.py.
============================================================
"""

import numpy as np


# ─────────────────────────────────────────────────────────
# Activation Functions
# Used by learning4e.py: LogisticLinearLeaner, LinearLearner
# ─────────────────────────────────────────────────────────

class Sigmoid:
    """Logistic / sigmoid activation function.
    Maps any real number to the range (0, 1).
    Used by LogisticLinearLeaner in learning4e.py.
    """

    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, value):
        """Derivative of sigmoid: s(x) * (1 - s(x))"""
        return value * (1.0 - value)


class ReLU:
    """Rectified Linear Unit activation function."""

    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, value):
        return (value > 0).astype(float)


class LeakyReLU:
    """Leaky ReLU – allows small gradient when unit is inactive."""

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, value):
        return np.where(value > 0, 1.0, self.alpha)


class Tanh:
    """Hyperbolic tangent activation function."""

    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, value):
        return 1.0 - value ** 2


class Linear:
    """Linear (identity) activation – output equals input."""

    def __call__(self, x):
        return x

    def derivative(self, value):
        return 1.0


class Softmax:
    """Softmax activation – converts scores to probabilities."""

    def __call__(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def derivative(self, value):
        return value * (1.0 - value)


# ─────────────────────────────────────────────────────────
# Layer classes (stubs – not used in Week 9 workshop)
# ─────────────────────────────────────────────────────────

class Layer:
    """Base class for neural network layers."""

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class DenseLayer(Layer):
    """Fully connected (dense) layer."""

    def __init__(self, in_size, out_size, activation=None):
        super().__init__()
        self.W = np.random.randn(in_size, out_size) * 0.01
        self.b = np.zeros(out_size)
        self.activation = activation or Linear()

    def forward(self, x):
        self.input = x
        self.output = self.activation(np.dot(x, self.W) + self.b)
        return self.output

    def backward(self, grad):
        return np.dot(grad, self.W.T)


# ─────────────────────────────────────────────────────────
# Neural Network
# ─────────────────────────────────────────────────────────

class NeuralNetwork:
    """Simple feedforward neural network (stub)."""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x):
        return self.forward(np.array(x))


# ─────────────────────────────────────────────────────────
# PerceptronLearner
# Imported by test_learning4e.py
# ─────────────────────────────────────────────────────────

def PerceptronLearner(dataset, learning_rate=0.01, epochs=100):
    """
    Single-layer perceptron classifier.
    Trains weights using the perceptron learning rule.

    Returns a predictor object with a .predict(example) method.
    """
    idx_i  = dataset.inputs
    idx_t  = dataset.target
    examples = dataset.examples
    n_inputs = len(idx_i)

    # One output node per class
    classes  = dataset.values[idx_t]
    n_classes = len(classes)

    # Initialise weights randomly: shape (n_classes, n_inputs + 1)
    # +1 for bias weight
    rng = np.random.default_rng(42)
    W = rng.uniform(-0.5, 0.5, (n_classes, n_inputs + 1))

    sigmoid = Sigmoid()

    for _ in range(epochs):
        for example in examples:
            # Build input vector with bias term
            x = np.array([1.0] + [example[i] for i in idx_i])

            # Forward pass
            scores = sigmoid(W.dot(x))

            # Build target vector
            target_class = example[idx_t]
            t = np.zeros(n_classes)
            if target_class in classes:
                t[classes.index(target_class)] = 1.0

            # Weight update (delta rule)
            error = t - scores
            for j in range(n_classes):
                W[j] += learning_rate * error[j] * x

    class Perceptron:
        def __init__(self, weights, classes, inputs, sigmoid):
            self.W       = weights
            self.classes  = classes
            self.inputs   = inputs
            self.sigmoid  = sigmoid

        def predict(self, example):
            x = np.array([1.0] + [example[i] for i in self.inputs])
            scores = self.sigmoid(self.W.dot(x))
            return np.argmax(scores)

        def predict_score(self, X):
            """Return raw scores for MultiClassLearner compatibility."""
            X = np.atleast_2d(X)
            results = []
            for row in X:
                x = np.array([1.0] + list(row))
                results.append(self.sigmoid(self.W.dot(x)))
            return np.array(results)

    return Perceptron(W, classes, idx_i, sigmoid)


# ─────────────────────────────────────────────────────────
# BackPropagationLearner (stub)
# Referenced in the AIMA notebook but not used in Week 9
# ─────────────────────────────────────────────────────────

def BackPropagationLearner(dataset, net, learning_rate=0.01, epochs=100):
    """
    Stub for the back-propagation learning algorithm.
    Not used in the Week 9 workshop.
    Included so that any import of this name does not raise an error.
    """
    return net


# ─────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────

def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    return -np.mean(np.array(y_true) * np.log(y_pred))


# ─────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────

def random_weights(n_inputs, n_outputs, scale=0.01):
    """Return randomly initialised weight matrix."""
    return np.random.randn(n_inputs, n_outputs) * scale
