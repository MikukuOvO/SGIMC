"""Sparse-dense operations for IMC."""
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

from .base import QuadraticApproximation


class QAObjectiveL2Loss(QuadraticApproximation):
    """Quadratic Approximation for the L2 loss."""

    def __init__(self, X, W, Y, H, R,
                 n_threads=1, approx_type="quadratic"):
        """Build the Quadratic approximation."""
        super(QAObjectiveL2Loss, self).__init__(
            X, W, Y, H, R,
            n_threads=n_threads,
            approx_type=approx_type)

    @staticmethod
    def v_func(predict, target):
        """Get the loss values."""
        return 0.5 * (predict - target)**2

    @staticmethod
    def g_func(predict, target):
        """Get the gradient statistics."""
        return predict - target

    @staticmethod
    def h_func(predict, target):
        """Get the Hessian statistics."""
        return np.ones(len(target), dtype="double")

    @staticmethod
    def score(predict, target):
        """Compute the RMSE."""
        return mean_squared_error(target, predict)


class QAObjectiveHuberLoss(QAObjectiveL2Loss):
    """Quadratic Approximation for the Huber loss."""

    def __init__(self, X, W, Y, H, R, epsilon=1e-2,
                 n_threads=1, approx_type="quadratic"):
        """Build the Quadratic approximation."""
        super(QAObjectiveHuberLoss, self).__init__(
            X, W, Y, H, R,
            n_threads=n_threads,
            approx_type=approx_type)

        self.epsilon = epsilon

    def v_func(self, predict, target):
        """Get the loss values."""
        eps = self.epsilon

        resid = abs(predict - target)
        return np.where(resid > eps,
                        eps * (resid - eps / 2),
                        0.5 * resid**2)

    def g_func(self, predict, target):
        """Get the gradient statistics."""
        eps = self.epsilon

        resid = predict - target
        return np.where(abs(resid) > eps,
                        eps * np.sign(resid),
                        resid)

    def h_func(self, predict, target):
        """Get the Hessian statistics."""
        eps = self.epsilon

        return (abs(predict - target) <= eps) * 1.


def sigmoid(x):
    """Compute the sigmoid transformation of the input."""
    out = 1 / (1 + np.exp(- abs(x)))
    return np.where(x > 0, out, 1 - out)


class QAObjectiveLogLoss(QuadraticApproximation):
    """Quadratic Approximation for the Logistic loss."""

    def __init__(self, X, W, Y, H, R,
                 n_threads=1, approx_type="quadratic"):
        """Build the Quadratic approximation."""
        super(QAObjectiveLogLoss, self).__init__(
            X, W, Y, H, R,
            n_threads=n_threads,
            approx_type=approx_type)

    @staticmethod
    def v_func(logit, target):
        """Get the loss values."""
        out = np.log1p(np.exp(- abs(logit)))
        out -= np.minimum(target * logit, 0)
        return out

    @staticmethod
    def g_func(logit, target):
        """Get the gradient statistics."""
        return sigmoid(logit * target) * target - target

    @staticmethod
    def h_func(logit, target):
        """Get the Hessian statistics."""
        prob = sigmoid(logit)
        return prob * (1 - prob)

    @staticmethod
    def score(logit, target):
        """Compute the misclassification rate."""
        predict = (2. * (logit > 0)) - 1
        return 1 - accuracy_score(target, predict)
