import numpy as np


def get_numeric_grad(f, x, eps):
    x_length = x.shape[0]
    f_err = np.array([0] * x_length)
    for i in range(x.shape[0]):
        e = np.eye(1, x_length, i)
        f_err[i] = abs(f(x + eps * e) - f(x)) / eps
    return f_err

def compute_balanced_accuracy(true_y, pred_y):
	"""
	Get balaced accuracy value

	Parameters
	----------
	true_y : numpy.ndarray
		True target.
	pred_y : numpy.ndarray
		Predictions.
	Returns
	-------
	: float
	"""
	possible_y = set(true_y)
	value = 0
	for current_y in possible_y:
		mask = true_y == current_y
		value += (pred_y[mask] == current_y).sum() / mask.sum()
	return value / len(possible_y)
