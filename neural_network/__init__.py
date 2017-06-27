"""
The :mod:`sklearn.neural_network` module includes models based on neural
networks.
"""

# License: BSD 3 clause

from .multilayer_perceptron import MLPRegressor
from .multilayer_perceptron_cuda import MLPRegressor as MLPRegressor_cuda

__all__ = ["MLPRegressor", "MLPRegressor_cuda"]

print()
print("---------------------------------------------------")
print("This is scikit-learn-plus!!!")
print("---------------------------------------------------")
print()
