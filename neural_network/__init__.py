"""
The :mod:`sklearn.neural_network` module includes models based on neural
networks.
"""

# License: BSD 3 clause

from .rbm import BernoulliRBM

from .multilayer_perceptron import MLPClassifier
from .multilayer_perceptron import MLPRegressor

__all__ = ["BernoulliRBM",
           "MLPClassifier",
           "MLPRegressor"]

print()
print("---------------------------------------------------")
print("This is scikit-learn-plus!!!")
print("---------------------------------------------------")
print()
