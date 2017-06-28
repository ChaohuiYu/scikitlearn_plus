"""CUDA version utilities for the neural network modules
"""

# License: BSD 3 clause

import numpy as np

from pycuda import gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import skcuda.misc as cumisc
cumisc.init()

blockSize = 192

mod = None
with open("../cuda/_base.cu", "r") as fin:
    mod = SourceModule(fin.read())

cuClipf = mod.get_function("clipf")
cuClip = mod.get_function("clip")

cuSigmoidf = mod.get_function("sigmoidf")
cuSigmoid = mod.get_function("sigmoid")
cuTanhf = mod.get_function("myTanhf")
cuTanh = mod.get_function("myTanh")

cuInplaceReluDerivativef = mod.get_function("inplaceReluDerivativef")
cuInplaceReluDerivative = mod.get_function("inplaceReluDerivative")

cuSquaredErrorf = mod.get_function("squaredErrorf")
cuSquaredError = mod.get_function("squaredError")
cuLogLossf = mod.get_function("logLossf")
cuLogLoss = mod.get_function("logLoss")
cuBinaryLogLossf = mod.get_function("binaryLogLossf")
cuBinaryLogLoss = mod.get_function("binaryLogLoss")


def identity(X):
    """Simply return the input array.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Data, where n_samples is the number of samples
        and n_features is the number of features.

    Returns
    -------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Same as the input data.
    """
    return X


def logistic(X):
    """Compute the logistic function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    if X.dtype == np.float64:
        cuSigmoid(
            X.gpudata,
            np.int32(X.size),
            block=(blockSize, 1, 1),
            grid=(int((X.size - 1) / blockSize + 1), 1, 1))
    else:
        cuSigmoidf(
            X.gpudata,
            np.int32(X.size),
            block=(blockSize, 1, 1),
            grid=(int((X.size - 1) / blockSize + 1), 1, 1))
    return X


def tanh(X):
    """Compute the hyperbolic tan function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    if X.dtype == np.float64:
        cuTanh(
            X.gpudata,
            np.int32(X.size),
            block=(blockSize, 1, 1),
            grid=(int((X.size - 1) / blockSize + 1), 1, 1))
    else:
        cuTanhf(
            X.gpudata,
            np.int32(X.size),
            block=(blockSize, 1, 1),
            grid=(int((X.size - 1) / blockSize + 1), 1, 1))
    return X


def relu(X):
    """Compute the rectified linear unit function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    if X.dtype == np.float64:
        cuClip(
            X.gpudata,
            np.float64(0.),
            np.finfo(X.dtype).max,
            np.int32(X.size),
            block=(blockSize, 1, 1),
            grid=(int((X.size - 1) / blockSize + 1), 1, 1))
    else:
        cuClipf(
            X.gpudata,
            np.float32(0.),
            np.float32(np.finfo(X.dtype).max),
            np.int32(X.size),
            block=(blockSize, 1, 1),
            grid=(int((X.size - 1) / blockSize + 1), 1, 1))
    return X

ACTIVATIONS = {'identity': identity, 'tanh': tanh, 'logistic': logistic,
               'relu': relu}


def inplace_identity_derivative(Z, delta):
    """Apply the derivative of the identity function: do nothing.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the identity activation function during
        the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    # Nothing to do


def inplace_logistic_derivative(Z, delta):
    """Apply the derivative of the logistic sigmoid function.

    It exploits the fact that the derivative is a simple function of the output
    value from logistic function.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the logistic activation function during
        the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= Z
    delta *= (1 - Z)


def inplace_tanh_derivative(Z, delta):
    """Apply the derivative of the hyperbolic tanh function.

    It exploits the fact that the derivative is a simple function of the output
    value from hyperbolic tangent.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the hyperbolic tangent activation
        function during the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= (1 - Z ** 2)


def inplace_relu_derivative(Z, delta):
    """Apply the derivative of the relu function.

    It exploits the fact that the derivative is a simple function of the output
    value from rectified linear units activation function.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the rectified linear units activation
        function during the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    if delta.dtype == np.float64:
        cuInplaceReluDerivative(
            Z.gpudata,
            delta.gpudata,
            np.int32(delta.size),
            block=(blockSize, 1, 1),
            grid=(int((delta.size - 1) / blockSize + 1), 1, 1))
    else:
        cuInplaceReluDerivativef(
            Z.gpudata,
            delta.gpudata,
            np.int32(delta.size),
            block=(blockSize, 1, 1),
            grid=(int((delta.size - 1) / blockSize + 1), 1, 1))


DERIVATIVES = {'identity': inplace_identity_derivative,
               'tanh': inplace_tanh_derivative,
               'logistic': inplace_logistic_derivative,
               'relu': inplace_relu_derivative}


def squared_loss(y_true, y_pred):
    """Compute the squared loss for regression.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) values.

    y_pred : array-like or label indicator matrix
        Predicted values, as returned by a regression estimator.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    tmp_gpu = gpuarray.GPUArray(y_true.shape, y_true.dtype)
    if y_true.dtype == np.float64:
        cuSquaredError(
            y_true.gpudata,
            y_pred.gpudata,
            tmp_gpu.gpudata,
            np.int32(y_true.size),
            block=(blockSize, 1, 1),
            grid=(int((y_true.size - 1) / blockSize + 1), 1, 1))
    else:
        cuSquaredErrorf(
            y_true.gpudata,
            y_pred.gpudata,
            tmp_gpu.gpudata,
            np.int32(y_true.size),
            block=(blockSize, 1, 1),
            grid=(int((y_true.size - 1) / blockSize + 1), 1, 1))
    mean = float(cumisc.mean(tmp_gpu).get())
    return (mean / 2)


def log_loss(y_true, y_prob):
    """Compute Logistic loss for classification.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    if y_prob.dtype == np.float64:
        cuClip(
            y_prob.gpudata,
            np.float64(1e-10),
            np.float64(1 - 1e-10),
            np.int32(y_prob.size),
            block=(blockSize, 1, 1),
            grid=(int((y_prob.size - 1) / blockSize + 1), 1, 1))
    else:
        cuClipf(
            y_prob.gpudata,
            np.float32(1e-10),
            np.float32(1 - 1e-10),
            np.int32(y_prob.size),
            block=(blockSize, 1, 1),
            grid=(int((y_prob.size - 1) / blockSize + 1), 1, 1))

    if y_prob.shape[1] == 1:
        y_prob = gpuarray.to_gpu(np.append(1 - y_prob.get(), y_prob.get(), axis=1))

    if y_true.shape[1] == 1:
        y_true = gpuarray.to_gpu(np.append(1 - y_true.get(), y_true.get(), axis=1))


    tmp_gpu = gpuarray.GPUArray(y_prob.shape, y_prob.dtype)
    if y_prob.dtype == np.float64:
        cuLogLoss(
            y_true.gpudata,
            y_prob.gpudata,
            tmp_gpu.gpudata,
            np.int32(y_prob.size),
            block=(blockSize, 1, 1),
            grid=(int((y_prob.size - 1) / blockSize + 1), 1, 1))
    else:
        cuLogLossf(
            y_true.gpudata,
            y_prob.gpudata,
            tmp_gpu.gpudata,
            np.int32(y_prob.size),
            block=(blockSize, 1, 1),
            grid=(int((y_prob.size - 1) / blockSize + 1), 1, 1))
    #total = float(misc.sum(y_true * tmp_gpu).get())
    total = float(cumisc.sum(tmp_gpu).get())
    return (-total) / y_prob.shape[0]


def binary_log_loss(y_true, y_prob):
    """Compute binary logistic loss for classification.

    This is identical to log_loss in binary classification case,
    but is kept for its use in multilabel case.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    if y_prob.dtype == np.float64:
        cuClip(
            y_prob.gpudata,
            np.float64(1e-10),
            np.float64(1 - 1e-10),
            np.int32(y_prob.size),
            block=(blockSize, 1, 1),
            grid=(int((y_prob.size - 1) / blockSize + 1), 1, 1))
    else:
        cuClipf(
            y_prob.gpudata,
            np.float32(1e-10),
            np.float32(1 - 1e-10),
            np.int32(y_prob.size),
            block=(blockSize, 1, 1),
            grid=(int((y_prob.size - 1) / blockSize + 1), 1, 1))

    tmp_gpu = gpuarray.GPUArray(y_prob.shape, y_prob.dtype)
    if y_prob.dtype == np.float64:
        cuBinaryLogLoss(
            y_true.gpudata,
            y_prob.gpudata,
            tmp_gpu.gpudata,
            np.int32(y_prob.size),
            block=(blockSize, 1, 1),
            grid=(int((y_prob.size - 1) / blockSize + 1), 1, 1))
    else:
        cuBinaryLogLossf(
            y_true.gpudata,
            y_prob.gpudata,
            tmp_gpu.gpudata,
            np.int32(y_prob.size),
            block=(blockSize, 1, 1),
            grid=(int((y_prob.size - 1) / blockSize + 1), 1, 1))

    total = float(cumisc.sum(tmp_gpu).get())
    return (-total) / y_prob.shape[0]


LOSS_FUNCTIONS = {'squared_loss': squared_loss, 'log_loss': log_loss,
                  'binary_log_loss': binary_log_loss}
