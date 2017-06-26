import sys
sys.path.append("../../")

from scikitlearn_plus.neural_network import _base_cuda
from scikitlearn_plus.neural_network import _base

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np

import timeit

correctnessTestTimes = 5
performanceTestTimes = 100

a = np.random.randn(1000000,10)
b = np.random.randn(1000000,10)

a = a.astype(np.float32)
b = b.astype(np.float32)

# origin
print('origin data')
print(a)
print(b)
print()

a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)

# gpu
print('gpu data')
print(a_gpu)
print(b_gpu)
print()

def correctnessTest():
    # activation
    print('activation')
    print()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    print('identity')
    for i in range(correctnessTestTimes):
        print(np.allclose(_base.identity(a), _base_cuda.identity(a_gpu).get()))
    print()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    print('tanh')
    for i in range(correctnessTestTimes):
        print(np.allclose(_base.tanh(a), _base_cuda.tanh(a_gpu).get()))
    print()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    print('logistic')
    for i in range(correctnessTestTimes):
        print(np.allclose(_base.logistic(a), _base_cuda.logistic(a_gpu).get()))
    print()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    print('relu')
    for i in range(correctnessTestTimes):
        print(np.allclose(_base.relu(a), _base_cuda.relu(a_gpu).get()))
    print()

    print()

    # derivatives
    print('derivatives')
    print()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    print('identity')
    for i in range(correctnessTestTimes):
        _base.inplace_identity_derivative(a, b)
        _base_cuda.inplace_identity_derivative(a_gpu, b_gpu)
        print(np.allclose(b, b_gpu.get()))
    print()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    print('tanh')
    for i in range(correctnessTestTimes):
        _base.inplace_tanh_derivative(a, b)
        _base_cuda.inplace_tanh_derivative(a_gpu, b_gpu)
        print(np.allclose(b, b_gpu.get()))
    print()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    print('logistic')
    for i in range(correctnessTestTimes):
        _base.inplace_logistic_derivative(a, b)
        _base_cuda.inplace_logistic_derivative(a_gpu, b_gpu)
        print(np.allclose(b, b_gpu.get()))
    print()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    print('relu')
    for i in range(correctnessTestTimes):
        _base.inplace_relu_derivative(a, b)
        _base_cuda.inplace_relu_derivative(a_gpu, b_gpu)
        print(np.allclose(b, b_gpu.get()))
    print()

    print()

    # loss function
    print('loss function')
    print()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    print('squared loss')
    for i in range(correctnessTestTimes):
        c = _base.squared_loss(a, b)
        d = _base_cuda.squared_loss(a_gpu, b_gpu)
        print(c, d)
        print(np.isclose(c, d))
    print()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    print('log loss')
    for i in range(correctnessTestTimes):
        c = _base.log_loss(a, b)
        d = _base_cuda.log_loss(a_gpu, b_gpu)
        print(c, d)
        print(np.isclose(c, d))
    print()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    print('binary log loss')
    for i in range(correctnessTestTimes):
        c = _base.binary_log_loss(a, b)
        d = _base_cuda.binary_log_loss(a_gpu, b_gpu)
        print(c, d)
        print(np.isclose(c, d))
    print()

def cpuPerformanceTest():
    # activation
    _base.identity(a)
    _base.tanh(a)
    _base.logistic(a)
    _base.relu(a)

    # derivatives
    _base.inplace_identity_derivative(a, b)
    _base.inplace_tanh_derivative(a, b)
    _base.inplace_logistic_derivative(a, b)
    _base.inplace_relu_derivative(a, b)

    # loss function
    _base.squared_loss(a, b)
    _base.log_loss(a, b)
    _base.binary_log_loss(a, b)

def dataToGpu():
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)

def gpuPerformanceTest():
    global a_gpu, b_gpu
    # activation
    _base_cuda.identity(a_gpu)
    _base_cuda.tanh(a_gpu)
    _base_cuda.logistic(a_gpu)
    _base_cuda.relu(a_gpu)

    # derivatives
    _base_cuda.inplace_identity_derivative(a_gpu, b_gpu)
    _base_cuda.inplace_tanh_derivative(a_gpu, b_gpu)
    _base_cuda.inplace_logistic_derivative(a_gpu, b_gpu)
    _base_cuda.inplace_relu_derivative(a_gpu, b_gpu)

    # loss function
    _base_cuda.squared_loss(a_gpu, b_gpu)
    _base_cuda.log_loss(a_gpu, b_gpu)
    _base_cuda.binary_log_loss(a_gpu, b_gpu)

if __name__ == '__main__':
    '''
    print('correctness check')
    print()
    correctnessTest()
    print()
    '''

    print('performance test')
    print()
    print('cpu:')
    print(timeit.timeit("cpuPerformanceTest()", setup="from __main__ import cpuPerformanceTest", number=performanceTestTimes))
    print()
    print('gpu:')
    print(timeit.timeit("gpuPerformanceTest()", setup="from __main__ import gpuPerformanceTest", number=1))
    print(timeit.timeit("gpuPerformanceTest()", setup="from __main__ import gpuPerformanceTest", number=performanceTestTimes))