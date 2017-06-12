import sys
sys.path.append("../../")

from scikitlearn_plus.neural_network import _base_cuda
from scikitlearn_plus.neural_network import _base

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit

import numpy
a = numpy.random.randn(4,4)

a = a.astype(numpy.float32)

a_gpu = gpuarray.to_gpu(a)

a_gpu = _base_cuda.relu(a_gpu)

print (a)
print (a_gpu.get())
print (_base.relu(a))