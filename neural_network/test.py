import numpy as np
import pycuda.gpuarray as gpuarray


a = np.array(range(20))

print(a)

a_gpu = gpuarray.to_gpu(a)




