import sys
sys.path.append("../../")

import numpy as np
import timeit
from scikitlearn_plus.neural_network import MLPRegressor_cuda as MLPRegressor_plus
from scikitlearn_plus.neural_network import MLPRegressor

X = [[0., 0., 0.], [1., 1., 1.]]
y = [0, 1]

testData = np.random.randn(10,10)

X = np.random.randn(10000,10)
y = np.random.randn(10000)

X = X.astype(np.float32)
y = y.astype(np.float32)

# sklearn MLPRegressor
regSgd = MLPRegressor(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(5, 5), random_state=1, max_iter=2000, verbose=True)

# scikitlearn_plus MLPRegressor
regSgd_plus = MLPRegressor_plus(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(5, 5), random_state=1, max_iter=2000, verbose=True)


print("------sklearn------")
regSgd.fit(X, y) 
print(regSgd.predict(testData))
print([coef for coef in regSgd.coefs_])
print()


print("------scikitlearn_plus------")
regSgd_plus.fit(X,y)
print(regSgd_plus.predict(testData))
print([coef for coef in regSgd_plus.coefs_])
print()

'''
if __name__ == '__main__':
    print('correctness check')
    print()
    correctnessTest()
    print()

    print('performance test')
    print()
    print('cpu:')
    print(timeit.timeit("cpuPerformanceTest()", setup="from __main__ import cpuPerformanceTest", number=performanceTestTimes))
    print()
    print('gpu:')
    print(timeit.timeit("dataToGpu()", setup="from __main__ import dataToGpu", number=1))
    print(timeit.timeit("gpuPerformanceTest()", setup="from __main__ import gpuPerformanceTest", number=performanceTestTimes))
'''