import sys
sys.path.append("../../")

import numpy as np
import timeit
from scikitlearn_plus.neural_network import MLPRegressor_cuda
from scikitlearn_plus.neural_network import MLPRegressor

import keras


import time

testData = np.random.randn(10,10)

X = np.random.randn(100,10)
y = np.random.randn(100)

X = X.astype(np.float32)
y = y.astype(np.float32)

layer = (10, 10, 10, 10, 10)

dnnInput = keras.layers.Input(shape = (X.shape[1],))
dnn = keras.layers.Dense(10, activation='linear')(dnnInput)
dnn = keras.layers.Dense(10, activation='linear')(dnn)
dnn = keras.layers.Dense(10, activation='linear')(dnn)
dnn = keras.layers.Dense(10, activation='linear')(dnn)
dnn = keras.layers.Dense(10, activation='linear')(dnn)
dnn = keras.layers.Dense(1, activation='linear')(dnn)
DNN = keras.models.Model(inputs=dnnInput, outputs=dnn)
DNN.compile(loss='mean_absolute_error', optimizer='adam')
DNN.summary()
start = time.time()
history = DNN.fit(
            X, y,
            batch_size=1024,
            )
stop = time.time()
print('fit time:', stop - start)
print(DNN.predict(testData))
print()




print('--------------sgd--------------')
# sklearn MLPRegressor sgd
regSgd = MLPRegressor(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=layer, random_state=1, max_iter=10, verbose=True)

# scikitlearn_plus MLPRegressors
regSgd_cuda = MLPRegressor_cuda(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=layer, random_state=1, max_iter=10, verbose=True)

print("------sklearn------")
start = time.time()
regSgd.fit(X, y) 
stop = time.time()
print('fit time:', stop - start)
print(regSgd.predict(testData))
print()


print("------scikitlearn_plus------")
start = time.time()
regSgd_cuda.fit(X,y)
stop = time.time()
print('fit time:', stop - start)
print(regSgd_cuda.predict(testData))
print()


print('--------------adam--------------')
# sklearn MLPRegressor adam
regadam = MLPRegressor(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=layer, random_state=1, max_iter=1000, verbose=True)

# scikitlearn_plus MLPRegressor
regadam_cuda = MLPRegressor_cuda(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=layer, random_state=1, max_iter=1000, verbose=True)

print("------sklearn------")
start = time.time()
regadam.fit(X, y) 
stop = time.time()
print('fit time:', stop - start)
print(regadam.predict(testData))
print()


print("------scikitlearn_plus------")
start = time.time()
regadam_cuda.fit(X,y)
stop = time.time()
print('fit time:', stop - start)
print(regadam_cuda.predict(testData))
print()