import sys
sys.path.append("../../")

from scikitlearn_plus.neural_network import MLPRegressor as MLPRegressor_plus
from sklearn.neural_network import MLPRegressor

X = [[0., 0.], [1., 1.]]
y = [0, 1]

# sklearn MLPRegressor
clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

# scikitlearn_plus MLPRegressor
clf_plus = MLPRegressor_plus(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)



clf.fit(X, y)   

print("------sklearn------")
print(clf.predict([[2., 2.], [-1., -2.]]))
print([coef.shape for coef in clf.coefs_])
print()

clf_plus.fit(X,y)

print("------scikitlearn_plus------")
print(clf_plus.predict([[2., 2.], [-1., -2.]]))
print([coef.shape for coef in clf_plus.coefs_])
print()
