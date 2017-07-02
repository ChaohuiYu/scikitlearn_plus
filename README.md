
## Welcome to ScikitPlus

High Performance with CUDAs
Scikit-Plus provides MLPRegressor with GPU support, which is able to speed up to 12x  compared with the original MLPRegressor.

Highly Compatible with Scikit-learn
Scikit-Plus‘s interface is highly compatible with scikit-learn; in most cases, it can be used as drop-in replacement.

```
from scikitlearn_plus.neural_network import MLPRegressor_cuda
from scikitlearn_plus.neural_network import MLPRegressor

# sklearn MLPRegressor sgd
regSgd = MLPRegressor(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=layer, random_state=1, max_iter=1000, verbose=True)

# scikitlearn_plus MLPRegressor
regSgd_cuda = MLPRegressor_cuda(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=layer, random_state=1, max_iter=1000, verbose=True)

```



@inproceedings
{sklearn_api,
  author    = {Lars Buitinck and Gilles Louppe and Mathieu Blondel and
               Fabian Pedregosa and Andreas Mueller and Olivier Grisel and
               Vlad Niculae and Peter Prettenhofer and Alexandre Gramfort
               and Jaques Grobler and Robert Layton and Jake VanderPlas and
               Arnaud Joly and Brian Holt and Ga{\"{e}}l Varoquaux},
  title     = {{{API}} design for machine learning software: experiences from the scikit-learn project},
  booktitle = {ECML PKDD Workshop: Languages for Data Mining and Machine Learning},
  year      = {2013},
  pages = {108--122},
}
