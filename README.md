
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

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ChaohuiYu/scikit_plus/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
scikit-learn plus






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
