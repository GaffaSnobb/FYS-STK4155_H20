## FYS-STK4155 project 2 - Classification and regression, from linear and logistic regression to neural networks
### Code structure
Parameters for each part of the project is located in an individual .py file in the ```src``` folder, named ```part_x.py``` (x = a, b, c, d, e). Simply run each file by

```
python part_x.py
```

Note that the runtime is somewhat long for some of the programs, since all experiments are repeated and averaged. If you wish to use less time (but get worse data), please edit the desired file and change the ```repetitions``` parameter. Several of the tasks use pre-calculated data located in ```src/data_files``` as the calculations are very heavy. You can force-run the calculations by removing the data files from the folder, forcing the program to calculate the data again. Much of the code is parallelized with ```ray``` to make the calculations feasible


A file ```common.py``` contains all the common functionality for some of the tasks, including the regression code from project 1. ```common.py``` is not run directly, but imported by the other programs.

```common.py``` contains a class ```Regression``` where certain common elements for all regression techniques are gathered, like set-up of the design matrix and its parameters, and splitting and scaling of the data.

The file ```neural_network.py``` contains, not surprisingly, the neural network. A superclass ```_FFNN``` contains the general framework of the neural networks, while subclasses ```FFNNClassifier```, ```FFNNRegressor```, and ```LogisticRegressor``` inherits these common methods, like initial set-up of weights and biases, the feedforward and backpropagation algorithms. ```FFNNClassifier``` is the classifier and sets up the initial state a bit different from ```FFNNRegressor```, using one-hot representations of the y-data. ```FFNNRegressor``` needs the y-data in a different shape, and sets the data up accordingly, and that is mostly the difference between the two. ```LogisticRegressor``` inherits from ```FFNNClassifier```, since the logistic regressor simply is a neural network with zero hidden layers. Pretty neat.

```activation_functions.py``` contains all the activation functions, to keep things tidy.

### Parallelization
Much of the code is parallelized using ```ray```. We have had issues using ```ray``` on Ubuntu, but it works just fine on macOS.

### Testing
Testing is performed by ```test.py``` which compares our implementation of a multi-layer neural network, using a single hidden layer, to the hard-coded single layer implementation from Mortens week 41 lecture notes. More or less every single value in all vectors and matrices are checked to agree within a tolerance. Run with ```pytest test.py```.


### Report
The report is located in ```doc/```, and all figures for the report is located in ```fig/```.

### Compatibility
All code is programmed in ```Python 3.8.5```. Earlier versions will probably work fine, but we guarantee full compatibility only with ```3.8.5```. Python packages used and their versions are: ```sklearn 0.23.2```, ```numpy 1.19.1```, ```matplotlib 3.3.1```, ```pytest 6.1.1```, ```seaborn 0.11.0```, ```ray 1.0.1```.