## FYS-STK4155 project 3 - The Bitcoin network and Bitcoin price predictions with long short-term memory recurrent neural networks
### Code structure
Parameters for each part of the project is located in an individual .py file in the ```src``` folder. ```analyse_dropout.py``` plots a dropout rate analysis. ```analyse_dropout_price_comparison.py``` generates a thorough price prediction for different dropout rates, employing the long short-term memory neural network. ```grid_search_best_parameters.py``` performs a grid search over all hyperparameters to find the combination which yields the smallest mean squared error. ```plot_btc_price.py``` simply plots the historical BTC/USD price data. ```regression.py``` reproduces the historical BTC/USD data using linear regression. ```lstm.py``` contains the framework for the LSTM RNN.

The runtime is very long for most of the programs, but lucky for you ```lstm.py``` saves the network state as ```.npy``` files. The network checks whether the save state for the given parameters already exists and loads the save state should it exist (we have uploaded all save states so you dont have to calculate them). All save state files are located in ```src/saved_state```.

The bitcoin price data is located in ```src/data/btc-usd-max.csv```.

### Parallelization
Most of the code is parallelized using ```ray```. It is tested to work fine on the latest Ubuntu and macOS.

### Report
The report is located in ```doc/```, and all figures for the report is located in ```fig/```.

### Compatibility
All code is programmed in ```Python 3.8.5```. Earlier versions will probably work fine, but we guarantee full compatibility only with ```3.8.5```. Python packages used and their versions are: ```sklearn 0.23.2```, ```numpy 1.19.1```, ```matplotlib 3.3.1```, ```ray 1.0.1```, ```tensorflow 2.3.1```, ```Keras 2.4.3```.