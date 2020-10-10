## FYS-STK4155 project 1 - Regression analysis and resampling methods
### Code structure
Parameters for each part of the project is located in an individual .py file, named part_x.py (x = a, b, c, d, e, g). Simply run each file by

```
python part_x.py
```

Note that the runtime is somewhat long for some of the programs, since all experiments are repeated and averaged. If you wish to use less time (but get worse data), please edit the desired file and change the ```repetitions``` parameter. You can save more time by editing the number of ridge and lasso regression parameters where applicable. These parameters are named ```n_lambdas``` and ```n_alphas``` respectively. All present parameters are guaranteed to produce similar results to those you find in the report and in ```/fig```.

A file ```common.py``` contains all the common functionality for most of the tasks, including but not limited to, ```create_design_matrix```, ```ols```, ```cross_validation```, ```bootstrap```. ```common.py``` is not run directly, but imported by the other programs.

```common.py``` contains a class ```Regression``` where certain common elements for all regression techniques are gathered, like set-up of the design matrix and its parameters, and splitting and scaling of the data.

General usage (this is handled by the programs for each subtask):

``` Python
q = Regression(number_of_data_points, noise_factor, max_polynomial_degree)  # Design matrix is created, noise is added to data, data is split and scaled.

for current_degree in list_of_degrees:
    return_values = q.standard_least_squares_regression(current_degree)
    return_values = q.bootstrap(current_degree, number_of_bootstrap_resamples)
    return_values = q.cross_validation(current_degree, number_of_folds)
```

The final task, ```part_g.py```, is a bit different since it shall handle the terrain data. A new class, ```Terrain```, is introduced which inherits from ```Regression```, but the constructor is overwritten with a constructor which handles the terrain data from the file ```SRTM_data_Norway_1.tif```. Usage of ```Terrain``` is programmed in ```part_g.py``` and is in general identical to that of ```Regression```, save for the initialization,

```
q = Terrain(max_polynomial_degree, step_size_of_terrain_data)
```

```part_g.py``` slices the terrain data to have an equal amount of x and y values for simplicity.


### Report
The report is located in ```/doc```, and all figures for the report is located in ```/fig```.

### Compatibility
All code is programmed in ```Python 3.8.5```. Earlier versions will probably work fine, but we guarantee full compatibility only with ```3.8.5```. Python packages used and their versions are: ```sklearn 0.23.2```, ```numpy 1.19.1```, ```matplotlib 3.3.1```, ```imageio 2.9.0```.