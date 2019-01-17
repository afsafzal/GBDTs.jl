# GBDTs.jl -- Grammar-Based Decision Trees

Grammar-based decision tree (GBDT) is an interpretable machine learning model that can be used for the classification and categorization of heterogeneous multivariate time series data. GBDTs combine decision trees with a grammar framework. Each split of the decision tree is governed by a logical expression derived from a user-supplied grammar. The flexibility of the grammar framework enables GBDTs to be applied to a wide range of problems. In particular, GBDT has been previously applied to analyze multivariate heterogeneous time series data of failures in aircraft collision avoidance systems [1].

[1] Lee et al. "Interpretable Categorization of Heterogeneous Time Series Data", preprint, 2018.

## Main Dependencies

To run the script you first need to install required packages:
```
pkg> add Reexport TikzGraphs LightGraphs StatsBase Discretizers AbstractTrees ExprRules ExprOptimization TikzPictures ArgParse HDF5 JLD
```
In addition you need to add `MultivariateTimeSeries`:
```
pkg> clone "https://github.com/sisl/MultivariateTimeSeries.jl"
```
`HDF5` may ask you to install packages on the machine.

## Usage

Please see the [example notebook](http://nbviewer.ipython.org/github/sisl/GBDTs.jl/blob/master/examples/Auslan.ipynb).

To learn GBDT model for Houston data:
```
julia learn.jl <data> <depth> [--output_dir OUTPUT_DIR] [--fuzzy] [--name NAME]
```
Example:
```
julia learn.jl data/train/ 5 --output_dir model/ --fuzzy --name factory.MAV_CMD_NAV_TAKEOFF
```
The data folder should include three files:
    1. `data.csv.gz`: a csv of all time series
    2. `labels.txt`: class label for each time series
    3. `_meta.txt`: the index of the start of a new time serie in the csv file.
The output of this script includes `NAME.jld` (by default `model.jld`) which stores the model, and `graph.pdf` that stores a representation of the tree.

To classify test data against a model:
```
julia test.jl <data> <model>
```
Example:
```
julia test.jl data/test/ model/factory.MAV_CMD_NAV_TAKEOFF.jld
```

## Maintainers:

* Ritchie Lee, ritchie.lee@sv.cmu.edu

[![Build Status](https://travis-ci.org/sisl/GBDTs.jl.svg?branch=master)](https://travis-ci.org/sisl/GBDTs.jl) [![Coverage Status](https://coveralls.io/repos/sisl/GBDTs.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/sisl/GBDTs.jl?branch=master)
