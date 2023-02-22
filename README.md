Taylor map flow is a package for a 'flowly' construction and learning of polynomial neural networks (PNN) for time-evolving process prediction.

Based on the input time-series data, it provides:
  - (construct) a module to construct ordinary differential equations (ODEs) in the polynomial form
  - (map) a module to construct a matrix Taylor map for ODEs
  - (learn) a TensorFlow-based module to build and train a polynomial neural network (PNN).
Taylor map matrices can be used as PNN initial weights.

PNN built in this flow way is strongly connected with ordinary differential equations.
This combination reveals the data-underlying deterministic process without manual equation derivation 
and allows treating cases even when only small datasets or partial measurements are available. 
The proposed hybrid models provide explainable and interpretable results to leverage optimal control applications.

'Construct', 'map', and 'learn' modules can be used sequentially or independently from each other.

# Construct
Module for Taylor Mapping construction.
It gives a tool to construct Taylor mapping for the given system of  Ordinary Differential Equations with polynomial right-hand side.

## Features

- Matrices of Taylor mapping completely describe the given ODEs' dynamics up to the required degree of accuracy for any initial conditions.

- It is possible to compute Taylor mapping matrices in a symbolic form.

- If the right-hand side of the ODE system depends on parameters, it is possible to construct parameter-dependent TM matrices.

## Interface 

The list of inputs for the module:
- NumPy array of matrices P in the ODEs right-hand side
- order of nonlinearity for Taylor mapping (order)
- Taylor mapping step (h)

The module output:
- NumPy array of Taylor mapping matrices R



# Installation

## Base library install
```shell
pip install tmflow
```
## Installation with additional dependencies for GitHub examples
```shell
pip install tmflow[examples]
```
## Library development installation for GitHub sources
The library can be installed in development mode, 
which enables library loading from the disk with all saved code changes. 
You may also need to uninstall the public version of the library.

Commands should be executed from the library project root folder.
```shell
pip uninstall tmflow
pip install -r requirements.txt
pip install -e .
```