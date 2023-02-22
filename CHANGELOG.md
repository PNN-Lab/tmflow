# Changelog

## v0.5.0 - 22.02.2023

### Changed

- Minimum required python version is now 3.9
- Required library versions now have a wider version range

## v0.4.0 - 26.01.2023

### Added

- New module "map" to construct a matrix Taylor map for the given system of ODEs
- A module to numerically solve systems of ODEs – "tm_solver" based on the "map" module.
- Examples to illustrate the use of "tm_solver" in the numerical solution of example ODEs (forward problem):
  - Examples of ODE systems (Lotka-Volterra system, Robertson system, generalized Lotka-Volterra system).
  - Example of ODE system with symbolic parameters on the right-hand side
- A "learn" module example based on the Lotka-Volterra system to show the ability of PNN
to learn system dynamics from data (inverse problem) 
and to demonstrate better training results of pre-initialized PNN
over PNN without initialization 

### Changed

- Simplified tmflow imports: 
```from tmflow.learn import KroneckerPolynomialLayer``` instead of 
```from tmflow.learn.kronecker_polynomial_layer import KroneckerPolynomialLayer```