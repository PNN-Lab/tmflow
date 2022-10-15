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
