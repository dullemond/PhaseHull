# PhaseHull

## Purpose

PhaseHull is a Python code to compute phase diagrams of solids and their melts, typically for geophysical and astrophysical purposes. PhaseHull is built on a convex hull algorithm, described in a forthcoming paper by Dullemond \& Young (2025/26). In contrast to codes such as MELTS (Ghiorso et al.) and MAGMAs (Fegley et al.), PhaseHull is not meant as a blackbox code with preset material coefficients. Instead, PhaseHull is meant as a toolbox for computing phase diagrams of any binary, ternary or higher-dimensional system of materials, allowing the user to specify the material properties through the Gibbs free energy function. A few well known models and databases are pre-installed, and more will likely come in the future. 

## Installation

Install PhaseHull by executing this in the top-level directory of this repository:

    pip install -e .

This installs a development version, i.e. if you modify the code, those changes will be reflected if you `import phasehull`. To install a static version of the current code, leave out the `-e` option.

## Examples

You can find examples in the models/ directory. They are .py files (not .ipynb files) so they should be run in an ipython session. Start iPython from the Unix/Linux/Mac command line as

   ipython --matplotlib

Then run the code using

   %run model_xxx.py

where xxx should be replaced by the model name in the models/ directory.

Note that the models use the mpltern python library for plotting ternary plots.

2025.09.11
