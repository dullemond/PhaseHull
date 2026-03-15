# PhaseHull

## Purpose

PhaseHull is a Python code to compute phase diagrams of solids and their melts, typically for geophysical and astrophysical purposes. PhaseHull is built on a convex hull algorithm, described in a forthcoming paper by Dullemond \& Young (2026). In contrast to codes such as MELTS (Ghiorso & Sack 1995) and MAGMAs (Fegley et al.), PhaseHull is not meant as a blackbox code with preset material coefficients. Instead, PhaseHull is meant as a toolbox for computing phase diagrams of any binary, ternary or higher-dimensional system of materials, allowing the user to specify the material properties through the Gibbs free energy function.

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

## Implemented models and databases

A few well-known models and databases are pre-installed (in the mineral_systems/ directory), and more will likely come in the future. But use these pre-installed models and databases with care, as they are often simplified versions of the original models from the literature, and they may contain errors, because their data, in many cases, are obtained from digitized scanned versions of the original papers. Currently provided systems are:

 - Solid/Liquid system of SiO2-CaO-Al2O3 by Berman & Brown (1984)
 - Solid/Liquid system of SiO2-MgO-CaO-Al2O3 by Berman (1983)
 - Solid/Liquid system of SiO2-TiO2-Al2O3-Fe2O3-MgCr2O4-Fe2SiO4-Mg2SiO4-CaSiO3-Na2SiO3-KAlSiO4 by Ghiorso & Sack (1995)
 - Solid mineral system of Na2O-K2O-CaO-MgO-FeO-Fe2O3-Al2O3-SiO2-TiO2-H2O-CO2 of Berman (1988)
 - Solid solution system for CaAl2Si2O8-NaAlSi3O8-KAlSi3O8 Feldspar of Elkins & Grove (1990)
 - Solid solution system for Diopside-Enstatite of Gasparik (1990)

2025.09.11 / 2026.03.15
