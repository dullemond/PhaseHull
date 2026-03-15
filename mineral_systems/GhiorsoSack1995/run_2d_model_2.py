import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern   # https://mpltern.readthedocs.io/en/latest/index.html
import phasehull as ph
from phasehull.phasehull_plottools import *
from model_GhiorsoSack1995 import GhiorsoSack95

relevel    = False
#relevel    = True

#T          = 1400.+273.15
T          = 1250.
#T          = 1290.

nrefine    = 4

components = ["SiO2","CaSiO3","Al2O3"]
#components = ["SiO2","CaSiO3","Mg2SiO4"]
#components = ["SiO2","Mg2SiO4","Al2O3"]
#components = ["CaSiO3","Mg2SiO4","Al2O3"]
compnames  = components
gs         = GhiorsoSack95(components,T)
def Gfunc(x):
    return gs.Gfunc(x)

crystaldb  = ph.CrystalDatabase(gs.mdb)
liquid     = ph.Liquid('magma',components,Gfunc)

nres0      = 30
phull      = ph.PhaseHull(components,crystaldb,liquid,nres0=nres0,nrefine=nrefine,T=T)

ax = plot_ternary(phull,order=[0,1,2],stride=4)
h = 0.07  # Vertical spacing of annotations
ax.text(1,-0.5,0.5,f'nres0 = {nres0}',ha='right')
ax.text(1-h,-0.5+0.5*h,0.5+0.5*h,f'Refinement levels: {nrefine}',ha='right')
