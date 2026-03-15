import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern   # https://mpltern.readthedocs.io/en/latest/index.html
import phasehull as ph
from phasehull.phasehull_plottools import *
from model_GhiorsoSack1995 import GhiorsoSack95

relevel    = False
#relevel    = True

#T          = 2100.+273.15
#T          = 1400.+273.15
#T          = 2300.+273.15
#T          = 2700.+273.15
#T          = 1200.+273.15
#T          = 1828.70555556
T          = 1400.+273.15
#T          = 2000.

nrefine    = 2

components = ["SiO2","CaSiO3","Al2O3"]
#components = ["SiO2","CaSiO3","Mg2SiO4"]
#components = ["SiO2","Mg2SiO4","Al2O3"]
#components = ["CaSiO3","Mg2SiO4","Al2O3"]
compnames  = components
gs         = GhiorsoSack95(components,T)
def Gfunc(x):
    return gs.Gfunc(x)

crystaldb  = ph.CrystalDatabase(gs.mdb,resetfunc=gs.reset_crystals)
liquid     = ph.Liquid('magma',components,Gfunc,resetfunc=gs.reset_liquid)

nres0      = 30
phull      = ph.PhaseHull(components,crystaldb,liquid,nres0=nres0,nrefine=nrefine)

fig,ax,cb  = interactive_ternary(phull,stride=4,return_callback=True)


