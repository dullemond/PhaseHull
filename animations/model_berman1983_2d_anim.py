import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern   # https://mpltern.readthedocs.io/en/latest/index.html
import phasehull as ph
from phasehull import plot_ternary
from mineral_systems.Berman1983 import Berman83

relevel    = False
#relevel    = True

T          = 1400.+273.15

#nrefine    = 3
#stride     = 4
nrefine    = 2
stride     = 4

components = ["SiO2","CaO","Al2O3"]
#components = ["SiO2","CaO","MgO"]
#components = ["SiO2","MgO","Al2O3"]
#components = ["CaO","MgO","Al2O3"]
compnames  = components
b          = Berman83(components,T)
def Gfunc(x):
    return b.Gfunc(x)

crystaldb  = ph.CrystalDatabase(b.mdb,resetfunc=b.reset_crystals)
liquid     = ph.Liquid('magma',components,Gfunc,resetfunc=b.reset_liquid)

nres0      = 30
phull      = ph.PhaseHull(components,crystaldb,liquid,nres0=nres0,nrefine=nrefine)
ax         = plot_ternary(phull,stride=4,compnames=components)

Tmin       = 1000 + 273.15
Tmax       = 1800 + 273.15
nT         = 80
Tgrid      = np.linspace(Tmin,Tmax,nT)
dnT        = 4
nnT        = nT*dnT

i = 0
for k in range(nT):
    T = Tgrid[k]    
    phull.reset(T=T)
    plt.cla()
    ax     = plot_ternary(phull,stride=stride,compnames=components,ax=ax)
    for l in range(dnT):
        plt.savefig('image_'+f'{i:03d}'+'.png')
        i += 1
