import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import phasehull as ph
from mineral_systems.Berman1983 import Berman83
from phasehull.phasehull_xT_tools import *
from phasehull.phasehull_colors import *

#relevel    = False
relevel    = True

#nT         = 200
#nx         = 400
nT         = 100
nx         = 100

#TminC      = 1200.
TminC      = 1000.
#TmaxC      = 3200.
TmaxC      = 2800.
Tmin       = TminC+273.15
Tmax       = TmaxC+273.15
T          = np.linspace(Tmin,Tmax,nT)

#endmembers = ["CaO","SiO2"]
#endmembers = ["SiO2","Al2O3"]
#endmembers = ["CaO","Al2O3"]
endmembers = ["MgO","SiO2"]
b          = Berman83(endmembers,T[0])
def Gfunc(x):
    return b.Gfunc(x)

crystaldb  = ph.CrystalDatabase(b.mdb,resetfunc=b.reset_crystals)
liquid     = ph.Liquid('magma',endmembers,Gfunc,resetfunc=b.reset_liquid)

phull      = ph.PhaseHull(endmembers,crystaldb,liquid,nres0=nx,nocompute=True,incl_ptnames=True)

sim_list   = []
ism_list   = []

for iT in range(nT):
    print(f'iT = {iT}')
    phull.reset(T[iT])
    simplices = phull.thesimplices[-1]
    isims     = np.argsort(simplices['x'][:,:,0].min(axis=-1))
    sim_list.append(simplices)
    ism_list.append(isims)

stypes     = list(find_all_stypes_in_series_of_phase_diagrams(sim_list))
stype_dict = {}
for st in stypes:
    stype_dict[st] = []

for iT in range(nT):
    simplices = sim_list[iT]
    isims     = ism_list[iT]
    for st in stypes:
        stype_dict[st].append(find_stypes_1d(simplices,isims,st))

region_dict = {}
for st in stypes:
    region_dict[st] = collect_x_T_phase_regions_1d(stype_dict[st],T,Tmargin=True)

plt.figure()
for st in stypes:
    for name in region_dict[st]:
        region = region_dict[st][name]
        size   = 6
        text   = name.replace('-1','Liq')
        if (region['x'].max()-region['x'].min())<0.1:
            text = text.replace('+','\n+\n')
            if (region['T'].max()-region['T'].min())<60:
                size*=(2/3)
        plt.fill(region['x'],region['T']-273.15,color=fillcolors[st]);  plt.plot(region['x'],region['T']-273.15,color='black',linewidth=1)
        plt.text(region['xcen'],region['Tcen']-273.15-10,text,ha='center',va='center',size=size)
plt.ylim(TminC,TmaxC)
plt.xlabel('x (molar)')
plt.ylabel('T [Celsius]')
ytxt = plt.gca().get_ylim()[0] - 200
plt.text(0,ytxt,ph.latexify_chemical_formula(endmembers[1]),ha='center')
plt.text(1,ytxt,ph.latexify_chemical_formula(endmembers[0]),ha='center')
ytxt = plt.gca().get_ylim()[1] - 200
plt.text(0.05,ytxt,f'P = 1 bar')
ytxt = plt.gca().get_ylim()[1] - 800
plt.text(0.35,ytxt,f'Liquid',size=6)
plt.title('Berman83')
plt.savefig(f'fig_binary_{endmembers[0]}_{endmembers[1]}_x_T.pdf')
plt.show()
