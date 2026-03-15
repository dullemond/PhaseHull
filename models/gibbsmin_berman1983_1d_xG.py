import numpy as np
import matplotlib.pyplot as plt
from mineral_systems import Berman83
import phasehull as ph

Rgas       = 8.314  # J/mol·K
T          = 2500.
P          = 1.

#components = ["CaO","SiO2"]
#components = ["SiO2","Al2O3"]
components = ["CaO","Al2O3"]
#components = ["MgO","SiO2"]
#components = ["MgO","CaO"]
#components = ["MgO","Al2O3"]

b          = Berman83(components,T)
m          = b.margules

gammafunc  = lambda x: b.margules.get_activity_coefficients_of_components(x,b.T,b.P)

crystaldb  = ph.CrystalDatabase(b.mdb,resetfunc=b.reset)
liquid     = ph.Liquid('magma',components,b.Gfunc,gammafunc=gammafunc,resetfunc=b.reset)

GMF        = ph.GibbsMinFinder(components,T,P,crystaldb=crystaldb,liquids=[liquid])

nx         = 100
xgrid      = np.linspace(0,1,nx)

print('Start')
Ysolution = np.zeros((nx,GMF.nphases))
Gsolution = np.zeros_like(xgrid)
Y      = np.zeros(GMF.nphases)
Y[0]   = 1.

from tqdm import tqdm
for ix in tqdm(range(nx)):
    x               = np.zeros(GMF.ncomp)
    x[0]            = xgrid[ix]
    x[1]            = 1-x[0]
    Y               = GMF.find_minimum(x,Y)
    Ysolution[ix,:] = Y
    Gsolution[ix]   = GMF.Gfunc(Ysolution[ix,:])
print('Done')

Gplane = (1-xgrid)*Gsolution[0]+xgrid*Gsolution[-1]

plt.figure()
plt.plot(xgrid,Gsolution-Gplane)

plt.figure()
for i in range(GMF.nphases):
    plt.plot(xgrid,Ysolution[:,i],label=GMF.Y_phase_name[i])
plt.ylabel('Y')
plt.xlabel('x')
plt.legend()

plt.show()

