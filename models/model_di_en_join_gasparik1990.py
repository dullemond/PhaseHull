import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mineral_systems import Gasparik90
from mineral_systems import Berman1983
import phasehull as ph

#ny     = 101
ny     = 401
y1d    = np.linspace(0,1,ny)
y      = np.zeros((ny,2))
y[:,0] = y1d
y[:,1] = 1-y[:,0]
#x      = 0.5 + 0.5*y
#x[:,1] = 1-x[:,0]
x      = y
#T      = 700.
#T      = 970.
#T      = 1200.+273.15    # In Fig 1 of Gasparik just below pigeonite region
#T      = 1295.+273.15    # In Fig 1 of Gasparik just below pigeonite region
T      = 1340.+273.15    # In Fig 1 of Gasparik: in the middle of the region where pigeonite (high-clino) appears
#T      = 1380.+273.15    # In Fig 1 of Gasparik: in the middle of the region where a tiny sliver of Ortho appears
#T      = 1450.+273.15    # In Fig 1 of Gasparik: in the middle of the region where there are new 
P      = 1.
q      = Gasparik90(T=T,P=P,mu00EnPolyM='proto')   # mu00EnPolyM='proto' because Berman83 (see below) specifies mu0 for proto enstatite

#q.incleint = 0.  # Uncomment to include only the entropy part

# Note that the entropy terms in the Gibbs energies refer to
# the mixing of moles of formula units of MgCaSi2O6 and Mg2Si2O6,
# each of which are 4 moles of components (SiO2,MgO,CaO) if we
# compare it to the Berman 1983 model. So the RT(xlnx) terms
# in the Gasparik model are energy per mole of formula unit
# MgCaSi2O6,Mg2Si2O6, not per mole of formula unit SiO2,MgO,CaO. 

b           = Berman1983.Berman83(T=T,P=P)

components  = ['Mg2Si2O6','CaMgSi2O6']  #  ['Enstatite','Diopside']
endmembers  = ['MgSiO3',  'CaMgSi2O6']  #  ['Enstatite','Diopside']
endmfact    = [2.,        1.         ]

sols        = q.solnames

mdb         = b.mdb.copy()    ##pd.read_fwf('minerals.fwf')
dbase       = ph.extract_from_mineral_database_based_on_components(mdb,components)
cdb         = ph.CrystalDatabase(dbase)

dbaseidx    = dbase.copy().set_index('Abbrev')
#mu00En      = 0.
#mu00Di      = 0.
mu00En      = dbaseidx.loc['PrEn']['mfDfG']
mu00Di      = dbaseidx.loc['Diop']['mfDfG']
q.reset(T,P,mu00En=mu00En,mu00Di=mu00Di)

G0          = lambda x: q.Gfunc(x,isol=0)
G1          = lambda x: q.Gfunc(x,isol=1)
G2          = lambda x: q.Gfunc(x,isol=2)
G3          = lambda x: q.Gfunc(x,isol=3)
G4          = lambda x: q.Gfunc(x,isol=4)
Gg          = [G0,G1,G2,G3,G4]

solsols     = []
for i in range(5):
    solsols.append(ph.SolidSolution(sols[i],components,endmembers,Gg[i],dbase,ygrid=y,endmfact=endmfact))

phull       = ph.PhaseHull(components,crystals=cdb,solsols=solsols)

xbase       = np.array([[0.0,1.0],[1.,0.]])
Gbase       = np.array([mu00Di,mu00En])
relevel     = True
#xbase       = None
#Gbase       = None
#relevel     = False
ax = ph.plot_binary_xG(phull,relevel=relevel,xbase=xbase,Gbase=Gbase,ymin=-2.5,ymax=1,xlabel='y')
#ax.set_xlim(xmax=0.76,xmin=1.0)
plt.savefig('dien_join_xG.pdf')
