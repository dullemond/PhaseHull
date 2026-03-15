import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern   # https://mpltern.readthedocs.io/en/latest/index.html
import phasehull as ph
from phasehull.phasehull_colors import linecolors,fillcolors
from phasehull import plot_ternary
from mineral_systems import Berman83,Gasparik90

relevel     = False
#relevel     = True
Rgas        = 8.314  # J/mol·K

#nrefine     = 0
#nrefine     = 2
nrefine     = 4

components  = ["SiO2","CaO","MgO"]      #  The components of the ternary
compnames   = components
endmembers  = ['MgSiO3',  'CaMgSi2O6']  #  ['Enstatite','Diopside']
endmfact    = [2.,        1.         ]

#ny          = 401
ny          = 101
y1d         = np.linspace(0,1,ny)
y           = np.zeros((ny,2))
y[:,0]      = y1d
y[:,1]      = 1-y[:,0]
#x           = 0.5 + 0.5*y
#x[:,1]      = 1-x[:,0]
#T           = 700.
#T           = 970.
#T           = 1200.+273.15    # In Fig 1 of Gasparik just below pigeonite region
#T           = 1295.+273.15    # In Fig 1 of Gasparik just below pigeonite region
T           = 1340.+273.15    # In Fig 1 of Gasparik: in the middle of the region where pigeonite (high-clino) appears
#T           = 1380.+273.15    # In Fig 1 of Gasparik: in the middle of the region where a tiny sliver of Ortho appears
#T           = 1400.+273.15    # 
#T           = 1450.+273.15    # In Fig 1 of Gasparik: in the middle of the region where there are new
#T           = 1437.+273.15    # In Fig 1 of Gasparik: in the middle of the region where there are new 
#T           = 1467.+273.15    # In Fig 1 of Gasparik: in the middle of the region where there are new 
P           = 1.

# Import the Berman 1983 model for the CaO,MgO,SiO2 system

b           = Berman83(components,T)
GfuncLiq    = lambda x: b.Gfunc(x)
liquid      = ph.Liquid('liquid',components,GfuncLiq,resetfunc=b.reset)
crystaldb   = ph.CrystalDatabase(b.mdb,resetfunc=b.reset)

# Set up Gasparik's 1990 polymorph binary model of the diopside-enstatite join solid solution

mdb         = crystaldb.dbase
def mu00func(T,P=1.):
    """
    The function used by the Gasparik model to recompute the enstatite and diopside
    mu00 values (which are the mu0 values of the protoenstatite and highclinodiopside
    solid solutions).
    """
    mdbidx      = mdb.copy().set_index('Abbrev')
    mu00En      = 2*mdbidx.loc['PrEn']['DfG']
    mu00Di      = mdbidx.loc['Diop']['DfG']
    return mu00En,mu00Di

q           = Gasparik90(T=T,P=P,mu00func=mu00func,mu00EnPolyM='proto',mu00DiPolyM='highclino')   # mu00EnPolyM='proto' because Berman83 (see below) specifies mu0 for proto enstatite
sols        = q.solnames
q.reset(T,P)
G0          = lambda x: q.Gfunc(x,isol=0)
G1          = lambda x: q.Gfunc(x,isol=1)
G2          = lambda x: q.Gfunc(x,isol=2)
G3          = lambda x: q.Gfunc(x,isol=3)
G4          = lambda x: q.Gfunc(x,isol=4)
Gg          = [G0,G1,G2,G3,G4]
solsols     = []
for i in range(5):
    solsols.append(ph.SolidSolution(sols[i],components,endmembers,Gg[i],mdb,ygrid=y,resetfunc=q.reset,endmfact=endmfact))

# Now put it all together in a PhaseHull instance

nres0      = 29 # 30
phull      = ph.PhaseHull(components,crystaldb,liquid,solsols,nres0=nres0,nrefine=nrefine)

ax         = plot_ternary(phull,order=[0,1,2],stride=4,compnames=compnames)
ax.text(1,0.5,-0.5,f'T = {T:.0f} K ({T-273.15:.0f} C)',ha='left')
ax.text(1,0.5+0.07,-0.5,f'P = 1 bar',ha='left')
plt.savefig('dien_join_ternary.pdf')

h   = 0.07  # Vertical spacing of annotations
fig = plt.figure()
ax  = fig.add_subplot(projection="ternary")
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)
x   = phull.complete_x(phull.thepoints[-1][:,:-1])
ax.scatter(x[:,0],x[:,1],x[:,2],marker='.',s=1,color='C1')
xj  = np.array([[0.5,0.25,0.25],[0.5,0.0,0.5]])
ax.plot(xj[:,0],xj[:,1],xj[:,2],color='C0')
ax.set_tlabel(compnames[0])
ax.set_llabel(compnames[1])
ax.set_rlabel(compnames[2])
ax.text(1,0.5,-0.5,f'T = {T:.0f} K ({T-273.15:.0f} C)',ha='left')
ax.text(1,0.5+0.07,-0.5,f'P = 1 bar',ha='left')
ax.text(1,-0.5,0.5,f'nres0 = {nres0}',ha='right')
ax.text(1-h,-0.5+0.5*h,0.5+0.5*h,f'Refinement levels: {nrefine}',ha='right')
plt.savefig('dien_join_ternary_gridpoints.pdf')
