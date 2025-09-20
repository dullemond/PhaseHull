import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import phasehull as ph
from mineral_systems.Berman1983 import Berman83

#relevel    = False
relevel    = True

#T          = 2100.+273.15
#T          = 2000.
#T          = 1250.+273.15
#T          = 3200.
#T          = 1400.+273.15

#T          = 1500.
#T          = 2000.
T          = 2500.

#components = ["CaO","SiO2"]
#components = ["SiO2","Al2O3"]
components = ["CaO","Al2O3"]
#components = ["MgO","SiO2"]
#components = ["MgO","CaO"]
#components = ["MgO","Al2O3"]

b          = Berman83(components,T)
def Gfunc(x):
    return b.Gfunc(x)

crystaldb  = ph.CrystalDatabase(b.mdb)
liquid     = ph.Liquid('magma',components,Gfunc)

phull      = ph.PhaseHull(components,crystaldb,liquid,nres0=100)

isel_allc  = phull.select_simplices_of_a_given_kind('allcryst')
isel_liq   = phull.select_simplices_of_a_given_kind('liquid')
isel_cltie = phull.select_simplices_of_a_given_kind('tieline_c1l1')
isel_inmis = phull.select_simplices_of_a_given_kind('tieline_c0l2')

simplices  = phull.thesimplices[-1]

phull.define_a_relevelling_plane(components=components)

G_liq      = phull.thepoints_liq[-1][:,-1]
G_cryst    = phull.thepoints_cryst[-1][:,-1]
ylabel     = 'G [kJ/mol]'

if relevel:
    simplices['G'] -= phull.G_relevelling_plane(simplices['x'])
    G_liq          -= phull.G_relevelling_plane(phull.thepoints_liq[-1][:,:-1])
    G_cryst        -= phull.G_relevelling_plane(phull.thepoints_cryst[-1][:,:-1])
    ylabel          = r'$\bar G$ [kJ/mol]'

#colors     = {'allcryst':'C1','liquid':'C0','cryst_1_liq_1':'C4','crystals':'C3','inmisc_liquids':'C9'}
#colors     = {'allcryst':'C1','liquid':'deepskyblue','tieline_c1l1':'C4','crystals':'C3','tieline_c0l2':'C9'}
colors     = {'allcryst':'C1','liquid':'#6090E0','tieline_c1l1':'C2','crystals':'C3','tieline_c0l2':'C9'}
#colors     = {'allcryst':'C1','liquid':'C0','tieline_c1l1':'C2','crystals':'C3','tieline_c0l2':'C9'}
labels     = {'allcryst':'cryst-cryst tie line','liquid':'liquid','tieline_c1l1':'cryst-liquid tie line','tieline_c0l2':'inmisc liquids'}
unit       = 1e3

plt.figure()
plt.plot(phull.thepoints_liq[-1][:,0],G_liq/unit,':',color=colors['liquid'])
plt.plot(phull.thepoints_cryst[-1][:,0],G_cryst/unit,'D',color=colors['crystals'])
for i in isel_allc:  plt.plot([simplices['x'][i,0,0],simplices['x'][i,1,0]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],'.-',color=colors['allcryst'],label=labels['allcryst']); labels['allcryst']=None
for i in isel_liq:   plt.plot([simplices['x'][i,0,0],simplices['x'][i,1,0]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],color=colors['liquid'],label=labels['liquid']); labels['liquid']=None
for i in isel_cltie: plt.plot([simplices['x'][i,0,0],simplices['x'][i,1,0]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],'.-',color=colors['tieline_c1l1'],label=labels['tieline_c1l1']); labels['tieline_c1l1']=None
for i in isel_inmis: plt.plot([simplices['x'][i,0,0],simplices['x'][i,1,0]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],'.-',color=colors['tieline_c0l2'],label=labels['tieline_c0l2']); labels['tieline_c0l2']=None
plt.xlabel('x')
plt.ylabel(ylabel)
ytxt = plt.gca().get_ylim()[0] + 3
plt.text(0,ytxt,ph.latexify_chemical_formula(components[1]),ha='center')
plt.text(1,ytxt,ph.latexify_chemical_formula(components[0]),ha='center')
ytxt = plt.gca().get_ylim()[1] - 6
plt.text(0.65,ytxt,f'T = {T:.0f} K')
plt.text(0.65,ytxt-4,f'P = 1 bar')
for icr,row in crystaldb.dbase.iterrows():
    plt.text(row['x'][0],(row['mfDfG']-phull.G_relevelling_plane(row['x']))/unit+2,row['Abbrev'],ha='center',size=7)
plt.legend()
if relevel:
    srel='_rel'
else:
    srel=''
plt.savefig(f'fig_binary_{components[0]}_{components[1]}_x_G_T={T:.0f}'+srel+'.pdf')
plt.show()
