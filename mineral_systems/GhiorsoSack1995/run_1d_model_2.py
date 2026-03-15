import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ternary
import phasehull as ph
from model_GhiorsoSack1995 import GhiorsoSack95

#relevel    = False
relevel    = True

MELTS      = False
#MELTS      = True

#nT         = 200
#nx         = 400
nT         = 100
nx         = 100

#Tmin       = 1200.+273.15
Tmin       = 700.+273.15
Tmax       = 3200.+273.15
T          = np.linspace(Tmin,Tmax,nT)

#components = ["CaSiO3","SiO2"]
components = ["SiO2","Al2O3"]
#components = ["CaSiO3","Al2O3"]
#components = ["Mg2SiO4","SiO2"]

b          = GhiorsoSack95(components,T[0],1.,MELTS=MELTS)
def Gfunc(x):
    return b.Gfunc(x)

crystaldb  = ph.CrystalDatabase(b.mdb,resetfunc=b.reset_crystals)
liquid     = ph.Liquid('magma',components,Gfunc,resetfunc=b.reset_liquid)

phull      = ph.PhaseHull(components,crystaldb,liquid,nres0=nx,nocompute=True,incl_ptnames=True)

def find_leftright_pt_1d(simplices,isim):
    # Make sure to order x points from left to right
    if(simplices['x'][isim][0,0]<simplices['x'][isim][1,0]):
        x       = np.array([simplices['x'][isim][0,0],simplices['x'][isim][1,0]])
        ptnames = [simplices['ptnames'][isim][0],simplices['ptnames'][isim][1]]
    else:
        x       = np.array([simplices['x'][isim][1,0],simplices['x'][isim][0,0]])
        ptnames = [simplices['ptnames'][isim][1],simplices['ptnames'][isim][0]]
    return x,ptnames

def find_stypes_1d(simplices,isims,stype):
    selected = {}
    if stype=='tieline_c0l2':
        # Special treatment for inmiscible liquids
        isimssel = []
        xsel     = []
        for i in isims:
            if simplices['stype'][i]==stype:
                x,ptname     = find_leftright_pt_1d(simplices,i)
                isimssel.append(i)
                xsel.append(0.5*(x[0]+x[1]))
        xsel  = np.array(xsel)
        isort = xsel.argsort()
        for ii in isort:
            i = isimssel[ii]
            x,ptname     = find_leftright_pt_1d(simplices,i)
            simname      = 'inmisc_'+str(ii)
            eu           = {}
            eu['x']      = [x[0],   x[1]]
            eu['xmid']   = 0.5*(x[0]+x[1])
            eu['ptname'] = [ptname[0],ptname[1]]
            selected[simname] = eu
    else:
        # Normal case
        for i in isims:
            if simplices['stype'][i]==stype:
                x,ptname     = find_leftright_pt_1d(simplices,i)
                simname      = str(ptname[0])+'+'+str(ptname[1])
                eu           = {}
                eu['x']      = [x[0],   x[1]]
                eu['ptname'] = [ptname[0],ptname[1]]
                selected[simname] = eu
    return selected

def find_all_stypes_in_series_of_phase_diagrams(sim_list,noliq=True):
    stypes = set()
    for simplices in sim_list:
        stypes = stypes.union(set(simplices['stype']))
    if noliq:
        stypes.remove('liquid')
    return stypes

def collect_x_T_phase_regions_1d(region_tlist,Temp,Tmargin=False,simplify_square=True):
    assert len(region_tlist)==len(Temp), 'Error: Regions list not same length as temperature list'
    rnames = set()
    for ireg,region in enumerate(region_tlist):
        rnames = rnames.union(set(region.keys()))
    region_dict = {}
    
    # First all the regions that are not inmiscible liquids
    for rname in rnames:
        if rname[:7]!='inmisc_':
            region = {'x_left':np.zeros(nT),'x_right':np.zeros(nT)}
            for it,T in enumerate(Temp):
                if rname in region_tlist[it]:
                    reg = region_tlist[it][rname]
                    if(reg['x'][0]<reg['x'][1]):
                        l  = 0
                        r  = 1
                    else:
                        l  = 1
                        r  = 0
                    region['x_left'][it]  = reg['x'][l]
                    region['x_right'][it] = reg['x'][r]
                else:
                    region['x_left'][it]  = np.nan
                    region['x_right'][it] = np.nan
            mask = np.logical_not(np.isnan(region['x_left']))
            region['T']  = np.hstack((Temp[mask],Temp[mask][::-1],[Temp[mask][0]]))
            region['x']  = np.hstack((region['x_left'][mask],region['x_right'][mask][::-1],region['x_left'][mask][0]))
            if simplify_square:
                if len(set(region['x']))==2:
                    TT   = region['T']
                    Tup  = TT.max()
                    Tlo  = TT.min()
                    TTn  = np.array([Tlo,Tup,Tup,Tlo,Tlo])
                    xl   = region['x'].min()
                    xr   = region['x'].max()
                    xn   = np.array([xl,xl,xr,xr,xl])
                    region['T'] = TTn
                    region['x'] = xn
            if Tmargin:
                # For nicer plotting results, we can vertically expand half a Delta T to the top and the bottom
                DT   = 0.5*np.abs(Temp[1]-Temp[0])   # Temperature grid must be uniform
                TT   = region['T']
                Tup  = TT.max()
                Tlo  = TT.min()
                if(Tup>Tlo):
                    Tupn = Tup+DT
                    Tlon = Tlo-DT
                    fact = (Tupn-Tlon)/(Tup-Tlo)
                    TTn  = (TT-Tlo)*fact+Tlon
                    region['T'] = TTn
                else:
                    Tupn = TT[0]+DT
                    Tlon = TT[0]-DT
                    TTn  = np.array([Tlon,Tupn,Tupn,Tlon,Tlon])
                    xl   = region['x'].min()
                    xr   = region['x'].max()
                    xn   = np.array([xl,xl,xr,xr,xl])
                    region['T'] = TTn
                    region['x'] = xn
            region['xcen'] = 0.5 * (region['x'].min()+region['x'].max())
            region['Tcen'] = 0.5 * (region['T'].min()+region['T'].max())
            region_dict[rname] = region

    # Now search for, and group, all inmiscible liquid regions.
    # This is a non-trivial problem
    distcrit  = 0.1
    iregnr    = 0
    #rnames_inmisc = []
    #for rname in rnames:
    #    if rname[:7]=='inmisc_':
    #        rnames_inmisc.append(rname)
    #rnames_inmisc.sort()
    region_inmisc_tlist = [[] for _ in range(len(Temp))]
    for it,T in enumerate(Temp):
        for rname in region_tlist[it]:
            if rname[:7]=='inmisc_':
                reg  = region_tlist[it][rname]
                region_inmisc_tlist[it].append(reg)
    nregnrmax = 30  # I expect no more than this nr of such regions
    for iregnr in range(nregnrmax):
        itstart = -1
        region  = {'x_left':np.zeros(nT),'x_right':np.zeros(nT)}
        region['x_left'][:]  = np.nan
        region['x_right'][:] = np.nan
        for it in range(len(Temp)):
            if(len(region_inmisc_tlist[it])>0):
                itstart = it
                break
        if itstart>=0:
            reg = region_inmisc_tlist[itstart].pop()
            if(reg['x'][0]<reg['x'][1]):
                l  = 0
                r  = 1
            else:
                l  = 1
                r  = 0
            region['x_left'][it]  = reg['x'][l]
            region['x_right'][it] = reg['x'][r]
            for it in range(itstart+1,len(Temp)):
                success     = False
                xmid_prev   = 0.5 * ( region['x_left'][it-1] + region['x_right'][it-1] )
                xleft_prev  = region['x_left'][it-1]
                xright_prev = region['x_right'][it-1]
                for i,reg in enumerate(region_inmisc_tlist[it]):
                    xmid_curr   = 0.5 * ( reg['x'][0] + reg['x'][1] )
                    xleft_curr  = reg['x'][0]
                    xright_curr = reg['x'][1]
                    if(((xmid_curr<=xright_prev) and (xmid_curr>=xleft_prev)) or ((xmid_prev<=xright_curr) and (xmid_prev<=xleft_curr))):
                        if(reg['x'][0]<reg['x'][1]):
                            l  = 0
                            r  = 1
                        else:
                            l  = 1
                            r  = 0
                        region['x_left'][it]  = reg['x'][l]
                        region['x_right'][it] = reg['x'][r]
                        del region_inmisc_tlist[it][i]
                        success = True
                        break
                if not success:
                    break
            mask = np.logical_not(np.isnan(region['x_left']))
            region['T']  = np.hstack((Temp[mask],Temp[mask][::-1],[Temp[mask][0]]))
            region['x']  = np.hstack((region['x_left'][mask],region['x_right'][mask][::-1],region['x_left'][mask][0]))
            region['xcen'] = 0.5 * (region['x'].min()+region['x'].max())
            region['Tcen'] = 0.5 * (region['T'].min()+region['T'].max())
            region_dict[f'inmiscregion_{iregnr}'] = region
        else:
            break
    return region_dict

sim_list   = []
ism_list   = []

from tqdm import tqdm
for iT in tqdm(range(nT)):
    #print(f'iT = {iT}')
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

colors     = {'allcryst':'C1','liquid':'deepskyblue','tieline_c1l1':'C4','crystals':'C3','tieline_c0l2':'C9'}
fillcols   = {'allcryst':'peachpuff','liquid':'lightcyan','tieline_c1l1':'thistle','tieline_c0l2':'paleturquoise'}

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
        plt.fill(region['x'],region['T']-273.15,color=fillcols[st]);  plt.plot(region['x'],region['T']-273.15,color='black',linewidth=1)
        plt.text(region['xcen'],region['Tcen']-273.15-10,text,ha='center',va='center',size=size)
plt.ylim(Tmin-273.15,Tmax-273.15)
plt.xlabel('x (molar)')
plt.ylabel('T [Celsius]')
ytxt = plt.gca().get_ylim()[0] - 200
plt.text(0,ytxt,ph.latexify_chemical_formula(components[1]),ha='center')
plt.text(1,ytxt,ph.latexify_chemical_formula(components[0]),ha='center')
ytxt = plt.gca().get_ylim()[1] - 200
plt.text(0.05,ytxt,f'P = 1 bar')
ytxt = plt.gca().get_ylim()[1] - 800
plt.text(0.35,ytxt,f'Liquid',size=6)
if MELTS:
    plt.title('MELTS-like')
else:
    plt.title('GS95')
plt.show()
