#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                                June 2025
#---------------------------------------------------------------------------

# Tools for book-keeping 1D (binary) phase regions across temperature
# for making x-T phase diagrams.

import numpy as np
import matplotlib.pyplot as plt

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
    if stype=='tieline_c0l2' or stype=='tieline_c0l2_crossliq':
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
    elif stype=='liquid':
        # Special treatment of liquids: glue neighboring liquid bits together
        iliq  = 0
        ii    = 0
        for i in isims:
            x,ptname     = find_leftright_pt_1d(simplices,i)
            if simplices['stype'][i]==stype:
                if iliq==0:
                    # Start a new region
                    iliq         = ptname[0]
                    eu           = {}
                    eu['x']      = [x[0],   x[1]]
                    eu['ptname'] = [ptname[0],ptname[1]]
                    simname      = str(ptname[0])+'_'+str(ii)
                elif iliq<0:
                    # We already had a previous liquid element right before
                    if iliq!=ptname[0]:
                        # This should not happen
                        print('Chancing liquid type without inmiscibility region...')
                        selected[simname] = eu
                        iliq         = int(ptname[0])
                        eu           = {}
                        eu['x']      = [x[0],   x[1]]
                        eu['ptname'] = [ptname[0],ptname[1]]
                        simname      = str(ptname[0])
                        ii          += 1
                    else:
                        # Prolong the region with the new liquid element
                        eu['x'][1]   = x[1]
                else:
                    raise ValueError('Error: Cannot have iliq>0 or nan')
                selected[simname] = eu

            else:
                # End of liquid region
                iliq = 0
                ii  += 1
                del simname
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
    nT = len(Temp)
    rnames = set()
    for ireg,region in enumerate(region_tlist):
        rnames = rnames.union(set(region.keys()))
    region_dict = {}
    
    # First all the regions that are not inmiscible liquids nor liquids
    for rname in rnames:
        if rname[:7]!='inmisc_' and rname[0]!='-':
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
            region['iliq_left']  = reg['ptname'][0]
            region['iliq_right'] = reg['ptname'][1]
            region_dict[rname] = region

    # Now search for, and group, all liquid and inmiscible liquid regions.
    # This is a non-trivial problem
    distcrit  = 0.1
    iregnr    = 0
    region_inmisc_tlist = [[] for _ in range(len(Temp))]
    for it,T in enumerate(Temp):
        for rname in region_tlist[it]:
            if rname[:7]=='inmisc_' or rname[0]=='-':
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
            iliq_left  = reg['ptname'][l]
            iliq_right = reg['ptname'][r]
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
                        if (reg['ptname'][l]==iliq_left) and (reg['ptname'][r]==iliq_right):
                            region['x_left'][it]  = reg['x'][l]
                            region['x_right'][it] = reg['x'][r]
                            del region_inmisc_tlist[it][i]
                            success = True
                            break
                if not success:
                    break
            region['iliq_left']  = iliq_left
            region['iliq_right'] = iliq_right
            mask = np.logical_not(np.isnan(region['x_left']))
            region['T']  = np.hstack((Temp[mask],Temp[mask][::-1],[Temp[mask][0]]))
            region['x']  = np.hstack((region['x_left'][mask],region['x_right'][mask][::-1],region['x_left'][mask][0]))
            region['xcen'] = 0.5 * (region['x'].min()+region['x'].max())
            region['Tcen'] = 0.5 * (region['T'].min()+region['T'].max())
            region_dict[f'region_{iregnr}'] = region
        else:
            break
    return region_dict
