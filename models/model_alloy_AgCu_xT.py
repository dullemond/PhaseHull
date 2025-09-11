# Model of binary alloy of Ag and Cu
# of Chu, Qin, Xiao, Shen, Su, Hu and Tang (2020)
# "Thermodynamic reassessment of the Ag–Cu phase diagram at nano-scale"
# CALPHAD: Computer Coupling of Phase Diagrams and Thermochemistry 72 (2021) 102233
# https://doi.org/10.1016/j.calphad.2020.102233
#
# Here we only use the macroscopic part (assuming 1/r=0).
#
# A model like this was described in a pedagogical way by Prof. Dr. M. Eschrig
# of the University of Greifswald in the following lecture notes:
#
# https://physik.uni-greifswald.de/storages/uni-greifswald/fakultaet/mnf/physik/ag_eschrig/teaching/thermostat/Termodynamik2019_1.pdf
#
import numpy as np
import mpltern
import phasehull as ph
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------
#                               Setup the problem 
#----------------------------------------------------------------------------------

endmembers = ["Ag","Cu"]

Rgas    = 8.314

def reset_liquids(T,P=1):
    #model   = 1
    #model   = 2
    model   = 10
    
    c_s = 0.
    c_l = 0.
    if model==0:
        mu_A0_l = 0.
        mu_B0_l = 0.
        mu_A0_s = -59725 + 48.35*T
        mu_B0_s = -75270 + 48.10*T
        a_s     = 0.
        b_s     = 0.
        a_l     = 0.
        b_l     = 0.
    elif model==1:
        mu_A0_l = 0.
        mu_B0_l = 0.
        #mu_A0_s = -59725 + 48.35*T # These are wrong. Error in the PDF of Eschrig.
        #mu_B0_s = -75270 + 48.10*T # These are wrong. Error in the PDF of Eschrig.
        mu_A0_s = -11025.293 + 8.890146*T  # Constructed from the model of Chu et al. (see model 10 below)
        mu_B0_s = -12964.84  + 9.510243*T  # Constructed from the model of Chu et al. (see model 10 below)
        a_s     = 20719.2 - 5.5068*T
        b_s     = -3597.6 + 1.0350*T
        a_l     =  9102.6 - 1.5222*T
        b_l     =   -1455 + 0.5676*T
        # Something must be wrong as the results do not look like the figures of the lecture of Eschrig
    elif model==2:
        mu_A0_l = 0.
        mu_B0_l = 0.
        #mu_A0_s = -59725 + 48.35*T # These are wrong. Error in the PDF of Eschrig.
        #mu_B0_s = -75270 + 48.10*T # These are wrong. Error in the PDF of Eschrig.
        mu_A0_s = -11025.293 + 8.890146*T  # Constructed from the model of Chu et al. (see model 10 below)
        mu_B0_s = -12964.84  + 9.510243*T  # Constructed from the model of Chu et al. (see model 10 below)
        a_s     = 34532  - 9.67*T
        b_s     = -5996  + 1.725*T
        a_l     = 15171  - 2.537*T
        b_l     =  -2425 + 0.946*T
    elif model==10:
        # Model of Chu et al. CALPHAD: Computer Coupling of Phase Diagrams and Thermochemistry 72 (2021) 102233
        # 
        # Ag for T<1235K
        mu_A0_s = -7209.512 + 118.200733*T -  23.84633*T*np.log(T) - 0.001790585*T**2 - 3.98587e-7*T**3 - 12011/T
        mu_A0_l =  3815.781 + 109.310587*T -  23.84633*T*np.log(T) - 0.001790585*T**2 - 3.98587e-7*T**3 - 12011/T -  1.0322e-20*T**7
        # Cu for T<1358K
        mu_B0_s = -7770.458 + 130.485403*T - 24.112392*T*np.log(T) -  0.00265684*T**2 + 1.29223e-7*T**3 + 52478/T
        mu_B0_l =  5194.382 +  120.97516*T - 24.112392*T*np.log(T) -  0.00265684*T**2 + 1.29223e-7*T**3 + 52478/T - 5.83932e-21*T**7
        #
        a_l     = 17534.6 - 4.45479*T
        b_l     = 2251.3  -  2.6733*T
        c_l     =  492.7
        a_s     = 33819.1 -  8.1236*T
        b_s     = -5601.9 + 1.32997*T
        c_s     = 0.
    else:
        raise ValueError("Unknown model")
    answer = {}
    answer['mu_A0_s'] = mu_A0_s
    answer['mu_A0_l'] = mu_A0_l
    answer['mu_B0_s'] = mu_B0_s
    answer['mu_B0_l'] = mu_B0_l
    answer['a_l']     = a_l
    answer['b_l']     = b_l
    answer['c_l']     = c_l
    answer['a_s']     = a_s
    answer['b_s']     = b_s
    answer['c_s']     = c_s
    return answer

def Gfunc_l(x,**kwargs):
    mu_A0_l = kwargs['mu_A0_l']
    mu_B0_l = kwargs['mu_B0_l']
    a_l     = kwargs['a_l']
    b_l     = kwargs['b_l']
    c_l     = kwargs['c_l']
    if np.isscalar(x[0]):
        xA = np.array([x[0]])
        xB = np.array([x[1]])
    else:
        xA = x[:,0]
        xB = x[:,1]
    return xA*mu_A0_l + xB*mu_B0_l + Rgas*T*( xA*np.log(xA+1e-99) + xB*np.log(xB+1e-99) ) + xA*xB*( a_l + (xA-xB) * b_l + (xA-xB)**2 * c_l )

def Gfunc_s(x,**kwargs):
    mu_A0_s = kwargs['mu_A0_s']
    mu_B0_s = kwargs['mu_B0_s']
    a_s     = kwargs['a_s']
    b_s     = kwargs['b_s']
    c_s     = kwargs['c_s']
    if np.isscalar(x[0]):
        xA = np.array([x[0]])
        xB = np.array([x[1]])
    else:
        xA = x[:,0]
        xB = x[:,1]
    return xA*mu_A0_s + xB*mu_B0_s + Rgas*T*( xA*np.log(xA+1e-99) + xB*np.log(xB+1e-99) ) + xA*xB*( a_s + (xA-xB) * b_s + (xA-xB)**2 * c_s )

#----------------------------------------------------------------------------------
#                         Now start the PhaseHull part
#----------------------------------------------------------------------------------


#nT         = 300
#nx         = 300
nT         = 100
nx         = 100

Tmin       = 600.+273.15
Tmax       = 1100.+273.15
Tgrid      = np.linspace(Tmin,Tmax,nT)
 
endm       = ['Ag','Cu']
liq        = ph.Liquid('liq',endm,Gfunc_l,resetfunc=reset_liquids)
sol        = ph.Liquid('sol',endm,Gfunc_s,resetfunc=reset_liquids)

phull      = ph.PhaseHull(endm,liquids=[liq,sol],nres0=nx,incl_ptnames=True,nocompute=True)

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
                        #print('Changing liquid type without inmiscibility region...')
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

sim_list   = []
ism_list   = []

for iT in range(nT):
    print(f'iT = {iT}')
    T         = Tgrid[iT]
    phull.reset(T)
    simplices = phull.thesimplices[-1]
    isims     = np.argsort(simplices['x'][:,:,0].min(axis=-1))
    sim_list.append(simplices)
    ism_list.append(isims)

stypes     = list(find_all_stypes_in_series_of_phase_diagrams(sim_list,noliq=False))
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
    region_dict[st] = collect_x_T_phase_regions_1d(stype_dict[st],Tgrid,Tmargin=True)

plt.figure()
for st in stypes:
    for name in region_dict[st]:
        region = region_dict[st][name]
        size   = 6
        text   = name.replace('-1','Liq')
        #color  = 'C9'
        color  = None
        if region['iliq_left']==-1 and region['iliq_right']==-2:
            color = '#90e4c1'
        if region['iliq_left']==-2 and region['iliq_right']==-1:
            color = '#90e4c1'
        if region['iliq_left']==-2 and region['iliq_right']==-2:
            x_left  = region['x_left']
            x_left  = x_left[np.logical_not(np.isnan(x_left))]
            x_left  = x_left.mean()
            x_right = region['x_right']
            x_right = x_right[np.logical_not(np.isnan(x_right))]
            x_right = x_right.mean()
            if x_right==1:
                color = '#E0A0A0'
            elif x_left==0:
                color = '#A0A0E0'
            else:
                color = 'peachpuff'
        if color is not None:
            plt.fill(1-region['x'],region['T']-273.15,color=color);  plt.plot(1-region['x'],region['T']-273.15,color='black',linewidth=1)
        #plt.text(1-region['xcen'],region['Tcen']-273.15-10,text,ha='center',va='center',size=size)
plt.ylim(Tgrid[0]-273.15,Tgrid[-1]-273.15)
plt.xlabel('x (molar)')
plt.ylabel('T [Celsius]')
ytxt = plt.gca().get_ylim()[0] + 20
plt.text(0,ytxt,ph.latexify_chemical_formula(endmembers[0]),ha='center')
plt.text(1,ytxt,ph.latexify_chemical_formula(endmembers[1]),ha='center')
ytxt = plt.gca().get_ylim()[1] - 50
plt.text(0.05,ytxt,f'P = 1 bar')
ytxt = plt.gca().get_ylim()[1] - 150
plt.text(0.35,ytxt,f'Liquid',size=6)
plt.savefig(f'fig_{endmembers[0]}_{endmembers[1]}_xT.pdf')
plt.show()
