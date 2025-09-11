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

#Tc      = 700.
Tc      = 800.
#Tc      = 990.7
#Tc      = 920.7
#Tc      = 850.7
#Tc      = 900.
Tk      = Tc + 273.15
T       = Tk

endmembers = ["Ag","Cu"]

Rgas    = 8.314

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

def Gfunc_l(x):
    if np.isscalar(x[0]):
        xA = np.array([x[0]])
        xB = np.array([x[1]])
    else:
        xA = x[:,0]
        xB = x[:,1]
    return xA*mu_A0_l + xB*mu_B0_l + Rgas*T*( xA*np.log(xA+1e-99) + xB*np.log(xB+1e-99) ) + xA*xB*( a_l + (xA-xB) * b_l + (xA-xB)**2 * c_l )

def Gfunc_s(x):
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

nx          = 300
x1d         = np.linspace(0,1,nx+1)
x           = np.zeros((nx+1,2))
x[:,0]      = x1d
x[:,1]      = 1-x1d
G_l         = Gfunc_l(x)
G_s         = Gfunc_s(x)
G_base      = (1-x1d)*G_l[0] + x1d*G_l[-1]

#plt.figure()
#plt.plot(1-x1d,(G_l-G_base)/1e3,label='liquid')
#plt.plot(1-x1d,(G_s-G_base)/1e3,label='solid')
#plt.xlabel('x')
#plt.ylabel(r'$G-G_{\mathrm{base}}$ [kJ/mol]')
#plt.ylim(-1e1,1e1)
#plt.legend()

# Calling PhaseHull
 
endm       = ['Ag','Cu']
liq        = ph.Liquid('liq',endm,Gfunc_l)
sol        = ph.Liquid('sol',endm,Gfunc_s)

phull      = ph.PhaseHull(endm,liquids=[liq,sol],nres0=300,incl_ptnames=True,T=T,P=1.)

isel_liq   = phull.select_simplices_of_a_given_kind('liquid')
isel_inmis = phull.select_simplices_of_a_given_kind('tieline_c0l2')
isel_crliq = phull.select_simplices_of_a_given_kind('tieline_c0l2_crossliq')

x_liq      = phull.thepoints_liq[-1][:,0]
G_liq      = phull.thepoints_liq[-1][:,-1]
G_liq     -= (1-x_liq)*G_l[0] + x_liq*G_l[-1]

simplices  = phull.thesimplices[-1]

colors     = {'liquid':'deepskyblue','solid':'C3',     'tieline_c0l2':'C9',     'tieline_c0l2_crossliq':'C2'}
labels     = {'liquid':'liquid',     'solid':'solid',  'tieline_c0l2':'inmisc', 'tieline_c0l2_crossliq':'tie line'}
unit       = 1e3

plt.figure()
plt.plot(1-x1d,(G_l-G_base)/1e3,':',label='liquid phase')
plt.plot(1-x1d,(G_s-G_base)/1e3,':',label='solid phase')
for i in isel_liq:
    x_left  = simplices['x'][i,0,0]
    x_right = simplices['x'][i,1,0]
    G_left  = ( simplices['G'][i,0] - ( (1-x_left)* G_l[0] + x_left* G_l[-1] ) ) /unit
    G_right = ( simplices['G'][i,1] - ( (1-x_right)*G_l[0] + x_right*G_l[-1] ) ) /unit
    if simplices['ptnames'][i][0]==-1:
        plt.plot([1-x_left,1-x_right],[G_left,G_right],color=colors['liquid'],label=labels['liquid']); labels['liquid']=None
    else:
        plt.plot([1-x_left,1-x_right],[G_left,G_right],color=colors['solid'],label=labels['solid']); labels['solid']=None
for i in isel_inmis:
    x_left  = simplices['x'][i,0,0]
    x_right = simplices['x'][i,1,0]
    G_left  = ( simplices['G'][i,0] - ( (1-x_left)* G_l[0] + x_left* G_l[-1] ) ) /unit
    G_right = ( simplices['G'][i,1] - ( (1-x_right)*G_l[0] + x_right*G_l[-1] ) ) /unit
    plt.plot([1-x_left,1-x_right],[G_left,G_right],'.-',color=colors['tieline_c0l2'],label=labels['tieline_c0l2']); labels['tieline_c0l2']=None
for i in isel_crliq:
    x_left  = simplices['x'][i,0,0]
    x_right = simplices['x'][i,1,0]
    G_left  = ( simplices['G'][i,0] - ( (1-x_left)* G_l[0] + x_left* G_l[-1] ) ) /unit
    G_right = ( simplices['G'][i,1] - ( (1-x_right)*G_l[0] + x_right*G_l[-1] ) ) /unit
    plt.plot([1-x_left,1-x_right],[G_left,G_right],'.-',color=colors['tieline_c0l2_crossliq'],label=labels['tieline_c0l2_crossliq']); labels['tieline_c0l2_crossliq']=None
plt.xlabel('x')
plt.ylabel(r'$G-G_{\mathrm{base}}$ [kJ/mol]')
ytxt = plt.gca().get_ylim()[0] + 0.3
plt.text(0,ytxt,ph.latexify_chemical_formula(endmembers[0]),ha='center')
plt.text(1,ytxt,ph.latexify_chemical_formula(endmembers[1]),ha='center')
plt.text(0.75,plt.gca().get_ylim()[1] - 0.3,f'T = {Tk:.0f} K ({Tc:.0f} C)')
plt.text(0.75,plt.gca().get_ylim()[0] + 0.3,f'P = 1 bar')
plt.legend()
plt.savefig(f'fig_{endmembers[0]}_{endmembers[1]}_x_G_T={Tk:.0f}.pdf')
plt.show()
