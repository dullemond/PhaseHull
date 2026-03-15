import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root

class Gasparik90(object):
    """
    The model for the Diopside - Enstatite join of Gasparik (1990) American Mineralogist,
    Volume 75, pages 1080-1091, 1990.
    """
    def __init__(self,T=970.,P=1.,mu00func=None,mu00EnPolyM='highclino',mu00DiPolyM='highclino'):
        """
        Arguments:
          T            Temperature in [K]. Default is 970 K, which is the default in table 1 of Gasparik.
          P            Pressure in [bar]. Default is 1.
          mu00func     The function mu00func(T,P) returning two values: mu00_Enstatite, mu00_Diopside
                       in units of J/mol, which the user should give to __init__(). These are the mu0
                       values of enstatite and diopside of the 'highclino' solution (for the default;
                       you can change that with mu00EnPolyM and mu00DiPolyM). All other solutions
                       are offset from the 'highclino' solution. If this function is not given, then by
                       default the mu00 of 'highclino' is considered to be 0. That is alright as long
                       as the Gasparik model is only used in isolation.
          mu00EnPolyM  The polymorph of Enstatite that the mu00func() function gives the mu00 for.
          mu00DiPolyM  The polymorph of Diopside that the mu00func() function gives the mu00 for.
        """
        self.mu00func = mu00func
        self.solnames = ['highclino','ortho','proto','hpclino','lowclino']
        self.incleint = 1.  # Only for debugging. Otherwise keep 1.
        self.mu00EnPolyM = mu00EnPolyM
        self.mu00DiPolyM = mu00DiPolyM
        self.reset(T,P)

    def reset(self,T=970.,P=1.,mu00En=0.,mu00Di=0.):
        """
        Reset the Gasparik model to a new temperature and pressure.
        
        Arguments:
          T            Temperature in [K]. Default is 970 K, which is the default in table 1 of Gasparik.
          P            Pressure in [bar]. Default is 1.
        """
        self.T        = T
        self.P        = P
        if self.mu00func is not None:
            mu00En,mu00Di = self.mu00func(T,P)
        self.mu00En   = mu00En
        self.mu00Di   = mu00Di
        self.mu0 = self.mu0_of_endmembers(self.T,self.P,mu0_highclino_En=self.mu00En,mu0_highclino_Di=self.mu00Di)
        # In case mu00EnPolyM and/or mu00DiPolyM are not 'highclino', correct this
        mu00En0 = self.mu0[self.mu00EnPolyM]['enstatite']
        mu00Di0 = self.mu0[self.mu00DiPolyM]['diopside']
        for poly in self.mu0:
            self.mu0[poly]['enstatite'] += mu00En-mu00En0
            self.mu0[poly]['diopside']  += mu00Di-mu00Di0

    def mu0_of_endmembers(self,T,P,mu0_highclino_En=0.,mu0_highclino_Di=0.):
        """
        From the equations of the reactions, we can compute the mu0 of all the
        enstatite and diopside polymorphs, assuming we know the values of mu0
        for high-clino enstatite and high-clino diopside (because all others
        are offset from those). If the mu0 of high-clino enstatite/diopside
        are unknown, they are assumed 0. For the phase diagram along this
        join that is okay, but if we embed this join into a bigger phase
        diagram, we need the mu0 of high-clino enstatite/diopside.
        """
        DeltaG_O_En,DeltaG_O_Di   = self.deltaG_of_reaction(T,P,'ortho_highclino')
        DeltaG_O_En               = -DeltaG_O_En + mu0_highclino_En
        DeltaG_O_Di               = -DeltaG_O_Di + mu0_highclino_Di
        DeltaG_P_En,DeltaG_P_Di   = self.deltaG_of_reaction(T,P,'proto_highclino')
        DeltaG_P_En               = -DeltaG_P_En + mu0_highclino_En
        DeltaG_P_Di               = -DeltaG_P_Di + mu0_highclino_Di
        DeltaG_HP_En,DeltaG_HP_Di = self.deltaG_of_reaction(T,P,'ortho_hpclino')
        DeltaG_HP_En              = DeltaG_O_En + DeltaG_HP_En
        DeltaG_HP_Di              = DeltaG_O_Di + DeltaG_HP_Di
        DeltaG_LC_En,DeltaG_LC_Di = self.deltaG_of_reaction(T,P,'ortho_lowclino')
        DeltaG_LC_En              = DeltaG_O_En + DeltaG_LC_En
        DeltaG_LC_Di              = DeltaG_O_Di + DeltaG_LC_Di
        DeltaG_HC_En              = mu0_highclino_En
        DeltaG_HC_Di              = mu0_highclino_Di
        mu0                       = {}
        mu0['highclino']          = {'enstatite':DeltaG_HC_En,'diopside':DeltaG_HC_Di}
        mu0['ortho']              = {'enstatite':DeltaG_O_En, 'diopside':DeltaG_O_Di }
        mu0['proto']              = {'enstatite':DeltaG_P_En, 'diopside':DeltaG_P_Di }
        mu0['high-p']             = {'enstatite':DeltaG_HP_En,'diopside':DeltaG_HP_Di}
        mu0['lowclino']           = {'enstatite':DeltaG_LC_En,'diopside':DeltaG_LC_Di}
        return mu0

    def deltaG_of_reaction(self,T,P,solutions):
        """
        Equations of Gasparik for the reactions. See his summary. These reactions
        define the relative energy differences at x=0 and x=1 of the different
        solid solutions. These will be used by the function mu0_of_endmembers()
        to compute the absolute energies mu0 at each end.

        NOTE: Somehow the Gasparik paper does not seem to give the Delta G of
              low-clino Diopside, only low-clino Enstatite. Maybe I overlooked.
              But I couldn't find it. So I set low-clino Diopside to high-clino
              Diopside. All the rest follows then by self-consistency.
        """
        if solutions=='ortho_highclino':
            DeltaG_En           =   3457. - 1.95*T  + 0.038*P + 1.7e-7*P**2
            DeltaG_Di           = -32845. +   12*T  +  0.09*P -  40e-7*P**2
        elif solutions=='proto_highclino':
            DeltaG_En           = -14475. + 33.6*T  - 0.6*T**1.5*(1-633e-8*P) - 0.282*P + 1.7e-7*P**2
            DeltaG_Di           = -11920. - 7*T
        elif solutions=='proto_ortho':
            DeltaG_En           = -17932. + 35.55*T - 0.6*T**1.5*(1-633e-8*P) - 0.32*P
            DeltaG_Di           =  20925. - 19*T    - 0.09*P  +  40e-7*P**2
        elif solutions=='ortho_hpclino':
            DeltaG_En           =   8300. + 6.2*T   - 0.2*P
            DeltaG_Di           = -15000. + 12*T    + 0.29*P  -  40e-7*P**2
        elif solutions=='ortho_lowclino':
            DeltaG_En           =  -1921. + 2.29*T  - 0.011*P +   1e-7*P**2
            DeltaG_Di           = -32845. +   12*T  +  0.09*P -  40e-7*P**2  # Same as 'ortho_highclino' for consistency with assumption LC Diop = HC Diop
        elif solutions=='highclino_lowclino':
            DeltaG_En           =  -5378  + 4.24*T  - 0.049*P - 0.7e-7*P**2
            DeltaG_Di           = 0.    # Assume LC Diop = HC Diop (Gasparik doesn't write DeltaG of LC Diop)
        else:
            raise ValueError(f'Error: Do not know reaction for {solutions}')
        return DeltaG_En,DeltaG_Di

    def rtln_activity_sol(self,T,P,xEn,xDi,solution):
        """
        The R*T*ln(a) of enstatite and diopside for a given solid solution.
        
        Arguments:
          T            Temperature in [K]. Default is 970 K, which is the default in table 1 of Gasparik.
          P            Pressure in [bar]. Default is 1.
          xEn          The x value of enstatite
          xDi          The x value of diopside. xEn+xDi must be 1.
          solution     The name of the solid solution. If 'highclino', then the
                       energy function is slightly different from the other solid solutions.
        """
        Rgas        = 8.314  # J/mol·K
        assert np.all(np.abs(xEn+xDi-1)<1e-5), 'Error: xEn+xDi not equal to 1'
        if solution=='highclino' or solution=='lowclino':
            AG      = 29270.  -0.03*P  # Page 1089, left column
            BG      = -2800.  +0.04*P  # Page 1089, left column
        else:
            AG      = 20000.           # Page 1089
            BG      = 0.
        RTlngEn     = AG*xDi**2 + BG*(4*xDi**3-3*xDi**2) # Page 1081, left column
        RTlngDi     = AG*xEn**2 + BG*(3*xEn**2-4*xEn**3) # Page 1081, left column
        RTlnaEn     = Rgas*T*np.log(xEn+1e-90) + self.incleint*RTlngEn
        RTlnaDi     = Rgas*T*np.log(xDi+1e-90) + self.incleint*RTlngDi
        return RTlnaEn,RTlnaDi

    def equation_for_reaction(self,x,solutions,T,P):
        assert not np.isscalar(x), 'Error: x must be array of 2 values of the two types of enstatite.'
        if type(x) is list: x = np.array(x)
        assert len(x)==2, 'Error: x must be array of 2 values of the two types of enstatite.'
        sols = solutions.split('_')
        # Note: in the stuff below I just for convenience call the x[0] clino and x[1] ortho,
        #       but the actual polymorph depends on the solutions variable.
        xCEn                = x[0]
        xOEn                = x[1]
        DeltaG_En,DeltaG_Di = self.deltaG_of_reaction(T,P,solutions)
        xCDi                = 1-xCEn
        xODi                = 1-xOEn
        # For eqs below, note: reverse order of sols, because in Eqs A to F
        # on page 1080 of the paper for OEn = CEn the eq is RT*(ln(aCEn)-ln(aOEn))
        rtlnaCEn,rtlnaCDi   = self.rtln_activity_sol(T,P,xCEn,xCDi,sols[1])
        rtlnaOEn,rtlnaODi   = self.rtln_activity_sol(T,P,xOEn,xODi,sols[0])
        eqEn                = rtlnaCEn - rtlnaOEn + DeltaG_En
        eqDi                = rtlnaCDi - rtlnaODi + DeltaG_Di
        return np.array([eqEn,eqDi])

    def solve_reactions(self,x0,solutions):
        T    = self.T
        P    = self.P
        if np.isscalar(x0): x0 = np.array([x0])
        assert len(x0)==2
        sols = solutions.split('_')
        eq   = lambda x: self.equation_for_reaction(x,solutions,T,P)
        res  = root(eq,x0)
        return res

    def Gfunc(self,x,isol):
        T    = self.T
        P    = self.P
        assert len(x.shape)==2, 'Error: x must be 2D'
        xEn  = x[:,0]
        xDi  = x[:,1]
        assert np.all(np.abs(xEn+xDi-1)<1e-5), 'Error: x does not add up to 1'
        nx   = x.shape[0]
        G    = np.zeros(nx)

        if isol==0:
            # The G function of high clino
            rtlnaHCEn,rtlnaHCDi     = self.rtln_activity_sol(T,P,xEn,xDi,'highclino')
            G[:]  = xEn*(self.mu0['highclino']['enstatite']+rtlnaHCEn) \
                  + xDi*(self.mu0['highclino']['diopside'] +rtlnaHCDi)
        elif isol==1:
            # The G function of ortho
            rtlnaOEn,rtlnaODi     = self.rtln_activity_sol(T,P,xEn,xDi,'ortho')
            G[:]  = xEn*(self.mu0['ortho']['enstatite']+rtlnaOEn)  \
                  + xDi*(self.mu0['ortho']['diopside'] +rtlnaODi)
        elif isol==2:
            # The G function of proto
            rtlnaPEn,rtlnaPDi     = self.rtln_activity_sol(T,P,xEn,xDi,'proto')
            G[:]  = xEn*(self.mu0['proto']['enstatite']+rtlnaPEn) \
                  + xDi*(self.mu0['proto']['diopside'] +rtlnaPDi)
        elif isol==3:
            # The G function of high-p
            rtlnaHPEn,rtlnaHPDi     = self.rtln_activity_sol(T,P,xEn,xDi,'hpclino')
            G[:]  = xEn*(self.mu0['high-p']['enstatite']+rtlnaHPEn) \
                  + xDi*(self.mu0['high-p']['diopside'] +rtlnaHPDi)
        elif isol==4:
            # The G function of lowclino
            rtlnaLCEn,rtlnaLCDi     = self.rtln_activity_sol(T,P,xEn,xDi,'lowclino')
            G[:]  = xEn*(self.mu0['lowclino']['enstatite']+rtlnaLCEn) \
                  + xDi*(self.mu0['lowclino']['diopside']+rtlnaLCDi)
        else:
            raise ValueError('isol has wrong value')
        return G

if __name__ == "__main__":

    nx     = 101
    x1d    = np.linspace(0,1,nx)
    x      = np.zeros((nx,2))
    x[:,0] = x1d
    x[:,1] = 1-x[:,0]
    T      = 970.
    #T      = 2200.
    #T      = 1600.
    P      = 1.
    q      = Gasparik90(T=T,P=P)

    plt.figure()
    for isol,name in enumerate(q.solnames):
        G = q.Gfunc(x,isol)
        plt.plot(x1d,G/1e3,color=f'C{isol}',label=name)
    plt.legend()
    plt.xlabel(r'x$_{\mathrm{En}}$')
    plt.ylabel(r'$\hat G-G_{\mathrm{base}}$ [kJ/mol]')
    plt.show()
    
    #print(q.solve_reactions([0.1,0.9],'highclino_lowclino'))

    import phasehull as ph
    import pandas as pd
    from mineral_systems import Berman1983

    b           = Berman1983.Berman83(T=T,P=P)

    components  = ['Mg2Si2O6','CaMgSi2O6']  #  ['Enstatite','Diopside']
    compprim    = ['MgSiO3',  'CaMgSi2O6']
    compnames   = ['PrEn',    'Diop'     ]

    sols        = q.solnames

    mdb         = b.mdb.copy()    ##pd.read_fwf('minerals.fwf')
    dbase       = ph.extract_from_mineral_database_based_on_components(mdb,components)

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
    nres        = 100
    for i in range(5):
        solsols.append(ph.SolidSolution(sols[i],components,compprim,Gg[i],dbase,nres=nres))
    
    phull       = ph.PhaseHull(components,solsols=solsols)

    xbase       = np.array([[0.,1.],[1.,0.]])
    Gbase       = np.array([mu00Di,mu00En])
    relevel     = True
    #xbase       = None
    #Gbase       = None
    #relevel     = False
    ax = ph.plot_binary_xG(phull,relevel=relevel,xbase=xbase,Gbase=Gbase)
    ax.set_ylim(ymax=3.)
