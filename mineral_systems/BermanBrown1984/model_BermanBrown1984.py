#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                                June 2025
#
# The Berman & Brown 1984 model.
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 1000)
import os
from phasehull import dissect_molecule,dissect_oxide,identify_endmember_minerals,Margules

class BermanBrown84(object):
    def __init__(self,endmselect,T,P=1,path=None,ext=''):
        if path is None: path = os.path.dirname(__file__)
        self.path       = path
        self.Rgas       = 8.314  # J/mol·K
        self.T          = T
        self.P          = P
        self.endmembers = ["SiO2","Al2O3","CaO"]
        self.endmselect = endmselect
        assert set(endmselect)<=set(self.endmembers), 'Error: The requested endmembers are not all in this model.'
        self.mdb_orig   = pd.read_fwf(os.path.join(path,'minerals'+ext+'.fwf'))
        self.ldb_orig   = pd.read_fwf(os.path.join(path,'liquids'+ext+'.fwf'))
        self.mdb        = self.extract_from_mineral_database_based_on_endmembers(self.mdb_orig,endmselect)
        self.ldb        = self.extract_from_mineral_database_based_on_endmembers(self.ldb_orig,endmselect)
        self.sort_liquid_endmembers()
        WH,WS           = self.get_Margules()
        self.margules   = Margules(self.endmembers)
        self.margules.load_w(WH,WS)
        self.margules.reduce_margules_to_subset(endmselect)
        self.reset(T,P)

    def reset(self,T,P=1):
        self.T          = T
        self.P          = P
        self.reset_crystals(T,P)
        self.reset_liquid(T,P)

    def reset_crystals(self,T,P=1):
        self.T          = T
        self.P          = P
        self.compute_DfG_with_mole_fraction_weighting(self.mdb,T)

    def reset_liquid(self,T,P=1):
        self.T          = T
        self.P          = P
        #self.margules.compute_w_gibbs(T,P)
        self.compute_DfG_with_mole_fraction_weighting(self.ldb,T)

    def Gfunc(self,x):
        if len(x.shape)==1:
            x = np.array([x,])
        assert x.shape[-1]==len(self.endmselect), 'Error: Dimension of x incorrect.'
        G = self.compute_G_of_liquid_mixture(self.ldb,self.T,x,self.endmselect)
        return G

    def get_Margules(self):
        """
        Berman & Brown 1983, table 5.
        Code inspired by code by D. Ebel 2001.
        https://research.amnh.org/~debel/vapors1/codes/B83a/B83acts1.txt
        """
        # 0 = Si
        # 1 = Al
        # 2 = Ca
        ncmp = 3
        WH   = np.zeros((ncmp,ncmp,ncmp,ncmp))
        WS   = np.zeros((ncmp,ncmp,ncmp,ncmp))
        
        # Si-Al System
        WH[0][1][1][1] =     63617.160
        WH[0][0][1][1] =   1642663.510
        WH[0][0][0][1] =   -106635.220
        WS[0][1][1][1] =        23.740
        WS[0][0][1][1] =       763.870
        WS[0][0][0][1] =       -28.130
        
        # Si-Ca System
        WH[0][2][2][2] =   -898692.640
        WH[0][0][2][2] =   -350208.150
        WH[0][0][0][2] =    -14081.800
        WS[0][2][2][2] =      -240.770
        WS[0][0][2][2] =        48.620
        WS[0][0][0][2] =        35.490
    
        # Al-Ca System
        WH[1][2][2][2] =   -455634.210
        WH[1][1][2][2] =   -725166.290
        WH[1][1][1][2] =   -240214.840
        WS[1][2][2][2] =        -2.470
        WS[1][1][2][2] =      -255.390
        WS[1][1][1][2] =       -26.700
    
        # Si-Al-Ca System
        WH[0][0][1][2] =  -2847911.220
        WH[0][1][1][2] =  -2149042.640
        WH[0][1][2][2] =    209108.930
        WS[0][0][1][2] =     -1046.350
        WS[0][1][1][2] =      -641.840
        WS[0][1][2][2] =       313.360
    
        return WH, WS

    def compute_G_of_liquid_mixture(self,ldb,T,x,endmembers,nomixG=False,
                                    incl_linear=True,incl_ideal=True,
                                    incl_nonideal=True):
        """
        Compute the full G(x,T) of a mixture of liquids, including the
        Gibbs of formation, the ideal Gibbs of mixing and the non-ideal Gibbs
        of mixing.
    
        Arguments:
    
          ldb          The database of liquids
    
          T            Temperature in Kelvin
    
          x            The mole (!) fractions. Must be array of shape [nx,N]
                       where nx is the number of points of x, and N is the
                       number of endmembers.
    
          endmembers   The endmembers (formulae, e.g. ['SiO2','Al2O3','MgO'])
    
        Options:
    
          nomixG       If True, then only return the unmixed mean G.
    
        Options for testing purposes:

          incl_linear    (default: True) Include the linear combination term
          incl_ideal     (default: True) Include the ideal mixing (entropy) term
          incl_nonideal  (default: True) Include the nonideal mixing (interaction) term
    
        """
        if T!=self.T: self.reset_liquid(T)
        if type(x) is list: x = np.array(x)
        N          = x.shape[-1]
        nx         = x.shape[0]
        assert N==len(endmembers), 'Error: Nr of endmembers and x components not equal'
        iendmembers,DfGendmembers = identify_endmember_minerals(ldb,endmembers)
    
        G  = np.zeros(nx)

        # First the linear combination of the N endmembers
        if incl_linear:
            for i in range(N):
                G += x[:,i]*DfGendmembers[i]

        # Then the mixing
        if not nomixG:
            if incl_ideal:    G += self.margules.compute_ideal_mixing_G(x,T)
            if incl_nonideal: G += self.margules.compute_interaction_G(x,T)

        return G
    
    def get_Cp(self,mdb,mineral,T,perafu=False):
        mn = mdb[mdb['Abbrev']==mineral].iloc[0]
        cp = mn['a'] + mn['c']/T**2 + mn['f']/np.sqrt(T) + mn['h']/T  # Joules/mole (mole of formula unit)
        if perafu:  # Joules per atoms-per-formula-unit (= Joules per atom)
            mol,mass,ch=dissect_molecule(mn['Formula'])
            n=0
            for m in mol:
                n+=mol[m]
            cp /= n
        return cp
    
    def get_int_Cp_dT(self,mdb,mineral,T,perafu=False):
        """
        The integral_{298.15}^T c_P(T) dT
        """
        mn    = mdb[mdb['Abbrev']==mineral].iloc[0]
        T1    = 298.15
        intcp = mn['a']*(T-T1) - mn['c']*(1/T-1/T1) + 2*mn['f']*(np.sqrt(T)-np.sqrt(T1)) + mn['h']*(np.log(T)-np.log(T1))  # Joules*K/mole (mole of formula unit)
        if perafu:  # Joules*K per atoms-per-formula-unit (= Joules*K per atom)
            mol,mass,ch=dissect_molecule(mn['Formula'])
            n=0
            for m in mol:
                n+=mol[m]
            intcp /= n
        return intcp
        
    def get_int_CpdivT_dT(self,mdb,mineral,T,perafu=False):
        """
        The integral_{298.15}^T (c_P(T)/T) dT
        """
        mn    = mdb[mdb['Abbrev']==mineral].iloc[0]
        T1    = 298.15
        intcp = mn['a']*(np.log(T)-np.log(T1)) - 0.5*mn['c']*(1/T**2-1/T1**2) - 2*mn['f']*(1/np.sqrt(T)-1/np.sqrt(T1)) - mn['h']*(1/T-1/T1)  # Joules*K/mole (mole of formula unit)
        if perafu:  # Joules*K per atoms-per-formula-unit (= Joules*K per atom)
            mol,mass,ch=dissect_molecule(mn['Formula'])
            n=0
            for m in mol:
                n+=mol[m]
            intcp /= n
        return intcp
        
    def get_mu0_at_T(self,mdb,mineral,T):
        """
        The mu_0 for this mineral at temperature T. Eq. 28 of Berman & Brown 1984.
        Note that because this is for the pure (!) substance, the meaning of mu_0
        is the same as of Delta G_f_0 (which is the formation gibbs energy per mole).
        This is because adding moles of material makes the total Gibbs energy increase
        linearly:
    
          Delta G_f(N) = Delta G_f_0 * N
    
        so that
    
                     dDelta G_f(N)
          mu_0 =def= ------------- = Delta G_f_0
                         dN
    
        Arguments:
    
          mineral          The abbreviated name of the mineral (column Abbrev in mdb)
          T                Temperature in [K]
    
        Returns:
    
          mu0              The mu_0 == Delta G_f_0 of the mineral [J/mole]
          
        """
        mn      = mdb[mdb['Abbrev']==mineral].iloc[0]
        intCp   = self.get_int_Cp_dT(mdb,mineral,T)
        intCpT  = self.get_int_CpdivT_dT(mdb,mineral,T)
        DHf0    = mn['Enthalpy']*1e3  # Note: 1e3 because it is given as kiloJoule/mol
        S0      = mn['Entropy']
        mu0     = DHf0 + intCp - T * ( S0 + intCpT )
        return mu0

    def extract_from_mineral_database_based_on_endmembers(self,mdb,endmembers):
        """
        Given a list of minerals in Pandas dataframe mdb (see read_minerals_and_liquids()), select only
        those minerals that are composed of the endmembers given in the list endmembers. Also add
        columns of x and moles.
    
        Arguments:
    
          mdb              The mineral database (see read_minerals_and_liquids())
          endmembers       List of the formulae of the endmembers, e.g. ['SiO2','MgO','Al2O3'].
    
        Returns:
    
          select           A version of mdb with only the minerals that can be created
                           from the endmembers, and a column with the x and moles values.
                           The x are the mole fractions. The moles are the nr of moles
                           of that mineral that can be made from 1 mole of endmembers.
                           Example: with 0.333 mole of SiO2 and 0.667 mole of MgO (in
                           total 1 mole worth of endmembers) you can create 0.333 mole of
                           Mg2SiO4.
        """
        nm     = len(mdb)
        nem    = len(endmembers)
        select = mdb.copy()
        select['ok']    = False
        select['x']     = np.zeros((nm,nem)).tolist()
        select['moles'] = 0.
        for i,mn in select.iterrows():
            d = dissect_oxide(mn['Formula'],endmembers=endmembers)
            if d['complete']:
                select.at[i,'ok']     = True
                select.at[i,'x']      = d['x']
                select.at[i,'moles']  = d['moles']
        select = select[select['ok']].copy().reset_index(drop=True).drop('ok',axis=1)
        return select

    def sort_liquid_endmembers(self):
        """
        In principle it should not be necessary to sort the Pandas database for the liquids
        (self.ldb), but depending on how external applications use it, it might lead to
        confusion if the order of the endmember liquids is different from the ones given
        in self.endmselect. So just to be on the safe side, we will order them here.
        """
        ldb = self.ldb.reset_index(drop=True).set_index('Formula')
        ldb['idxendm'] = -1
        for iend,endm in enumerate(self.endmselect):
            ldb.at[endm,'idxendm'] = iend
        self.ldb = ldb.sort_index().reset_index()

    def compute_DfG_with_mole_fraction_weighting(self,mdb,T,no_mfDfG=False):
        """
        After having removed all minerals from the mdb database that are not part of the
        endmember system with extract_from_mineral_database_based_on_endmembers(mdb,endmembers),
        and (with the same function) computed the mole fractions x, we can now compute the
        Gibbs free energies for all remaining minerals.
    
        Each pure substance has a Delta_f G(T,p), which is the Gibbs free energy [J/mole] required
        to create the substance out of its standard state constituents (usually, but not necessarily,
        the elements in their atomic form) at the given temperature T [K] and pressure p [bar].
        For simplicity we write Delta_f G as DfG.
    
        Since we are concerned with processes happening in "solar nebular conditions" (meaning the
        densities and pressures in the protoplanetary disk), where the pressure << 1 bar, and
        given that for most minerals the difference in equilibrium is negligible between 0 and
        1 bar, we (for now) omit the p-dependency, and take p=1bar as standard value.
    
        The mass-fraction-weighted version of DfG means, e.g., that with 0.333 mole of SiO2 and
        0.667 mole of MgO (in total 1 mole worth of endmembers) you can create 0.333 mole of
        Mg2SiO4. So mfDfG=0.333*DfG for Mg2SiO4 where DfG is the Delta_f G for 1 mole of Mg2SiO4.
    
        Note: mdb must have a column 'moles' (how many moles of that mineral can we create from
              1 mole total of endmembers). It is easiest to use the function
              extract_from_mineral_database_based_on_endmembers(mdb,endmembers) to automatically
              add this column. If you do not want to compute mfDfG (the Delta_f G for mdb['mole']
              amounts of moles of mineral), you get set no_mfDfG=True
    
        Arguments:
    
          mdb              The mineral database (see read_minerals_and_liquids())
          T                The temperature in [K]
    
        Returns:
    
          modifies the mdb database in-place.
        
        """
        if T!=self.T: self.reset(T)
        mdb['DfG']   = 1e90   # The DfG per mole of this substance
        if not no_mfDfG:
            mdb['mfDfG'] = 1e90   # The DfG per mole of the constituent endmembers
        for i,row in mdb.iterrows():
            DfG                = self.get_mu0_at_T(mdb,row['Abbrev'],T)
            mdb.at[i,'DfG']    = DfG
            if not no_mfDfG:
                mdb.at[i,'mfDfG']  = DfG * row['moles']
