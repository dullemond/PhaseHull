#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                                June 2025
#
# The Berman 1983 PhD thesis quaternary model.
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 1000)
import os
from phasehull import dissect_molecule,dissect_oxide,identify_endmember_minerals,Margules

class Berman83(object):
    def __init__(self,endmselect,T,P=1,path=None,ext=''):
        if path is None: path = os.path.dirname(__file__)
        self.path       = path
        self.Rgas       = 8.314  # J/mol·K
        self.T          = T
        self.P          = P
        self.endmembers = ["CaO","MgO","Al2O3","SiO2"]
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
        self.compute_DfG_with_mole_fraction_weighting(self.ldb,T)

    def Gfunc(self,x):
        if len(x.shape)==1:
            x = np.array([x,])
        assert x.shape[-1]==len(self.endmselect), 'Error: Dimension of x incorrect.'
        G = self.compute_G_of_liquid_mixture(self.ldb,self.T,x,self.endmselect)
        return G

    def get_Margules(self):
        """
        Berman 1983 PhD thesis, table XI.
        Code inspired by code by D. Ebel 2001.
        https://research.amnh.org/~debel/vapors1/codes/B83a/B83acts1.txt
        """
        # 0 = Ca
        # 1 = Mg
        # 2 = Al
        # 3 = Si
        ncmp = 4
        WH   = np.zeros((ncmp,ncmp,ncmp,ncmp))
        WS   = np.zeros((ncmp,ncmp,ncmp,ncmp))
        
        # Ca-Mg Binary System
        WH[0][0][0][1]=    318144.87       # for  C  C  C  M
        WH[0][0][1][1]=    590616.72       # for  C  C  M  M
        WH[0][1][1][1]=    114076.93       # for  C  M  M  M
        WS[0][0][0][1]=       154.79       # for  C  C  C  M
        WS[0][0][1][1]=       322.37       # for  C  C  M  M
        WS[0][1][1][1]=        58.66       # for  C  M  M  M

        # Ca-Al Binary System
        WH[0][0][0][2]=   -617537.61       # for  A  C  C  C
        WH[0][0][2][2]=   -734020.10       # for  A  A  C  C
        WH[0][2][2][2]=   -197743.47       # for  A  A  A  C
        WS[0][0][0][2]=       -76.83       # for  A  C  C  C
        WS[0][0][2][2]=      -251.94       # for  A  A  C  C
        WS[0][2][2][2]=        -1.11       # for  A  A  A  C

        # Ca-Si Binary System
        WH[0][0][0][3]=   -960867.88       # for  S  C  C  C
        WH[0][0][3][3]=   -341962.81       # for  S  S  C  C
        WH[0][3][3][3]=    -25525.64       # for  S  S  S  C
        WS[0][0][0][3]=      -247.11       # for  S  C  C  C
        WS[0][0][3][3]=        56.62       # for  S  S  C  C
        WS[0][3][3][3]=        34.19       # for  S  S  S  C

        # Mg-Al Binary System
        WH[1][1][1][2]=   -691193.17       # for  A  M  M  M
        WH[1][1][2][2]=    727706.30       # for  A  A  M  M
        WH[1][2][2][2]=   -641890.87       # for  A  A  A  M
        WS[1][1][1][2]=      -227.94       # for  A  M  M  M
        WS[1][1][2][2]=       290.88       # for  A  A  M  M
        WS[1][2][2][2]=      -224.47       # for  A  A  A  M

        # Mg-Si Binary System
        WH[1][1][1][3]=   -610906.87       # for  S  M  M  M
        WH[1][1][3][3]=   -270581.70       # for  S  S  M  M
        WH[1][3][3][3]=     94145.91       # for  S  S  S  M
        WS[1][1][1][3]=      -195.01       # for  S  M  M  M
        WS[1][1][3][3]=       -59.98       # for  S  S  M  M
        WS[1][3][3][3]=        52.77       # for  S  S  S  M

        # Al-Si Binary System
        WH[2][2][2][3]=    258911.07       # for  S  A  A  A
        WH[2][2][3][3]=   1803871.61       # for  S  S  A  A
        WH[2][3][3][3]=   -161039.81       # for  S  S  S  A
        WS[2][2][2][3]=       110.47       # for  S  A  A  A
        WS[2][2][3][3]=       844.79       # for  S  S  A  A
        WS[2][3][3][3]=       -60.41       # for  S  S  S  A
    
        # Ca-Mg-Al Ternary System
        WH[0][0][1][2]=  -2440837.52       # for  A  C  C  M
        WH[0][1][1][2]=  -3334297.25       # for  A  C  M  M
        WH[0][1][2][2]=    343546.79       # for  A  A  C  M
        WS[0][0][1][2]=      -526.78       # for  A  C  C  M
        WS[0][1][1][2]=     -1148.10       # for  A  C  M  M
        WS[0][1][2][2]=       160.77       # for  A  A  C  M

        # Ca-Mg-Si Ternary System
        WH[0][0][1][3]=  -2464803.70       # for  S  C  C  M
        WH[0][1][1][3]=  -2026666.90       # for  S  C  M  M
        WH[0][1][3][3]=  -1143506.91       # for  S  S  C  M
        WS[0][0][1][3]=      -669.00       # for  S  C  C  M
        WS[0][1][1][3]=      -555.03       # for  S  C  M  M
        WS[0][1][3][3]=      -279.90       # for  S  S  C  M

        # Ca-Al-Si Ternary System
        WH[0][0][2][3]=    580678.70       # for  S  A  C  C
        WH[0][2][2][3]=  -2833471.13       # for  S  A  A  C
        WH[0][2][3][3]=  -2685775.05       # for  S  S  A  C
        WS[0][0][2][3]=       526.17       # for  S  A  C  C
        WS[0][2][2][3]=      -976.80       # for  S  A  A  C
        WS[0][2][3][3]=      -917.87       # for  S  S  A  C

        # Mg-Al-Si Ternary System
        WH[1][1][2][3]=    652384.49       # for  S  A  M  M
        WH[1][2][2][3]=  -3201173.35       # for  S  A  A  M
        WH[1][2][3][3]=  -1828080.99       # for  S  S  A  M
        WS[1][1][2][3]=       397.38       # for  S  A  M  M
        WS[1][2][2][3]=     -1382.29       # for  S  A  A  M
        WS[1][2][3][3]=      -693.44       # for  S  S  A  M

        # quaternary parameters 
        WH[0][1][2][3]=   2179011.74       # for  S  A  C  M
        WS[0][1][2][3]=      1328.50       # for  S  A  C  M

        # Note: S  A  M  M value is in error in Berman's PhD and in diCapitani & Brown '87 
        #       In Beckett's code (and Yoneda's), the correct numbers are printed.
        #       Error found and corrected by D. Ebel; The above values are correct.

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
        cp = mn['a'] + mn['c']/T**2 + mn['d']/np.sqrt(T) + mn['f']/T  # Joules/mole (mole of formula unit)
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
        intcp = mn['a']*(T-T1) - mn['c']*(1/T-1/T1) + 2*mn['d']*(np.sqrt(T)-np.sqrt(T1)) + mn['f']*(np.log(T)-np.log(T1))  # Joules*K/mole (mole of formula unit)
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
        intcp = mn['a']*(np.log(T)-np.log(T1)) - 0.5*mn['c']*(1/T**2-1/T1**2) - 2*mn['d']*(1/np.sqrt(T)-1/np.sqrt(T1)) - mn['f']*(1/T-1/T1)  # Joules*K/mole (mole of formula unit)
        if perafu:  # Joules*K per atoms-per-formula-unit (= Joules*K per atom)
            mol,mass,ch=dissect_molecule(mn['Formula'])
            n=0
            for m in mol:
                n+=mol[m]
            intcp /= n
        return intcp
        
    def get_H_at_T(self,mdb,mineral,T):
        """
        Compute the enthalpy of a mineral.

        Arguments:

          mdb              The database to use
          mineral          The abbreviated name of the mineral (column Abbrev in mdb)
          T                Temperature in [K]
    
        Returns:
          H                The enthalpy [J/mole]
        """
        mn      = mdb[mdb['Abbrev']==mineral].iloc[0]
        intCp   = self.get_int_Cp_dT(mdb,mineral,T)
        DHf0    = mn['Enthalpy']*1e3  # Note: 1e3 because it is given as kiloJoule/mol
        H       = DHf0 + intCp
        return H
        
    def get_S_at_T(self,mdb,mineral,T):
        """
        Compute the entropy of a mineral.

        Arguments:

          mdb              The database to use
          mineral          The abbreviated name of the mineral (column Abbrev in mdb)
          T                Temperature in [K]
    
        Returns:
          S                The entropy [J/mole/K]
        """
        mn      = mdb[mdb['Abbrev']==mineral].iloc[0]
        intCpT  = self.get_int_CpdivT_dT(mdb,mineral,T)
        S0      = mn['Entropy']
        S       = S0 + intCpT
        return S

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

          mdb              The database to use
          mineral          The abbreviated name of the mineral (column Abbrev in mdb)
          T                Temperature in [K]
    
        Returns:
    
          mu0              The mu_0 == Delta G_f_0 of the mineral [J/mole]
          
        """
        if 'a' in mdb.columns:
            H       = self.get_H_at_T(mdb,mineral,T)
            S       = self.get_S_at_T(mdb,mineral,T)
            mu0     = H - T * S
        elif 'k0' in mdb.columns:
            if not hasattr(self,'berman88'):
                import sys
                sys.path.append('../')
                from Berman1988 import Berman88
                #self.berman88 = Berman88(T,path=self.path,ext='_berman88')
                self.berman88 = Berman88(self.endmselect,T)
                self.berman88.mdb_orig = mdb.copy()
                self.berman88.mdb = mdb
                self.berman88.reset(T)
            else:
                if T!=self.T: self.berman88.reset(T)
            mu0 = self.berman88.get_mu0_at_T(self.berman88.mdb,mineral,T,1.)
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
