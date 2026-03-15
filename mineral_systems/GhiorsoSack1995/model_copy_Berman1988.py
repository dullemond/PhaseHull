#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                                June 2025
#
# The Berman 1988 database. This one has no liquid phase parameters.
# But in conjunction with the Ghiorso & Sack 1995 interaction parameters
# it can be used also for solid-liquid phase diagrams. Note that for use
# with the Ghiorso & Sack interaction parameters, the Anorthite should be
# adjusted. See Ghiorso & Sack 1995.
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 1000)
import os
from phasehull import dissect_molecule,dissect_oxide,identify_component_minerals,Margules

class copyBerman88(object):
    """
    The Berman 1988 database of crystal phases. This does not have a liquid phase.
    You first set up this object (see arguments below). Then you can use the following
    functions and data (use ? to read their doc strings):

      reset(T,P)           Change the fixed temperature and pressure to new values.
                           Calls reset_crystals() and reset_liquids().

      reset_crystals(T,P)  Change the fixed temperature and pressure of the crystal
                           database mdb to new values, and stores T,P..
                           You can pass this on to PhaseHull.CrystalDatabase() for
                           allowing PhaseHull to automatically reset the T and P if
                           necessary.
    
      DaG(f,ph,T,P)        Returns the Delta_a G for formula f at temperature
                           T and P (may internally reset T, P). Note that this function
                           is just for your convenience, and is not necessary for the
                           PhaseHull algorithm.

      mdb                  The Pandas database of crystal phases

    """
    def __init__(self,compselect=None,T=298.15,P=1,path=None,ext=''):
        if path is None: path = os.path.dirname(__file__)
        self.Rgas       = 8.314  # J/mol·K
        self.T          = T
        self.P          = P
        self.mdb_orig   = pd.read_fwf(os.path.join(path,'minerals'+ext+'.fwf'))
        if compselect is None:
            self.mdb    = self.mdb_orig.copy()
        else:
            self.mdb    = self.extract_from_mineral_database_based_on_components(self.mdb_orig,compselect)
        self.reset(T,P)

    def reset(self,T,P=1):
        self.T          = T
        self.P          = P
        self.reset_crystals(T,P)

    def reset_crystals(self,T,P=1):
        self.T          = T
        self.P          = P
        self.compute_DfG_with_mole_fraction_weighting(self.mdb,T,P)

    def get_Cp(self,mdb,mineral,T,perafu=False):
        mn = mdb[mdb['Abbrev']==mineral].iloc[0]
        cp = mn['k0'] + 1e5*mn['k2x1e-5']/T**2 + 1e2*mn['k1x1e-2']/np.sqrt(T) + 1e7*mn['k3x1e-7']/T**3  # Joules/mole (mole of formula unit)
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
        intcp = mn['k0']*(T-T1) - 1e5*mn['k2x1e-5']*(1/T-1/T1) + 1e2*2*mn['k1x1e-2']*(np.sqrt(T)-np.sqrt(T1)) - 1e7*0.5*mn['k3x1e-7']*(1/T**2-1/T1**2)  # Joules*K/mole (mole of formula unit)
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
        intcp = mn['k0']*(np.log(T)-np.log(T1)) - 1e5*0.5*mn['k2x1e-5']*(1/T**2-1/T1**2) - 1e2*2*mn['k1x1e-2']*(1/np.sqrt(T)-1/np.sqrt(T1)) - 1e7*(1/3.)*mn['k3x1e-7']*(1/T**3-1/T1**3)  # Joules*K/mole (mole of formula unit)
        if perafu:  # Joules*K per atoms-per-formula-unit (= Joules*K per atom)
            mol,mass,ch=dissect_molecule(mn['Formula'])
            n=0
            for m in mol:
                n+=mol[m]
            intcp /= n
        return intcp

    def get_int_volume_dP(self,mdb,mineral,T,P,perafu=False):
        """
        The integral_Pr^P V(T,P) dP, which is the
        bottom two lines of Eq. 6 of Berman 1988.
        """
        mn     = mdb[mdb['Abbrev']==mineral].iloc[0]
        T1     = 298.15
        P1     = 1.
        v1     = 1e-6*mn['v1x1e6']
        v2     = 1e-12*mn['v2x1e12']
        v3     = 1e-6*mn['v3x1e6']
        v4     = 1e-10*mn['v4x1e10']
        dP1    = P-P1
        dP2    = P**2-P1**2
        dP3    = P**3-P1**3
        dT1    = T-T1
        dT2    = (T-T1)**2
        intvdp = mn['Volume']*((v1/2-v2*P1)*dP2+v2*dP3/3+(1-v1+v2*P1+v3*dT1+v4*dT2)*dP1)
        return intvdp

    def get_Cp_lambda(self,mdb,mineral,T,P,perafu=False):
        mn = mdb[mdb['Abbrev']==mineral].iloc[0]
        if mn['Tlam']>0:
            Tlam  = mn['Tlam']
            Tref  = mn['Tref']
            k     = mn['dT/dP']
            l1    = 1e-2*mn['l1x1e2']
            l2    = 1e-5*mn['l2x1e5']
            Tlamp = Tlam + k*(P-1)
            Td    = Tlam - Tlamp
            Tpr   = T + Td
            Cp    = Tpr*(l1+l2*Tpr)**2
            if np.isscalar(T):
                if Tpr>Tlam:
                    Cp = 0.
            else:
                mask  = Tpr>Tlam
                Cp[mask] = 0.
        else:
            Cp    = 0
        return Cp

    def get_DlG_lambda(self,mdb,mineral,T,P,perafu=False):
        """
        The treatment of the Lambda-transition. See Berman 1988 Eqs. 10+11.
        """
        mn = mdb[mdb['Abbrev']==mineral].iloc[0]
        if mn['Tlam']>0:
            Tlam  = mn['Tlam']
            Tref  = mn['Tref']
            k     = mn['dT/dP']
            l1    = 1e-2*mn['l1x1e2']
            l2    = 1e-5*mn['l2x1e5']
            l1_2  = l1**2
            l2_2  = l2**2
            l12   = l1*l2
            Tlamp = Tlam + k*(P-1)
            Td    = Tlam - Tlamp
            Tpr   = T + Td
            if np.isscalar(T):
                if Tpr>Tlam:
                    Tpr = Tlam
            else:
                mask  = Tpr>Tlam
                Tpr[mask] = Tlam
            #cp    = Tpr*(l1+l2*Tpr)**2
            tr    = Tref-Td
            x1    = l1_2*Td + 2*l12*Td**2 +   l2_2*Td**3
            x2    = l1_2    + 4*l12*Td    + 3*l2_2*Td**2
            x3    =           2*l12       + 3*l2_2*Td
            x4    =                           l2_2
            if np.isscalar(T):
                Tint  = T
                if Tint>Tlamp:
                    Tint = Tlamp
            else:
                Tint       = T.copy()
                mask       = Tint>Tlamp
                Tint[mask] = Tlamp
            dT1   = Tint    - tr
            dT2   = Tint**2 - tr**2
            dT3   = Tint**3 - tr**3
            dT4   = Tint**4 - tr**4
            dT0   = np.log(Tint) - np.log(tr)
            DlH   = x1*dT1 + (x2/2)*dT2 + (x3/3)*dT3 + (x4/4)*dT4
            DlS   = x1*dT0 +     x2*dT1 + (x3/2)*dT2 + (x4/3)*dT3
            DlG   = DlH - T*DlS
            if np.isscalar(T):
                if Tpr<Tref:
                    DlG = 0.
            else:
                mask  = Tpr<Tref
                DlG[mask] = 0.
        else:
            DlG   = 0.
        if perafu:  # Joules per atoms-per-formula-unit (= Joules per atom)
            mol,mass,ch=dissect_molecule(mn['Formula'])
            n=0
            for m in mol:
                n+=mol[m]
            DlG /= n
        return DlG

    def get_mu0_at_T(self,mdb,mineral,T,P,nolambda=False):
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
          P                Pressure in [bar]
    
        Returns:
    
          mu0              The mu_0 == Delta G_f_0 of the mineral [J/mole]
          
        """
        mn      = mdb[mdb['Abbrev']==mineral].iloc[0]
        intCp   = self.get_int_Cp_dT(mdb,mineral,T)
        intCpT  = self.get_int_CpdivT_dT(mdb,mineral,T)
        intVol  = self.get_int_volume_dP(mdb,mineral,T,P)
        DlGlamb = self.get_DlG_lambda(mdb,mineral,T,P)
        # At the moment the order/disorder contributions (Berman 1988 Eqs 15-20, table 5)
        # is not included. It applies only to Dolomite, Gehlenite and potassium feldspar.
        DHf0    = mn['Enthalpy']*1e3  # Note: 1e3 because it is given as kiloJoule/mol
        S0      = mn['Entropy']
        if nolambda: DlGlamb=0
        mu0     = DHf0 + intCp - T * ( S0 + intCpT ) + intVol + DlGlamb
        return mu0

    def extract_from_mineral_database_based_on_components(self,mdb,components):
        """
        Given a list of minerals in Pandas dataframe mdb (see read_minerals_and_liquids()), select only
        those minerals that are composed of the components given in the list components. Also add
        columns of x and moles.
    
        Arguments:
    
          mdb              The mineral database (see read_minerals_and_liquids())
          components       List of the formulae of the components, e.g. ['SiO2','MgO','Al2O3'].
    
        Returns:
    
          select           A version of mdb with only the minerals that can be created
                           from the components, and a column with the x and moles values.
                           The x are the mole fractions. The moles are the nr of moles
                           of that mineral that can be made from 1 mole of components.
                           Example: with 0.333 mole of SiO2 and 0.667 mole of MgO (in
                           total 1 mole worth of components) you can create 0.333 mole of
                           Mg2SiO4.
        """
        nm     = len(mdb)
        nem    = len(components)
        select = mdb.copy()
        select['ok']    = False
        select['x']     = np.zeros((nm,nem)).tolist()
        select['moles'] = 0.
        for i,mn in select.iterrows():
            d = dissect_oxide(mn['Formula'],components=components)
            if d['complete'] and d['positive']:
                select.at[i,'ok']     = True
                select.at[i,'x']      = d['x']
                select.at[i,'moles']  = d['moles']
        select = select[select['ok']].copy().reset_index(drop=True).drop('ok',axis=1)
        return select

    def compute_DfG_with_mole_fraction_weighting(self,mdb,T,P,no_mfDfG=False):
        """
        After having removed all minerals from the mdb database that are not part of the
        component system with extract_from_mineral_database_based_on_components(mdb,components),
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
        0.667 mole of MgO (in total 1 mole worth of components) you can create 0.333 mole of
        Mg2SiO4. So mfDfG=0.333*DfG for Mg2SiO4 where DfG is the Delta_f G for 1 mole of Mg2SiO4.
    
        Note: mdb must have a column 'moles' (how many moles of that mineral can we create from
              1 mole total of components). It is easiest to use the function
              extract_from_mineral_database_based_on_components(mdb,components) to automatically
              add this column. If you do not want to compute mfDfG (the Delta_f G for mdb['mole']
              amounts of moles of mineral), you get set no_mfDfG=True
    
        Arguments:
    
          mdb              The mineral database (see read_minerals_and_liquids())
          T                The temperature in [K]
    
        Returns:
    
          modifies the mdb database in-place.
        
        """
        mdb['DfG']   = 1e90   # The DfG per mole of this substance
        if not no_mfDfG:
            mdb['mfDfG'] = 1e90   # The DfG per mole of the constituent components
        for i,row in mdb.iterrows():
            DfG                = self.get_mu0_at_T(mdb,row['Abbrev'],T,P)
            mdb.at[i,'DfG']    = DfG
            if not no_mfDfG and 'moles' in row:
                mdb.at[i,'mfDfG']  = DfG * row['moles']

    # The functions below are not used by PhaseHull, but are more convenient for
    # other uses.

    def DaG(self,formula,phase,T,P=1.,name=None):
        """
        (convenience wrapper)
        Apparent Delta Gibbs energy of formation from the elements. So you form the compound
        at 298.15 K from the standard state of the elements, and then you heat up to the
        desired temperature. See Berman 1988 for more details. The equation is:

           Delta_a G = Delta_a H - T*S

        in units of [J/mol], where

           Delta_a H = Delta_f H(298.15) + int_298.15^T C_p(T)dT
           S         = S(298.15)         + int_298.15^T C_p(T)/T dT

        Arguments:

          formula    String chemical formula, e.g., 'Mg2SiO4'

          phase      String phase. Since Berman1988 only has solid
                     phase, this should always be 's' or 'sol' or 'cr'

          T          Temperature in [K]. Can be scalar or array.

          name       In case a formula has multiple different crystal structure
                     options, this name helps identify which one to use. If not
                     specified, this function will take the first one in the
                     mineral database.
        
        Returns:

          Delta_a G  The apparent Delta Gibbs energy of formation from the elements in [J/mol]

        Note about solids:

          For some chemical formulae, there exist multiple crystal types. Use the name keyword to
          select. Otherwise a random crystal is automatically chosen.
        
        """
        if phase=='l' or phase=='liq' or phase=='melt':
            raise ValueError('The Berman 1988 model does not have a liquid phase')
        elif phase=='s' or phase=='sol' or phase=='cr' or phase=='cryst':
            mdb = self.mdb
            mdb = mdb[mdb['Formula']==formula]
            if len(mdb)==0:
                raise ValueError(f'Error: Could not find formula {formula} in solid mineral database.')
            elif len(mdb)==1:
                row = mdb.iloc[0]
            else:
                if name is not None:
                    if name in list(mdb['Abbrev']):
                        mdb = mdb[mdb['Abbrev']==name]
                    elif name in list(mdb['Mineral']):
                        mdb = mdb[mdb['Mineral']==name]
                    else:
                        raise ValueError(f'Error: Could not find formula {formula} with name {name} in solid mineral database.')
                    if len(mdb)>1:
                        raise ValueError(f'Error: Found more than one formula {formula} with name {name} in solid mineral database.')
                    row = mdb.iloc[0]
                else:
                    print(f'Found more than one possible crystals matching {formula}. Options are:')
                    for name in list(mdb['Abbrev']):
                        print(name)
                    row  = mdb.iloc[0]
                    name = row['Mineral']
                    print(f'Taking arbitrary one: {name}')
            DaG = self.get_mu0_at_T(self.mdb,row['Abbrev'],T,P)
        else:
            raise ValueError(f'Error: I do not know phase {phase}')
        return DaG
