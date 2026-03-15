#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                              January 2026
#
# The Ghiorso & Sack 1995 model, the model from the MELTS code.
# The formulas in this code are copies from the equivalent formulas
# from the LiquidMelts*.m files of the MELTS code. The MELTS code
# Created by Mark Ghiorso on 6/18/10, Copyright 2010 OFM Research Inc.
# is open source and can be downloaded from gitlab using
# git clone https://gitlab.com/ENKI-portal/ThermoEngine.git
# However, the input data are not hardcoded here, but instead are
# in the files minerals.fwf, liquids.fwf. Only W is hardcoded.
# Some things are simplified here, e.g., while MELTS treats SiO2
# separately, here SiO2 is treated just like the other minerals,
# with parameters adjusted such that the H, S and G functions
# overlap nearly perfectly with the MELTS values. H2O is implemented
# here also in a simpler way than MELTS, and while our implementation
# (the same as for all components) works fairly well at low pressure,
# it becomes unreliable for pressures >1000 bar.
#
# Another thing to note about H2O is that the model of Ghiorso & Sack
# treats H2O also special by adding the xw*ln(xw) and (1-xw)*ln(1-xw)
# terms. This is also included here.
#
# Also note that, in the current version, there are no solid solutions
# implemented yet, so the solids are just individual fixed-composition
# crystals.
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 1000)
import os
from phasehull import dissect_molecule,dissect_oxide,identify_component_minerals,Margules
try:
    from .model_copy_Berman1988 import copyBerman88   # From inside this package
except:
    from model_copy_Berman1988 import copyBerman88    # From outside this package

class GhiorsoSack95(object):
    """
    The Ghiorso & Sach 1995 model with some additions from the MELTS code, but
    without the solid solutions (so far). Only the liquid and the fixed-composition
    crystals.
    
    You first set up this object (see arguments below). Then you can use the following
    functions and data (use ? to read their doc strings):

      reset(T,P)           Change the fixed temperature and pressure to new values.
                           Calls reset_crystals() and reset_liquids().

      reset_crystals(T,P)  Change the fixed temperature and pressure of the crystal
                           database mdb to new values, and stores T,P..
                           You can pass this on to PhaseHull.CrystalDatabase() for
                           allowing PhaseHull to automatically reset the T and P if
                           necessary.
    
      reset_liquid(T,P)    Change the fixed temperature and pressure of the liquid
                           database ldb to new values, and stores T,P.
                           You can pass this on to PhaseHull.Liquid() for
                           allowing PhaseHull to automatically reset the T and P if
                           necessary.
    
      liquid_Gfunc(x)      The function to compute the Delta_a G for the liquid phase,
                           to pass on to the PhaseHull code.
    
      DaG(f,ph,T,P)        Returns the Delta_a G for formula f in phase ph at temperature
                           T and P (may internally reset T, P). Note that this function
                           is just for your convenience, and is not necessary for the
                           PhaseHull algorithm.

      mdb                  The Pandas database of crystal phases

      ldb                  The Pandas database of liquid phases at the corners of the
                           phase diagram (the system components).
    """
    def __init__(self,compselect=None,T=298.15,P=1,path=None,ext=None,MELTS=False):
        """
        Arguments:

          compselect       List of system components you want to restrict to. You
                           can choose from SiO2, TiO2, Al2O3, Fe2O3, MgCr2O4, Fe2SiO4,
                           Mn2SiO4, Mg2SiO4, Ni2SiO4, Co2SiO4, CaSiO3, Na2SiO3,
                           KAlSiO4, Ca3(PO4)2, and H2O.  If no selection
                           is made, the full set is used.

          T                Temperature in [K]. You can always change it later, using the
                           reset(T,P) function . But the philosophy of the models here is 
                           that you fix a T and P, and compute the phase diagram for those.

          P                Pressure in [bar]. Similar philosophy as for T.

          MELTS            Default False: Use original Ghiorso & Sack 1995 data from
                           their original 1995 paper: minerals.fwf
                           If True: Use minerals_MELTS.fwf, which contains the
                           mineral data extracted (parsed) from the open-source
                           files (in objective-C) of the MELTS code. See
                           https://melts.ofm-research.org. A python parsing script
                           was used, with some manual post-processing, to create
                           this file.

          ext              If ext is a string, it overrides the MELTS key, and uses
                           the ext as extension. So set it to "" to read minerals.fwf

        Some tips:

          -  In the MELTS code you typically set the mass-percentages of the primitive
             components SiO2, TiO2, Al2O3, Fe2O3, Cr2O3, FeO, MnO, MgO, NiO, CoO, CaO, Na2O,
             K2O, P2O5, and H2O (or a subset of them). You can use the function
             convert_mass_fraction_into_mole_fraction() from phasehull_support.py to
             convert these to mole fractions (of the primitive components), and then
             use the SubSystem class and its function convert_from_xprim_to_xcomp()
             to compute the mole fractions of the complex components. That x is then
             what you can use here. Example:

                from phasehull.phasehull_support import *
                from phasehull.phasehull_subsystem import *
                comp     = ['SiO2', 'Al2O3', 'Fe2O3', 'Fe2SiO4','Mg2SiO4','CaSiO3','Na2SiO3','KAlSiO4','Mn2SiO4']
                compfact = [1     , 1      , 1      , 1        , 1       , 1      , 1       , 1       , 0.5     ]
                compox   = ['SiO2', 'Al2O3', 'Fe2O3', 'FeO',    'MgO',    'CaO',   'Na2O',   'K2O'    , 'MnO'   ]
                xmassox  = np.array([77.5, 12.5, 0.207, 0.473, 0.03, 0.43, 3.98, 4.88, 0.01])  # Input of MELTS
                xmassox /= xmassox.sum()
                xmolox   = convert_mass_fraction_into_mole_fraction(compox,xmassox)
                subsys   = SubSystem(compox,comp,comp_comp_weights=np.array(compfact))
                xmol     = subsys.convert_from_xprim_to_xcomp(xmolox)

             Note the 0.5 weight of Mn2SiO4, because in the liquid model of Ghiorso
             and Sack 1995 (implemented in MELTS) for the components Mn2SiO4, Ni2SiO4 
             and Co2SiO4 only half these units are used, i.e. MnSi(1/2)O2 etc.

          -  You can convert the activities you obtain from this model (which are
             activities of the complex components) into activities of the primitive
             components (see above) by using the SubSystem class and its function
             convert_activity_from_comp_to_prim(). Same for chemical potentials:
             the function convert_chemical_potential_from_comp_to_prim().
        
        """
        if path is None: path = os.path.dirname(__file__)
        self.Rgas       = 8.314  # J/mol·K
        
        # The solids/crystals of the Ghiorso & Sack 1995 model are based on the Berman 1998 data
        # These data are in the file minerals_GS95.fwf. In addition to the Berman 1998 minerals, 
        # this file also contains the additional minerals from the Ghiorso & Sack 1995 paper in 
        # their big table. They are automatically read as well with the Berman88() call.
        #
        # NOTE: With the MELTS=True keyword, this module will use the file minerals_MELTS.fwf
        #       instead. These data were directly parsed/extracted from the open-source
        #       MELTS code from the BermanStoichiometricPhases.m file using an automatic
        #       python script. This does not guarantee that this is then exactly the
        #       same model as the MELTS code, because some of the minerals are handled
        #       differently in MELTS than in the Berman 1988 paper. But most minerals
        #       will be MELTS-conform, I hope ;-).
        if ext is None:
            if MELTS:
                ext = "_MELTS"
            else:
                ext = "_GS95"
        self.berman88   = copyBerman88(None,T,P,path=path,ext=ext)
        self.T          = T
        self.P          = P
        
        # The components of the Ghiorso & Sack 1995 model are the following, where it is
        # to be noted that for Mn2SiO4, Ni2SiO4 and Co2SiO4 only half these units are used.
        # See "Factor" column in liquids.fwf and factors in the subroutines below.
        self.components = ['SiO2','TiO2','Al2O3','Fe2O3','MgCr2O4','Fe2SiO4','Mn2SiO4','Mg2SiO4','Ni2SiO4','Co2SiO4','CaSiO3','Na2SiO3','KAlSiO4','Ca3(PO4)2','H2O']
        self.compfactors= [   1.0,   1.0,   1.0,     1.0,      1.0,      1.0,      0.5,      1.0,      0.5,      0.5,     1.0,      1.0,      1.0,        1.0,  1.0]

        # The liquids parameters are in liquids.fwf
        self.ldb_orig   = pd.read_fwf(os.path.join(path,'liquids.fwf'))

        # Just for safety, check that the factors in the liquids.fwf file are the same as the above ones
        factors         = self.get_factors_of_components(self.components)
        assert np.max(np.abs(np.array(factors)-np.array(self.compfactors)))<1e-8, 'ERROR: Something went wrong with the factors of the componenets'
        
        # You can reduce the model by selecting only a subset of the above 15 components
        # by setting the compselect argument.
        self.compabbrev = ['']
        self.compselect = compselect
        if compselect is None:
            self.ldb    = self.ldb_orig.copy()
            self.mdb    = self.extract_from_mineral_database_based_on_components(self.berman88.mdb_orig,self.components)
        else:
            assert set(compselect)<=set(self.components), 'Error: The requested components are not all in this model.'
            self.ldb    = self.extract_from_mineral_database_based_on_components(self.ldb_orig,compselect)
            self.mdb    = self.extract_from_mineral_database_based_on_components(self.berman88.mdb_orig,compselect)

        # Make sure that the berman88 class uses the selected components only
        self.berman88.mdb = self.mdb

        # Make sure that the liquid components are correctly sorted in the self.ldb
        self.sort_liquid_components()

        # Complete the liquid database with the enthalpy and entropy at T=Tfusion
        self.compute_all_H_and_S_Liq_at_Tfusion(self.ldb)
        
        # The Margules parameters (the interaction parameters) of the  Ghiorso & Sack 1995 model
        # are binary interaction parameters.
        WH,cm,cw        = self.get_Margules()
        assert cm==self.components, 'ERROR: The Margules order of components is unequal to the full set of system components'
        assert cw==self.compfactors,'ERROR: The Margules factors of components is unequal to the full set of system components'

        # Set the Margules class, which contains functions to use the Margules formalism for the liquid
        self.margules   = Margules(self.components)
        self.margules.load_w(WH)

        # Now reduce the Margules matrix to the selected components only
        self.margules.reduce_margules_to_subset(compselect)
        
        # Add a warning:
        if 'H2O' in compselect:
            print('WARNING: As of now, H2O is in this code treated like any component. This is not how it is implemented in the MELTS code / the Ghiorso & Sack 1995 paper, where water is treated special. These corrections are not yet built in here as of now, but are planned to be at a later time.')
        #
        self.reset(T,P)

    def reset(self,T,P=1):
        self.T          = T
        self.P          = P
        self.reset_crystals(T,P)
        self.reset_liquid(T,P)

    def reset_crystals(self,T,P=1):
        self.T          = T
        self.P          = P
        self.berman88.compute_DfG_with_mole_fraction_weighting(self.berman88.mdb,T,P)

    def reset_liquid(self,T,P=1):
        self.T          = T
        self.P          = P
        self.compute_DfG_liquid_components(self.ldb,T,P)

    def liquid_Gfunc(self,x,incl_linear=True,incl_ideal=True,incl_nonideal=True):
        """
        The function of the composition x (given in terms of the liquid components
        listed in self.components) that returns (for the temperature self.T and
        pressure self.P) the Gibbs free energy of the liquid phase.

        Arguments:

          x            The mole (!) fractions. Must be array of shape [nx,N]
                       where nx is the number of points of x, and N is the
                       number of components.

        Returns:

          G            Gibbs free energy of the liquid in J/mole-of-system-component.
        """
        if len(x.shape)==1:
            x = np.array([x,])
        assert x.shape[-1]==len(self.compselect), 'Error: Dimension of x incorrect.'
        G = self.compute_G_of_liquid_mixture(self.ldb,self.T,self.P,x,self.compselect,
                                             incl_linear=incl_linear,incl_ideal=incl_ideal,
                                             incl_nonideal=incl_nonideal)
        return G

    def Gfunc(self,x):
        """
        Only for backward compatibility. It calls the self.liquid_Gfunc() function.
        """
        return self.liquid_Gfunc(x)

    def get_Margules(self):
        # Table 4 of Ghiorso & Sack 1995 with the extra components of MELTS included
        # These data/lines were copied from the MELTS source file LiquidMelts.m
        self.referenceValuesOfModelParameters = [
          26266.7,  #   0 W(TiO2      ,SiO2      )
         -39120.0,  #   1 W(Al2O3     ,SiO2      )
           8110.3,  #   2 W(Fe2O3     ,SiO2      )
          27886.3,  #   3 W(MgCr2O4   ,SiO2      )
          23660.9,  #   4 W(Fe2SiO4   ,SiO2      )
          18393.9,  #   5 W(MnSi0.5O2 ,SiO2      )
           3421.0,  #   6 W(Mg2SiO4   ,SiO2      )
          25197.4,  #   7 W(NiSi0.5O2 ,SiO2      )
          14802.8,  #   8 W(CoSi0.5O2 ,SiO2      )
           -863.7,  #   9 W(CaSiO3    ,SiO2      )
         -99039.0,  #  10 W(Na2SiO3   ,SiO2      )
         -33921.7,  #  11 W(KAlSiO4   ,SiO2      )
          61891.6,  #  12 W(Ca3(PO4)2 ,SiO2      )
          30967.3,  #  13 W(H2O       ,SiO2      )

         -29449.8,  #  14 W(Al2O3     ,TiO2      )
         -84756.9,  #  15 W(Fe2O3     ,TiO2      )
         -72303.4,  #  16 W(MgCr2O4   ,TiO2      )
           5209.1,  #  17 W(Fe2SiO4   ,TiO2      )
         -16123.5,  #  18 W(MnSi0.5O2 ,TiO2      )
          -4178.3,  #  19 W(Mg2SiO4   ,TiO2      )
           3614.8,  #  20 W(NiSi0.5O2 ,TiO2      )
          -1640.0,  #  21 W(CoSi0.5O2 ,TiO2      )
         -35372.5,  #  22 W(CaSiO3    ,TiO2      )
         -15415.6,  #  23 W(Na2SiO3   ,TiO2      )
         -48094.6,  #  24 W(KAlSiO4   ,TiO2      )
          25938.8,  #  25 W(Ca3(PO4)2 ,TiO2      )
          81879.1,  #  26 W(H2O       ,TiO2      )

         -17089.4,  #  27 W(Fe2O3     ,Al2O3     )
         -31770.3,  #  28 W(MgCr2O4   ,Al2O3     )
         -30509.0,  #  29 W(Fe2SiO4   ,Al2O3     )
         -53874.9,  #  30 W(MnSi0.5O2 ,Al2O3     )
         -32880.3,  #  31 W(Mg2SiO4   ,Al2O3     )
           2985.2,  #  32 W(NiSi0.5O2 ,Al2O3     )
          -2677.4,  #  33 W(CoSi0.5O2 ,Al2O3     )
         -57917.9,  #  34 W(CaSiO3    ,Al2O3     )
        -130785.0,  #  35 W(Na2SiO3   ,Al2O3     )
         -25859.2,  #  36 W(KAlSiO4   ,Al2O3     )
          52220.8,  #  37 W(Ca3(PO4)2 ,Al2O3     )
         -16098.1,  #  38 W(H2O       ,Al2O3     )

          21605.9,  #  39 W(MgCr2O4   ,Fe2O3     )
        -179064.9,  #  40 W(Fe2SiO4   ,Fe2O3     )
           3907.9,  #  41 W(MnSi0.5O2 ,Fe2O3     )
         -71518.6,  #  42 W(Mg2SiO4   ,Fe2O3     )
            408.7,  #  43 W(NiSi0.5O2 ,Fe2O3     )
           -223.7,  #  44 W(CoSi0.5O2 ,Fe2O3     )
          12076.6,  #  45 W(CaSiO3    ,Fe2O3     )
        -149662.2,  #  46 W(Na2SiO3   ,Fe2O3     )
          57555.9,  #  47 W(KAlSiO4   ,Fe2O3     )
          -4213.9,  #  48 W(Ca3(PO4)2 ,Fe2O3     )
          31405.5,  #  49 W(H2O       ,Fe2O3     )

         -82971.8,  #  50 W(Fe2SiO4   ,MgCr2O4   )
            182.4,  #  51 W(MnSi0.5O2 ,MgCr2O4   )
          46049.2,  #  52 W(Mg2SiO4   ,MgCr2O4   )
           -266.0,  #  53 W(NiSi0.5O2 ,MgCr2O4   )
           -384.0,  #  54 W(CoSi0.5O2 ,MgCr2O4   )
          30704.7,  #  55 W(CaSiO3    ,MgCr2O4   )
         113646.0,  #  56 W(Na2SiO3   ,MgCr2O4   )
          75709.1,  #  57 W(KAlSiO4   ,MgCr2O4   )
           5341.8,  #  58 W(Ca3(PO4)2 ,MgCr2O4   )
              0.0,  #  59 W(H2O       ,MgCr2O4   )

          -6823.9,  #  60 W(MnSi0.5O2 ,Fe2SiO4   )
         -37256.7,  #  61 W(Mg2SiO4   ,Fe2SiO4   )
         -17019.8,  #  62 W(NiSi0.5O2 ,Fe2SiO4   )
         -11746.3,  #  63 W(CoSi0.5O2 ,Fe2SiO4   )
         -12970.8,  #  64 W(CaSiO3    ,Fe2SiO4   )
         -90533.8,  #  65 W(Na2SiO3   ,Fe2SiO4   )
          23649.4,  #  66 W(KAlSiO4   ,Fe2SiO4   )
          87410.3,  #  67 W(Ca3(PO4)2 ,Fe2SiO4   )
          28873.6,  #  68 W(H2O       ,Fe2SiO4   )

         -13040.1,  #  69 W(Mg2SiO4   ,MnSi0.5O2 )
            785.8,  #  70 W(NiSi0.5O2 ,MnSi0.5O2 )
            -50.6,  #  71 W(CoSi0.5O2 ,MnSi0.5O2 )
           2934.6,  #  72 W(CaSiO3    ,MnSi0.5O2 )
         -15780.8,  #  73 W(Na2SiO3   ,MnSi0.5O2 )
          23727.4,  #  74 W(KAlSiO4   ,MnSi0.5O2 )
              0.0,  #  75 W(Ca3(PO4)2 ,MnSi0.5O2 )
              0.0,  #  76 W(H2O       ,MnSi0.5O2 )

         -21175.5,  #  77 W(NiSi0.5O2 ,Mg2SiO4   )
         -14994.9,  #  78 W(CoSi0.5O2 ,Mg2SiO4   )
         -31731.9,  #  79 W(CaSiO3    ,Mg2SiO4   )
         -41876.9,  #  80 W(Na2SiO3   ,Mg2SiO4   )
          22323.1,  #  81 W(KAlSiO4   ,Mg2SiO4   )
         -23208.8,  #  82 W(Ca3(PO4)2 ,Mg2SiO4   )
          35633.7,  #  83 W(H2O       ,Mg2SiO4   )

            258.9,  #  84 W(CoSi0.5O2 ,NiSi0.5O2 )
           7027.5,  #  85 W(CaSiO3    ,NiSi0.5O2 )
          -3647.8,  #  86 W(Na2SiO3   ,NiSi0.5O2 )
           4261.4,  #  87 W(KAlSiO4   ,NiSi0.5O2 )
              0.0,  #  88 W(Ca3(PO4)2 ,NiSi0.5O2 )
              0.0,  #  89 W(H2O       ,NiSi0.5O2 )

         -26685.7,  #  90 W(CaSiO3    ,CoSi0.5O2 )
            531.2,  #  91 W(Na2SiO3   ,CoSi0.5O2 )
            265.7,  #  92 W(KAlSiO4   ,CoSi0.5O2 )
              0.0,  #  93 W(Ca3(PO4)2 ,CoSi0.5O2 )
              0.0,  #  94 W(H2O       ,CoSi0.5O2 )

         -13247.1,  #  95 W(Na2SiO3   ,CaSiO3    )
          17111.1,  #  96 W(KAlSiO4   ,CaSiO3    )
          37070.3,  #  97 W(Ca3(PO4)2 ,CaSiO3    )
          20374.6,  #  98 W(H2O       ,CaSiO3    )

           6522.8,  #  99 W(KAlSiO4   ,Na2SiO3   )
          15571.9,  # 100 W(Ca3(PO4)2 ,Na2SiO3   )
         -96937.6,  # 101 W(H2O       ,Na2SiO3   )

          17100.6,  # 102 W(Ca3(PO4)2 ,KAlSiO4   )
          10374.2,  # 103 W(H2O       ,KAlSiO4   )

          43451.3]  # 104 W(H2O       ,Ca3(PO4)2 )
        NA      = 15
        WH      = np.zeros((NA,NA))  # The binary Margules parameters for the liquid components 
        i       = 0
        j       = 1
        NK      = (NA*NA-NA)//2
        assert len(self.referenceValuesOfModelParameters)==NK, 'Error in interaction parameters'
        for k in range(NK):
            WH[i,j] = self.referenceValuesOfModelParameters[k]
            WH[j,i] = self.referenceValuesOfModelParameters[k]
            j += 1
            if j>=NA:
                i+=1
                j=i+1
        assert j==i+1, 'Something went wrong with WH.'
        # NEVER change the below two lines, as that will mess up the association of the 
        # matrix indices to the components.
        components  = ['SiO2','TiO2','Al2O3','Fe2O3','MgCr2O4','Fe2SiO4','Mn2SiO4','Mg2SiO4','Ni2SiO4','Co2SiO4','CaSiO3','Na2SiO3','KAlSiO4','Ca3(PO4)2','H2O']
        compfactors = [   1.0,   1.0,   1.0,     1.0,      1.0,      1.0,      0.5,      1.0,      0.5,      0.5,     1.0,      1.0,      1.0,        1.0,  1.0]
        return WH,components,compfactors

    def compute_G_of_liquid_mixture(self,ldb,T,P,x,components,nomixG=False,
                                    check=True,incl_linear=True,incl_ideal=True,
                                    incl_nonideal=True):
        """
        Compute the full G(x,T) of a mixture of liquids, including the
        Gibbs of formation, the ideal Gibbs of mixing and the non-ideal Gibbs
        of mixing.
    
        Arguments:
    
          ldb          The database of liquids
    
          T            Temperature in Kelvin
          P            Pressure in bar
    
          x            The mole (!) fractions. Must be array of shape [nx,N]
                       where nx is the number of points of x, and N is the
                       number of components.
    
          components   The components (formulae, e.g. ['SiO2','Al2O3','MgO'])
    
        Options:
    
          nomixG       If True, then only return the unmixed mean G.

        Options for testing purposes:

          incl_linear    (default: True) Include the linear combination term
          incl_ideal     (default: True) Include the ideal mixing (entropy) term
          incl_nonideal  (default: True) Include the nonideal mixing (interaction) term
    
        """
        if T!=self.T or P!=self.P: self.reset_liquid(T,P)
        if type(x) is list: x = np.array(x)
        N          = x.shape[-1]
        nx         = x.shape[0]
        assert N==len(components), 'Error: Nr of components and x components not equal'
        icomponents,DfGcomponents = identify_component_minerals(ldb,components)
    
        G  = np.zeros(nx)
        
        # First the linear combination of the N components
        if incl_linear:
            for i in range(N):
                G += x[:,i]*DfGcomponents[i]
    
        # Then the mixing
        if not nomixG:
            if incl_ideal:    G += self.margules.compute_ideal_mixing_G(x,T,check=check)
            if incl_nonideal: G += self.margules.compute_interaction_G(x,T,P,check=check)
            if self.incl_h2o:
                # Ghiorso & Sack 1995 treat water special by adding two terms:
                iw = self.component_index['H2O']
                xw = x[...,iw]
                RT = self.Rgas*self.T
                G += RT*(xw*np.log(xw+1e-90)+(1-xw)*np.log(1-xw+1e-90))
        return G
    
    def get_H_at_T_solid(self,mdb,mineral,T):
        """
        Compute the enthalpy of a mineral.

        Arguments:

          mdb              The database to use
          mineral          The abbreviated name of the mineral (column Abbrev in mdb)
          T                Temperature in [K]
    
        Returns:
          H                The enthalpy [J/mole]
        """
        t       = T
        mn      = mdb[mdb['Abbrev']==mineral].iloc[0]
        hf0     = mn['Enthalpy']*1e3  # Note: 1e3 because it is given as kiloJoule/mol
        tr      = 298.15
        pr      = 1.
        trl     = mn['Trl']
        k0      = mn['k0']
        k1      = mn['k1x1e-2']*1e2
        k2      = mn['k2x1e-5']*1e5
        k3      = mn['k3x1e-7']*1e7
        l1      = mn['l1x1e2']*1e-2
        l2      = mn['l2x1e5']*1e-5
        Tt      = mn['Tlam']
        deltaH  = mn['DlH']
        result  = hf0 + k0*(t-tr) + 2.0*k1*(np.sqrt(t)-np.sqrt(tr)) - k2*(1.0/t-1.0/tr) - 0.5*k3*(1.0/(t*t)-1.0/(tr*tr))
        if(Tt > 0.0):
            if (t > Tt):
                result += deltaH + 0.5*l1*l1*(Tt*Tt-trl*trl) + (2.0/3.0)*l1*l2*(Tt*Tt*Tt-trl*trl*trl) + 0.25*l2*l2*(Tt*Tt*Tt*Tt-trl*trl*trl*trl)
            else:
                result += 0.5*l1*l1*(t*t-trl*trl) + (2.0/3.0)*l1*l2*(t*t*t-trl*trl*trl) + 0.25*l2*l2*(t*t*t*t-trl*trl*trl*trl)
        return result

    def get_S_at_T_solid(self,mdb,mineral,T):
        """
        Compute the entropy of a mineral.

        Arguments:

          mdb              The database to use
          mineral          The abbreviated name of the mineral (column Abbrev in mdb)
          T                Temperature in [K]
    
        Returns:
          S                The entropy [J/mole/K]
        """
        t       = T
        mn      = mdb[mdb['Abbrev']==mineral].iloc[0]
        sf0     = mn['Entropy']
        tr      = 298.15
        pr      = 1.
        trl     = mn['Trl']
        k0      = mn['k0']
        k1      = mn['k1x1e-2']*1e2
        k2      = mn['k2x1e-5']*1e5
        k3      = mn['k3x1e-7']*1e7
        l1      = mn['l1x1e2']*1e-2
        l2      = mn['l2x1e5']*1e-5
        Tt      = mn['Tlam']
        deltaH  = mn['DlH']
        result = sf0 + k0*np.log(t/tr) - 2.0*k1*(1.0/np.sqrt(t)-1.0/np.sqrt(tr)) - 0.5*k2*(1.0/(t*t)-1.0/(tr*tr)) - (1.0/3.0)*k3*(1.0/(t*t*t)-1.0/(tr*tr*tr))
        if(Tt > 0.0):
            if(t > Tt):
                result += deltaH/Tt + l1*l1*(Tt-trl) + l1*l2*(Tt*Tt-trl*trl) + (1.0/3.0)*l2*l2*(Tt*Tt*Tt-trl*trl*trl)
            else:
                result += l1*l1*(t-trl) + l1*l2*(t*t-trl*trl) + (1.0/3.0)*l2*l2*(t*t*t-trl*trl*trl)
        return result

    def get_H_at_T_P_liquid(self,ldb,mineral,T,P,weighted=False):
        """
        Compute the enthalpy of a pure liquid component

        Arguments:

          ldb              The database to use
          mineral          The abbreviated name of the mineral (column Abbrev in mdb)
          T                Temperature in [K]
          P                Pressure in [bar]
    
        Options:

          weighted         If True, then multiply the result
                           by the 'Factor' column. This is necessary
                           for the Ghiorso & Sack model, for the
                           components Mn2SiO4, Ni2SiO4 and Co2SiO4,
                           which the model uses in their half formula
                           units MnSi0.5O2, NiSi0.5O2 and CoSi0.5O2.

        Returns:
          H                The enthalpy [J/mole]
        """
        t             = T
        p             = P
        mn            = ldb[ldb['Abbrev']==mineral].iloc[0]
        hf0           = mn['Enthalpy']*1e3  # Note: 1e3 because it is given as kiloJoule/mol
        tr            = 298.15
        pr            = 1.
        trl           = mn['Trl']
        vLiq          = mn['Volume'] # In J/bar. Note that 1 J/bar = 10 cm^3, meaning volume = vLiq*10 cm^3/mol
        dvdtLiq       = mn['dVdTx1e4']*1e-4
        dvdpLiq       = mn['dVdPx1e5']*1e-5
        d2vdtdpLiq    = mn['d2VdPdTx1e8']*1e-8
        d2vdp2Liq     = mn['d2VdP2x1e10']*1e-10
        tFusion       = mn['Tfus']
        cpLiq         = mn['Cp0']
        hLiqAtTfusion = mn['HLiqAtTf']*1e3
        H             = hLiqAtTfusion + cpLiq*(t-tFusion)                                       \
            + (vLiq + dvdtLiq*(t-trl))*(p-pr)                                                   \
            + 0.5*(dvdpLiq + (t-trl)*d2vdtdpLiq)*(p*p-pr*pr)                                    \
            - (dvdpLiq + (t-trl)*d2vdtdpLiq)*pr*(p-pr)                                          \
            + d2vdp2Liq*( (p*p*p-pr*pr*pr)/6.0 - pr*(p*p-pr*pr)/2.0 + pr*pr*(p-pr)/2.0 )        \
            - t*(dvdtLiq*(p-pr) + 0.5*d2vdtdpLiq*(p-pr)*(p-pr))
        if weighted:
            mn      = ldb[ldb['Abbrev']==mineral].iloc[0]
            H      *= mn['Factor']
        return H

    def get_S_at_T_P_liquid(self,ldb,mineral,T,P,weighted=False):
        """
        Compute the entropy of a pure liquid component

        Arguments:

          ldb              The database to use
          mineral          The abbreviated name of the mineral (column Abbrev in mdb)
          T                Temperature in [K]
          P                Pressure in [bar]
    
        Options:

          weighted         If True, then multiply the result
                           by the 'Factor' column. This is necessary
                           for the Ghiorso & Sack model, for the
                           components Mn2SiO4, Ni2SiO4 and Co2SiO4,
                           which the model uses in their half formula
                           units MnSi0.5O2, NiSi0.5O2 and CoSi0.5O2.

        Returns:
          S                The entropy [J/mole/K]
        """
        t             = T
        p             = P
        mn            = ldb[ldb['Abbrev']==mineral].iloc[0]
        hf0           = mn['Entropy']
        tr            = 298.15
        pr            = 1.
        trl           = mn['Trl']
        dvdtLiq       = mn['dVdTx1e4']*1e-4
        d2vdtdpLiq    = mn['d2VdPdTx1e8']*1e-8
        tFusion       = mn['Tfus']
        cpLiq         = mn['Cp0']
        sLiqAtTfusion = mn['SLiqAtTf']
        S             = sLiqAtTfusion + cpLiq*np.log(t/tFusion) - (dvdtLiq*(p-pr) + 0.5*d2vdtdpLiq*(p-pr)*(p-pr))
        if weighted:
            mn      = ldb[ldb['Abbrev']==mineral].iloc[0]
            S      *= mn['Factor']
        return S

    def compute_H_and_S_Liq_at_Tfusion(self,ldb,mineral):
        mn            = ldb[ldb['Abbrev']==mineral].iloc[0]
        Tfusion       = mn['Tfus']
        Sfusion       = mn['DSfus']
        hLiqAtTfusion = self.get_H_at_T_solid(ldb,mineral,Tfusion)
        sLiqAtTfusion = self.get_S_at_T_solid(ldb,mineral,Tfusion)
        hLiqAtTfusion += Sfusion*Tfusion
        sLiqAtTfusion += Sfusion
        return hLiqAtTfusion,sLiqAtTfusion

    def compute_all_H_and_S_Liq_at_Tfusion(self,ldb):
        for i,row in ldb.iterrows():
            # NOTE: Only calculate if not given in liquids.fwf
            #       Otherwise use the value in liquids.fwf
            if np.isnan(ldb['HLiqAtTf'].iloc[i]):
                assert np.isnan(ldb['HLiqAtTf'].iloc[i]), 'Error: If HLiqAtTf not specified, then SLiqAtTf must be not specified'
                mineral = row['Abbrev']
                H,S     = self.compute_H_and_S_Liq_at_Tfusion(ldb,mineral)
                ldb.at[i,'HLiqAtTf'] = H/1e3
                ldb.at[i,'SLiqAtTf'] = S
            else:
                assert not np.isnan(ldb['HLiqAtTf'].iloc[i]), 'Error: If HLiqAtTf specified, then SLiqAtTf must be specified'

    def get_mu0_at_T_P_liquid(self,ldb,mineral,T,P,weighted=False):
        """
        The mu_0 (=Delta_f G) for this liquid component at temperature T and pressure P.
        This follows the big equation of the appendix of Ghiorso & Sack 1995, on their page
        208, see also the function getGibbsFreeEnergy in LiquidMeltsGenericEM.m

        Arguments:
    
          mineral          The abbreviated name of the mineral (column Abbrev in ldb)
          T                Temperature in [K]
          P                Pressure in [bar]
    
        Options:

          weighted         If True, then multiply the result
                           by the 'Factor' column. This is necessary
                           for the Ghiorso & Sack model, for the
                           components Mn2SiO4, Ni2SiO4 and Co2SiO4,
                           which the model uses in their half formula
                           units MnSi0.5O2, NiSi0.5O2 and CoSi0.5O2.

        Returns:
    
          mu0              The mu_0 == Delta G_f_0 of the liquid mineral [J/mole]
        """
        H       = self.get_H_at_T_P_liquid(ldb,mineral,T,P)
        S       = self.get_S_at_T_P_liquid(ldb,mineral,T,P)
        mu0     = H - T * S
        if weighted:
            mn      = ldb[ldb['Abbrev']==mineral].iloc[0]
            mu0    *= mn['Factor']
        return mu0

    def get_mu0_at_T_P(self,mdb,mineral,T,P):
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
        return self.berman88.get_mu0_at_T(mdb,mineral,T,P)
    
    def get_factors_of_components(self,components):
        """
        Because the Ghiorso & Sack MELTS code uses composite components with
        half factors (e.g. MnSi0.5O2), we need the find these factors.
        """
        assert hasattr(self,'ldb_orig'), 'Error: Cannot find ldb_orig.'
        assert 'Factor' in self.ldb_orig.columns, 'Error: The weight factor is not in the liquids.fwf file.'
        factors = []
        for e in components:
            row = self.ldb_orig[self.ldb_orig['Formula']==e]
            assert len(row)==1, 'Error while finding weight factor in ldb_orig'
            row = row.iloc[0]
            w   = float(row['Factor'])
            factors.append(w)
        return factors

    def extract_from_mineral_database_based_on_components(self,mdb,components,onlypositive=True):
        """
        Given a list of minerals in Pandas dataframe mdb (see read_minerals_and_liquids()), select only
        those minerals that are composed of the components given in the list components. Also add
        columns of x and moles.

        Note: Some of the components are onlt half of the formula given in the components list.
              This is called the 'factor'. The factors are automatically handled here, so that
              the composition mole fractions in the 'x' column of the returned dataframe are
              correct.
    
        Arguments:
    
          mdb              The mineral database (see read_minerals_and_liquids())
          components       List of the formulae of the components, e.g. ['SiO2','MgO','Al2O3'].

        Optional:

          onlypositive     If True, then only include minerals that can be constructed
                           from the components through >=0 contributions.
    
        Returns:
    
          select           A version of mdb with only the minerals that can be created
                           from the components, and a column with the x and moles values.
                           The x are the mole fractions. The moles are the nr of moles
                           of that mineral that can be made from 1 mole of components.
                           Example: with 0.333 mole of SiO2 and 0.667 mole of MgO (in
                           total 1 mole worth of components) you can create 0.333 mole of
                           Mg2SiO4.
        """
        # Get the factors of the components
        factors = self.get_factors_of_components(components)
            
        # Now extract the minerals that can be constructed from the components
        nm      = len(mdb)
        nem     = len(components)
        select  = mdb.copy()
        select['ok']    = False
        select['x']     = np.zeros((nm,nem)).tolist()
        select['moles'] = 0.
        for i,mn in select.iterrows():
            d = dissect_oxide(mn['Formula'],components=components,weights=factors)
            include = d['complete']
            if onlypositive:
                include = include and d['positive']
            if include:
                select.at[i,'ok']     = True
                select.at[i,'x']      = d['x']
                select.at[i,'moles']  = d['moles']
        select = select[select['ok']].copy().reset_index(drop=True).drop('ok',axis=1)
        return select

    def sort_liquid_components(self):
        """
        In principle it should not be necessary to sort the Pandas database for the liquids
        (self.ldb), but depending on how external applications use it, it might lead to
        confusion if the order of the component liquids is different from the ones given
        in self.compselect. So just to be on the safe side, we will order them here.
        Also, we check if water (H2O) is included, because in the GS95 model water is
        treated special.
        """
        ldb = self.ldb.reset_index(drop=True).set_index('Formula')
        ldb['idxcomp'] = -1
        for icomp,comp in enumerate(self.compselect):
            ldb.at[comp,'idxcomp'] = icomp
        ldb = ldb.reset_index().set_index('idxcomp')
        self.ldb = ldb.sort_index().reset_index()
        self.component_index = dict(self.ldb.copy().set_index('Formula')['idxcomp'])
        if 'H2O' in self.component_index:
            self.incl_h2o=True
        else:
            self.incl_h2o=False

    def compute_DfG_liquid_components(self,ldb,T,P,weighted=False):
        """
        Compute for all liquid components the Delta_f G(T,P) for one mole.
    
        Arguments:
    
          ldb              The liquid mineral database
          T                The temperature in [K]

        Options:

          weighted         If True, then multiply the result
                           by the 'Factor' column. This is necessary
                           for the Ghiorso & Sack model, for the
                           components Mn2SiO4, Ni2SiO4 and Co2SiO4,
                           which the model uses in their half formula
                           units MnSi0.5O2, NiSi0.5O2 and CoSi0.5O2.

        Returns:
    
          modifies the ldb database in-place.
        
        """
        ldb['DfG']   = 1e90   # The DfG per mole of this substance
        for i,row in ldb.iterrows():
            DfG                = self.get_mu0_at_T_P_liquid(ldb,row['Abbrev'],T,P,weighted=weighted)
            ldb.at[i,'DfG']    = DfG

    # The functions below are not used by PhaseHull, but are more convenient for
    # other uses.

    def get_activities_of_liquid_components(self,x,return_dict=False,return_gamma=False):
        """
        Compute the activities of the liquid components from the Margules parameters.

        Arguments:
        
          x      The x vector x[0:ncomponents] or array of vectors x[0:nx,0:ncomponents], such
                 that x.sum(axis=-1)==1.

          return_dict   If True, then instead of a list, return a
                        dict, so that it is clearer, which activity
                        belongs to which component. NOTE: Some components
                        have weight 0.5: Mn2SiO4, Ni2SiO4 and Co2SiO4,
                        which the model uses in their half formula
                        units MnSi0.5O2, NiSi0.5O2 and CoSi0.5O2.

          return_gamma  If True, then instead of returning the activity
                        itself, it will return the activity coefficient.

        Returns:

          a      List or dict of the activities of the (selected) components of the liquid
        """
        if type(x) is list:
            x = np.array(x)
        if len(x.shape)==1:
            scalar = True
            x = np.stack([x])
        else:
            scalar = False
        ncomp = x.shape[-1]
        assert ncomp==len(self.ldb), 'Error in x: does not contain same amount of components as the liquid'
        gamma = self.margules.get_activity_coefficients_of_components(x,self.T,self.P).T
        if self.incl_h2o:
            # Ghiorso & Sack 1995 treat water special by adding two terms:
            iw = self.component_index['H2O']
            xw = x[...,iw]
            mult         = np.ones_like(gamma) * (1-xw)[...,:]
            mult[...,iw] = xw
            gamma  *= mult
        if not return_gamma:
            activ = x*gamma
        else:
            activ = gamma    # Return the gamma instead of the activity
        if return_dict:
            actlist = activ.T
            activ   = {}
            for i,a in enumerate(actlist):
                name = self.compselect[i]
                if scalar:
                    a = a[0]
                activ[name] = a
        else:
            if scalar:
                activ = activ[0]
        return activ

    def get_chemical_potentials_of_liquid_components(self,x,return_dict=False):
        """
        Compute the chemical potentials of the liquid components from the Margules parameters
        and the mu0 values.

        Arguments:
        
          x      The x vector x[0:ncomponents] or array of vectors x[0:nx,0:ncomponents], such
                 that x.sum(axis=-1)==1.

          return_dict   If True, then instead of a list, return a
                        dict, so that it is clearer, which chemical potential
                        belongs to which component. NOTE: Some components
                        have weight 0.5: Mn2SiO4, Ni2SiO4 and Co2SiO4,
                        which the model uses in their half formula
                        units MnSi0.5O2, NiSi0.5O2 and CoSi0.5O2.

        Returns:

          mu     List or dict of the chemical potentials of the (selected) components of the liquid
        """
        activ = self.get_activities_of_liquid_components(x)
        if type(x) is list:
            x = np.array(x)
        if len(x.shape)==1:
            scalar = True
            x = np.stack([x])
        else:
            scalar = False
        ncomp = x.shape[-1]
        assert ncomp==len(self.ldb), 'Error in x: does not contain same amount of components as the liquid'
        RT    = self.Rgas*self.T
        mu    = np.array(self.ldb["DfG"])[...,:] + RT*np.log(activ)
        if return_dict:
            mulist = mu.T
            mu     = {}
            for i,m in enumerate(mulist):
                name = self.compselect[i]
                if scalar:
                    m = m[0]
                mu[name] = m
        else:
            if scalar:
                mu = mu[0]
        return mu
