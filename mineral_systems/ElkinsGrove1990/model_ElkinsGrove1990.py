#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                                June 2025
#
# The Elkins & Grove 1990 model for Feldspar in Na-K-Ca system
# Elkins, Linda T., Grove, Timothy L.
# Ternary feldspar experiments and thermodynamic models (1990)
# American Mineralogist 75, 544-559
#
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 1000)
import os
from phasehull import dissect_molecule,dissect_oxide,identify_component_minerals,Margules

class ElkinsGrove90(object):
    def __init__(self,compselect,T,P=1,path=None,ext=''):
        if path is None: path = os.path.dirname(__file__)
        self.T          = T
        self.P          = P
        self.components = ["CaAl2Si2O8","NaAlSi3O8","KAlSi3O8"]  #   Anorthite, Albite, Orthoclase
        self.compselect = compselect
        assert set(compselect)<=set(self.components), 'Error: The requested components are not all in this model.'
        WH,WS,WV        = self.get_Margules()
        self.margules   = Margules(self.components)
        self.margules.load_w(WH,WS,WV)
        self.margules.reduce_margules_to_subset(compselect)
        self.reset(T,P)

    def reset(self,T,P=1):
        self.T          = T
        self.P          = P

    def Gfunc(self,x):
        if len(x.shape)==1:
            x = np.array([x,])
        assert x.shape[-1]==len(self.compselect), 'Error: Dimension of x incorrect.'
        G = self.margules.compute_ideal_mixing_G(x,self.T) + self.margules.compute_interaction_G(x,self.T,self.P)
        return G

    def get_Margules(self):
        """
        Elkins & Grove 1990 Table 4. Data copied from the MELTS code by Ghiorso, the
        file FeldsparBerman.m.
        """
        whabor   = 18810.0  # joules     
        wsabor   = 10.3     # joules/K   
        wvabor   = 0.4602   # joules/bar 
        whorab   = 27320.0  # joules     
        wsorab   = 10.3     # joules/K   
        wvorab   = 0.3264   # joules/bar 
        whaban   = 7924.0   # joules     
        whanab   = 0.0      # joules     
        whoran   = 40317.0  # joules     
        whanor   = 38974.0  # joules     
        wvanor   = -0.1037  # joules/bar 
        whabanor = 12545.0  # joules     
        wvabanor = -1.095   # joules/bar 

        An = 0   # Anorthite CaAl2Si2O8 is the zeroth component
        Ab = 1   # Albite NaAlSi3O8 is the first component
        Or = 2   # Orthoclase KAlSi3O8 is the second component

        WH = np.zeros((3,3,3))
        WS = np.zeros((3,3,3))
        WV = np.zeros((3,3,3))
        
        WH[Ab,Or,Or]     = whabor
        WH[Or,Ab,Ab]     = whorab
        WH[Ab,An,An]     = whaban
        WH[An,Ab,Ab]     = whanab
        WH[An,Or,Or]     = whanor
        WH[Or,An,An]     = whoran
        WH[Or,Ab,An]     = ((whaban+whanab+whabor+whorab+whanor+whoran)/2+whabanor)

        WS[Ab,Or,Or]     = wsabor
        WS[Or,Ab,Ab]     = wsorab
        WS[Or,Ab,An]     = (wsabor+wsorab)/2

        WV[Ab,Or,Or]     = wvabor
        WV[Or,Ab,Ab]     = wvorab
        WV[An,Or,Or]     = wvanor
        WV[Or,Ab,An]     = ((wvabor+wvorab+wvanor)/2+wvabanor)

        return WH, WS, WV
