#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                                June 2025
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

#-----------------------------------------------------------------------
#                        Solids (crystals)
#-----------------------------------------------------------------------

def extract_from_mineral_database_based_on_components(mdb,components):
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
        if d['complete']:
            select.at[i,'ok']     = True
            select.at[i,'x']      = d['x']
            select.at[i,'moles']  = d['moles']
    select = select[select['ok']].copy().reset_index(drop=True).drop('ok',axis=1)
    return select

def identify_component_minerals(mdb,components):
    """
    After having determined the DfG and mfDfG values of the points in the mdb database
    using compute_DfG_with_mole_fraction_weighting(mdb,T), this function determines which
    of them are the true components with the lowest DfG.

    Arguments:

      mdb              The mineral database (see read_minerals_and_liquids())
      components       List of the formulae of the components, e.g. ['SiO2','MgO','Al2O3'].

    Returns:

      icomponents      List of integer indices of the mdb database pointing to the true
                       components. If the mdb database has different versions of the components
                       (e.g. alpha-quartz or beta-quartz for SiO2), then the version is
                       picked that has (for the given T) the smallest value of DfG.
      DfGcomponents    List of DfG values of these components.
      
    """
    icomponents   = np.zeros(len(components),dtype=int)
    DfGcomponents = np.zeros(len(components))+1e90
    for k,e in enumerate(components):
        ms = mdb[mdb['Formula']==e]
        assert(len(ms)>0), f'Error: Could not find component mineral {e} among minerals'
        DfG  = 1e90
        iend = -1
        for i,row in ms.iterrows():
            if(row['DfG']<DfG):
                iend = i
                DfG  = row['DfG']
        assert i>-1, f'Error: Could not find component mineral {e} among minerals (stranger version)'
        icomponents[k]   = iend
        DfGcomponents[k] = DfG
    return icomponents,DfGcomponents

def compute_re_level_G_in_database(mdb,components):
    """
    Like re_level_points(), but now done inplace within the database of
    solids, by creating a new column 'mfDfG00', and if 'mfDfGmass' is
    available, also a new column 'mfDfGmass00'. This is all just for
    convenience. In fact, this function will call re_level_points()
    internally.

    Arguments:

      mdb          The mineral database of solids

      components   List of components

    Returns:

      inplace new columns in the database

      icomponents     List of indices of the components
      DfGcomponents   Delta_f G of these components

    NOTE: The DfGcomponents should also be used if you want to
          re-level the liquid(s), because all substances should
          use the same relevelling energies.

    """
    icomponents,DfGcomponents = identify_component_minerals(mdb,components)
    pts            = np.stack(np.array(mdb['x']).copy())
    pts[:,-1]      = np.array(mdb['mfDfG']).copy()
    Gzero          = re_level_points(pts,icomponents,return_only_Gzero=True)
    mdb['mfDfG00'] = mdb['mfDfG'] - Gzero
    if('mfDfGmass' in mdb.columns):
        pts        = np.stack(np.array(mdb['xmass']).copy())
        pts[:,-1]  = np.array(mdb['mfDfGmass']).copy()
        Gzero      = re_level_points(pts,icomponents,return_only_Gzero=True)
        mdb['mfDfGmass00'] = mdb['mfDfGmass'] - Gzero
    return icomponents,DfGcomponents

#-----------------------------------------------------------------------
#                            Liquids
#-----------------------------------------------------------------------

def compute_re_level_G_of_fluid_continuum(x,G,zeroGs):
    """
    Like re_level_points(), but now with more simple arguments, more
    useable for fluid continua on an x grid. In fact, this function calls
    re_level_points() internally. Of course, you can do the re-leveling
    also after you combined the solid points and liquid continuum points
    into the same point set (when you compute the convex hull). But it
    can be more convenient to do it beforehand, when the points are not
    yet mixed. That is what this function is for. Convenience.

    Arguments:

      x            The mole or mass fractions. Must be array of shape [nx,N]
                   where nx is the number of points of x, and N is the
                   number of components.

      G            The G values to re-level

      zeroGs       For an N-component system, an array of N values of
                   the Gs at x = [1,0,0...], [0,1,...] etc to [0,...,1]
                   to which the G should be re-levelled.

    Returns:

      Gnew         The re-levelled G.
    """
    N          = x.shape[-1]
    nx         = x.shape[0]
    pts        = np.zeros_like(x)
    pts[:,:-1] = x[:,:-1]
    pts[:,-1]  = G
    Gzero      = re_level_points(pts,zeroGs=zeroGs,return_only_Gzero=True)
    return G-Gzero

def compute_mu_of_liquid_for_given_mineral_stoichiometry_and_activity_coefficient(ldb,formula,T,x,gamma):
    """
    Eq. 3 of Berman & Brown 1984 says that if a pure mineral solid is in equibrium with the liquid
    (which consists of the four basic components SiO2, MgO, CaO and Al2O3 in that paper), then
    the mu of the liquid, computed by summing the mu's of the components multiplied by the
    stoichiometry of the solid mineral, must be equal to that of the solid mineral. The
    computation of the right-hand-side of Eq. 3 is done here.

    Note that if one of the x goes to 0, the musum slowly goes to -infinity. But only if the
    nu (stoichiometry coefficient) of that x for that mineral is non-zero.

    Arguments:

      ldb           The database of liquid components
      formula       Example: liquid silica would be 'SiO2', in the database ldb it would be 'SiO2Liq'

    Returns:

      musum         The mu [J/mole] of the liquid mineral.
    """
    Rgas  = 8.314  # J/mol·K
    RT    = Rgas*T
    dis   = dissect_oxide(formula)
    nu    = dis['nu']
    unit  = dis['unit']
    musum = 0.
    for m in nu:
        mu0    = get_mu0_at_T(ldb,unit[m]+'Liq',T)
        assert m in x, f'Error: Need x[{m}] (the molar fraction of component [{unit[m]}]) for computing this.'
        assert m in gamma, f'Error: Need gamma[{m}] (the activity coefficient of component [{unit[m]}]) for computing this.'
        mu     = mu0 + RT * ( np.log(x[m]+1e-90) + np.log(gamma[m]) )
        musum += nu[m] * mu
    return musum

#-----------------------------------------------------------------------
#                          General functions
#-----------------------------------------------------------------------

def dissect_molecule(spec):
    """
    Returns the elements of which this molecule is made, their order, the total mass and the total charge.
    """
    theelements = {'H':1,'D':2,'He':4,'C':12,'N':14,'O':16,'S':32,'P':31,'Fe':56,'Si':28,'Na':23,'Mg':24,'Cl':35,'K':39,'F':19,'Al':27,'Ca':40,'Ti':48,'Cr':52,'Mn':55,'Co':59,'Ni':59}
    groups      = {'(OH)':17,'(H2O)':18,'(CO2)':44,'(CO3)':60,'(PO4)':95}
    if spec=='e-':
        return {'e':1},0,-1
    else:
        mol      = {}
        charge   = 0
        for g in groups:
            if g in spec:
                n     = 1
                lspec = spec.split(g)
                if len(lspec[-1])>1:
                    if(lspec[-1][:2].isdecimal()):
                        n=int(lspec[-1][:2])
                        lspec[-1]=lspec[-1][2:]
                    elif(lspec[-1][0].isdecimal()):
                        n=int(lspec[-1][0])
                        lspec[-1]=lspec[-1][1:]
                elif len(lspec[-1])>0:
                    if(lspec[-1][0].isdecimal()):
                        n=int(lspec[-1])
                        lspec[-1]=lspec[-1][1:]
                mol[g]=n
                spec = ''.join(lspec)
        cspec    = list(spec)
        if 'J' in cspec: cspec.remove('J')   # J means: on a grain surface
        while '+' in cspec:
            cspec.remove('+')
            charge += 1
        while '-' in cspec:
            cspec.remove('-')
            charge -= 1
        cspec_cp = cspec.copy()
        for i,c in enumerate(cspec_cp):
            if c.islower():
                cspec[i-1] = cspec[i-1]+c
                cspec[i]   = ''
        while '' in cspec: cspec.remove('')
        for i in range(len(cspec)-1):
            if cspec[i].isdecimal() and cspec[i+1].isdecimal():
                cspec[i]   = cspec[i]+cspec[i+1]
                cspec[i+1] = ''
                if i<len(cspec)-2:
                    if cspec[i+2].isdecimal():
                        cspec[i]   = cspec[i]+cspec[i+2]
                        cspec[i+2] = ''
        while '' in cspec: cspec.remove('')
        cspec_cp = cspec.copy()
        m = 0
        for i,c in enumerate(cspec_cp):
            if c.isdecimal():
                k=int(c)-1
                cspec[i+m]=''
                for l in range(k): cspec.insert(i+m,cspec_cp[i-1])
                m+=k
        while '' in cspec: cspec.remove('')
        elem = list(set(cspec))   # Get all elements of this molecule
        for e in elem:
            mol[e] = cspec.count(e)
        mass=0
        for e,n in mol.items():
            if e in theelements:
                mass+=n*theelements[e]
            elif e in groups:
                mass+=n*groups[e]
            else:
                raise ValueError('Weird error in dissect_molecule()')
        return mol,mass,charge

def dissect_oxide(formula,components=None,weights=None):
    """
    Dissect an oxide into its components.

    Example:
      dissect_oxide('MgSiO3',components=['SiO2','Al2O3','MgO'])
    will give x=[0.5, 0. , 0.5] and moles=0.5
      dissect_oxide('Mg2SiO4',components=['SiO2','Al2O3','MgO'])
    will give x=[0.33333333, 0.        , 0.66666667] and moles=0.33333333

    Arguments:

      formula        The formula for the mineral

      components     (optional) If set to a list of components, then it will compute
                     x (the mole fractions), moles (how many moles of this mineral
                     can be made with 1 mole of components)

      weights        (optional) If set to an array of length len(components)
                     containing only values 1, then nothing changes. But if
                     a value is, say, 0.5, then the corresponding component
                     is considered to be only half that formula unit. Example:
                     in the MELTS code one of the components is Mn2SiO4, but
                     it is in fact handled as MnSi0.5O2 internally. This changes
                     the molar fractions x. The weight would then be 0.5 for
                     this component, because in the liquids.fwf it is given as
                     Mn2SiO4 (with "Factor" column = weight = 0.5), while in the
                     computation of x we have to consider it to be MnSi0.5O2.

    Returns:

      answer         A dictionary with the results:

         nu          Dictionary of the stoichiometric moles of each primitive
                     unit (or element) for 1 mole of substance.
         unit        Dictionary of the primitive oxide units it consists of.
         nrox        Dictionary of the nr of O-atoms each unit represents
         unitelem    Reverse dictionary from primitive unit back to elements
    
      and if components has been passed in the arguments, the answer contains
      the following additional contents:
    
         x           The molar weight fractions of this oxide in terms of the
                     given components.
         xcomponents The list of end member names associated with the x vector
         nucomp      Same as x, but not normalized to 1. So it gives the
                     portions of each component (typically integer). Put in
                     another way: x = nucomp/nucomp.sum()
         moles       The number of moles of the dissected oxide one obtains
                     from one total mole of components. Example: dissecting
                     'Mg2Si2O6' into components=['Mg2SiO4','SiO2'] with
                     weights=[0.5,1] will give x=[0.667,0.333] and moles=0.333,
                     because 2 units of 'MgSi0.5O2' (note the weight=0.5) and
                     1 unit of SiO2 will give 1 unit of Mg2Si2O6 for three
                     total units of components, i.e. 1/3 unit of Mg2Si2O6 for
                     one unit of components. Put in another way:
                     moles=1/nucomp.sum()
         complete    If True, then the dissection into the given components
                     was successful. If False, then some parts are still left
                     and cannot be decomposed into the components.
         positive    Since the use of composite components could, concievably,
                     lead to negative mole fractions, this boolean tells if
                     all x are positive (True) or one or more is negative
                     (False). 
    
    """
    # List of primitive units we consider the oxide to be consisting of.
    # Each unit is a primitive oxide, i.e. O with a single other element.
    # Normally, each element is uniquely associated with a single unit.
    # Iron (Fe) is a special case, because iron can be in different
    # ionic states. We assume here that the number of O-atoms will uniquely
    # decide which ionic state Fe is in, if it is present. If Fe is not
    # present, the number of O atoms is uniquely determined by the other
    # elements. If the actual nr of O atoms is not equal to that value,
    # an error is given.
    
    # The lists below have to be extended to other elements
    # Note that Fe is treated separately (see below)
    formula_units_name = {'Fem':'Fe','Fe++':'FeO','Fe+++':'Fe2O3','Feh++':'Fe3O4','Mg':'MgO','Si':'SiO2','Ca':'CaO','Al':'Al2O3','Na':'Na2O','K':'K2O','Ti':'TiO2','Cr':'Cr2O3','Mn':'MnO','Co':'CoO','Ni':'NiO','(OH)':'H2O','(H2O)':'H2O','(CO2)':'CO2','(CO3)':'CO2','(PO4)':'P2O5'}
    formula_units_mult = {'Fem':1,   'Fe++':1,    'Fe+++':2,      'Feh++':3,      'Mg':1,    'Si':1,     'Ca':1,    'Al':2,      'Na':2,     'K':2    ,'Ti':1     ,'Cr':2      ,'Mn':1    ,'Co':1    ,'Ni':1    ,'(OH)':2    ,'(H2O)':1    ,'(CO2)':1    ,'(CO3)':1    ,'(PO4)':2     }
    formula_units_nrox = {'Fem':0,   'Fe++':1,    'Fe+++':3,      'Feh++':4,      'Mg':1,    'Si':2,     'Ca':1,    'Al':3,      'Na':1,     'K':1    ,'Ti':2     ,'Cr':3      ,'Mn':1    ,'Co':1    ,'Ni':1    ,'(OH)':-1   ,'(H2O)':0    ,'(CO2)':0    ,'(CO3)':-1   ,'(PO4)':-3    }
    formula_units_elem = {}
    for e in formula_units_name:
        formula_units_elem[formula_units_name[e]] = e
    nu   = {}
    unit = {}
    nrox = {}
    noxt = 0
    contains_iron = False
    if formula=='CO2':
        formula='(CO2)'
    if formula=='H2O':
        formula='(H2O)'
    mol,mass,charge    = dissect_molecule(formula)
    for m in mol:
        if m!='O':
            if m=='Fe':  # Iron can be in different oxidation states, so treat separate
                contains_iron = True
                Fenu    = mol[m]
            else:
                assert m in formula_units_mult.keys(), f'Error: Oxide has unknown component {m}'
                nu[m]   = mol[m]/formula_units_mult[m]
                unit[m] = formula_units_name[m]
                nrox[m] = formula_units_nrox[m]*nu[m]
                noxt   += nrox[m]
    if 'O' in mol:
        noxtexpect = mol['O']
    else:
        noxtexpect = 0
    if not contains_iron:
        assert noxt==noxtexpect, 'Error: Nr of oxygen atoms not correct.'
    else:
        # Check if iron is in FeO form (Fe++), in Fe2O3 form (Fe+++), in between Fe3O4 (Feh++), or metallic (Fem)
        nroxex  = noxtexpect-noxt
        nroxexp = (6*nroxex)//Fenu
        if(nroxexp==6):
            # FeO (ferrous oxide)
            m        = 'Fe++'
            nu[m]    = Fenu
            unit[m]  = 'FeO'
            nrox[m]  = 1
        elif(nroxexp==9):
            # Fe2O3 (ferric oxide)
            m        = 'Fe+++'
            nu[m]    = Fenu/2
            unit[m]  = 'Fe2O3'
            nrox[m]  = 3
        elif(nroxexp==8):
            # Fe3O4 (magnetite)
            m        = 'Feh++'
            nu[m]    = Fenu/3
            unit[m]  = 'Fe3O4'
            nrox[m]  = 4
        elif(nroxexp==0):
            # Fem (metallic iron)
            m        = 'Fem'
            nu[m]    = Fenu
            unit[m]  = 'Fem'
            nrox[m]  = 0
        else:
            raise ValueError(f'Error: Cannot identify oxidation state of iron in this formula: {formula}')
    unitelem = {}
    for u in unit:
        unitelem[unit[u]] = u
    answer = {'nu':nu,'unit':unit,'nrox':nrox,'unitelem':unitelem}

    # To be able to make e.g. ternary plots, we need to reconstruct the
    # relative mole fractions. The components keyword should be a list
    # of components (units).
    #
    # Note that components can be non-primitive. Primitive components
    # are e.g. MgO, SiO2, Na2O etc, which contain 1 and only 1 metal.
    # Non-primitive components are e.g. Mg2SiO4, Na2SiO3 etc. They
    # are, themselves, a combination (MgO)2SiO2 and (Na2O)SiO2,
    # respectively. That is why the code below is a bit complex.

    if components is not None:
        ncomp            = len(components)
        components_units = []
        components_elems = []
        if weights is None:
            weights = np.ones(ncomp)
        else:
            assert len(weights)==ncomp, 'Error: Weights not same length as component list'
        # Since we allow for composite components, we should dissect each one of
        # them, too, in order to be able to reconstruct their molar fractions in
        # the oxide we are dissecting.
        for comp in components:
            if comp in formula_units_elem:
                # Component is a unit itself
                components_units.append({comp:1})
                components_elems.append({comp:formula_units_elem[comp]})
            else:
                # Check if component is a combination of units
                diss    = dissect_oxide(comp)  # Recursive call: Dissect the composite component into primitive units
                em      = {}
                el      = {}
                for d in diss['nu']:
                    assert d in formula_units_name, f'Error: Component {comp} is not known and cannot be constructed from known units.'
                    em[formula_units_name[d]] = diss['nu'][d]
                    el[formula_units_name[d]] = d
                components_units.append(em)
                components_elems.append(el)

        # Make a list of all elements contained in the composite components
        elems = set([])
        for i in range(len(components_elems)):
            elems = elems.union(set(components_elems[i].values()))
        elems = list(elems)

        # Each element is uniquely associated to one primitive unit
        units = [formula_units_name[e] for e in elems]
        assert ncomp<=len(elems), 'Components are not linearly independent'

        # Check if we have a difficult situation, where the number of elements
        # (minus oxygen) is larger than the number of components. If that is
        # the case, the decomposition into components may still work, but it
        # is a lot more tricky, as we have to avoid accidental matrix degeneracy.
        if ncomp<len(elems):
            # Tricky situation, could lead to degenerate matrix even when
            # the decomposition is possible.
            # print('Warning: Fewer components than elements in components.')
            # Check that the first ncomp elements/units cover all components
            nelems = len(elems)
            indep  = False
            import itertools
            combis = list(itertools.combinations(np.arange(nelems),ncomp))
            for combi in combis:
                checkcomp = np.zeros(ncomp)
                for c in combi:
                    for i in range(ncomp):
                        if units[c] in components_units[i]:
                            checkcomp[i] = 1
                if np.all(checkcomp>0): indep = True
                if indep: break
            if not indep:
                raise ValueError('Error: Could not resolve degeneracy problem')
            # Now permutate the units and elements accordingly
            combi = list(combi)
            set(np.arange(nelems))-set(combi)
            combi = combi+list(set(np.arange(nelems))-set(combi))
            units = list(np.array(units)[list(combi)])
            elems = list(np.array(elems)[list(combi)])
            # Now hope this avoids a degenerate matrix...

        # Create the matrix that converts from vector of component moles to vector of element moles.
        matrix = np.zeros((ncomp,ncomp))    # Matrix[index_of_element,index_of_component]
        for i in range(ncomp):
            for k in range(ncomp):
                if units[k] in components_units[i]:
                    matrix[k,i]  = weights[i]*components_units[i][units[k]]

        # Invert this matrix to get the conversion from vector of element moles to vector of component moles
        matinv = scipy.linalg.inv(matrix)   # Matinv[index_of_component,index_of_element]
        vector = np.zeros(ncomp)
        for i in range(ncomp):
            e         = elems[i]
            if e in nu:
                vector[i] = nu[e]
        x = np.matmul(matinv,vector)
        nucomp = x.copy()
        xsum = x.sum()
        if(xsum>0):
            x /= xsum
            moles = 1/xsum
        else:
            moles = np.nan
        answer['x'] = x
        answer['xcomponents'] = components.copy()
        answer['nucomp'] = nucomp
        answer['moles'] = moles                    # If we have 1 mole in total of the components, how many moles of this oxide we get?

        # Check if the decomposition into components is complete
        complete = True
        ## Old method:
        #for e in list(nu):
        #    if e not in elems:
        #        complete = False
        # New method:
        reconstruct_units = {}
        for i,comp in enumerate(components_units):
            for u in comp:
                count = nucomp[i]*comp[u]
                if u in reconstruct_units:
                    reconstruct_units[u] += count
                else:
                    reconstruct_units[u]  = count
        actual_units = {}
        for u in unitelem:
            actual_units[u] = nu[unitelem[u]]
        nonpresent_units = list(set(reconstruct_units)-set(actual_units))
        for u in nonpresent_units:
            if reconstruct_units[u]!=0:
                complete = False
        for u in actual_units:
            if u not in reconstruct_units:
                complete = False
            else:
                if actual_units[u] != reconstruct_units[u]:
                    complete = False

        # Check if all x are positive
        positive = np.all(x>=0)

        # Finalize the answer and return
        answer['complete']         = complete
        answer['positive']         = positive
        answer['components']       = components
        answer['components_units'] = components_units
    return answer

#-----------------------------------------------------------------------
#                  Simplices, components, hyperplanes
#-----------------------------------------------------------------------

def convert_mole_fraction_into_mass_fraction(mdb,components,x,icomponents=None,return_also_mtot=False,inplace=False):
    """
    If you have a mole fraction x (such that x.sum(axis=-1)==1), or an array
    of them (again such that x.sum(axis=-1)==1, so x[...,:]), then you can
    convert them into mass fractions xm (again such that xm.sum(axis=-1)==1)
    with this function.

    Arguments;

      mdb              The mineral database (see read_minerals_and_liquids())
      components       List of the formulae of the components, e.g. ['SiO2','MgO','Al2O3'].
      x                Mole fraction x values. E.g. x = np.array([0.2,0.3,0.5]) or an
                       array of them, e.g. x = np.array([[0.2,0.3,0.5],[0.1,0.4,0.5]])

    Optional:

      icomponents      Indices (in the mdb database) of the component minerals,
                       so that these do not have to be first found in the database,
                       if you already know them. Just for speed-up.

      return_also_mtot If set to True, then also return mtot, so that you can
                       scale the Gibbs function with that (by dividing the
                       Gibbs function by mtot) to obtain the Gibbs per gram
                       instead.

      inplace          If True, then add columns xmass and mfDfGmass to the database mdb
                       and not return anything. Note: This option ignores the given
                       x-values and instead uses those of the database.

    Returns (if not inplace):

      xm               Array of the same dimension as x, but this time with the
                       mass fractions instead of the mole fractions.

      mtot             (if return_also_mtot==True) the mass of 1 mole of this mineral.

    If inplace:

      mdb updated with new columns 'xmass' and 'mfDfGmass'

    """
    if inplace:
        x = np.stack(np.array(mdb['x']))
    if icomponents is None:
        icomponents,DfGcomponents = identify_component_minerals(mdb,components)
    componmass = np.zeros(len(icomponents))
    for k in range(len(icomponents)):
        i               = icomponents[k]
        formula         = mdb.iloc[i]['Formula']
        mol,mass,charge = dissect_molecule(formula)
        componmass[k]   = mass
    if len(x.shape)==1:
        mtot = 0.
    else:
        mtot = np.zeros_like(x[...,0])
    for k in range(len(icomponents)):
        mtot += x[...,k]*componmass[k]
    xm = np.zeros_like(x)
    for k in range(len(icomponents)):
        xm[...,k] = x[...,k]*componmass[k]/mtot
    if inplace:
        mdb['xmass'] = xm.tolist()
        if 'mfDfG' in mdb.columns:
            mdb['mfDfGmass'] = mdb['mfDfG']/mtot
    else:
        if return_also_mtot:
            return xm,mtot
        else:
            return xm

def convert_mass_fraction_into_mole_fraction(mdb,components,xmass,icomponents=None,return_also_moltot=False):
    """
    The inverse of convert_mole_fraction_into_mass_fraction().

    If you have a mass fraction x (such that x.sum(axis=-1)==1), or an array
    of them (again such that x.sum(axis=-1)==1, so x[...,:]), then you can
    convert them into mole fractions xmol (again such that xmol.sum(axis=-1)==1)
    with this function.

    Arguments;

      mdb              The mineral database (see read_minerals_and_liquids())
      components       List of the formulae of the components, e.g. ['SiO2','MgO','Al2O3'].
      xmass            Mass fraction x values. E.g. x = np.array([0.2,0.3,0.5]) or an
                       array of them, e.g. x = np.array([[0.2,0.3,0.5],[0.1,0.4,0.5]])

    Optional:

      icomponents      Indices (in the mdb database) of the component minerals,
                       so that these do not have to be first found in the database,
                       if you already know them. Just for speed-up.

      return_also_moltot If set to True, then also return moltot, so that you can
                       scale the Gibbs function with that (by dividing the
                       Gibbs function by mtot) to obtain the Gibbs per gram
                       instead.

    Returns (if not inplace):

      xmol             Array of the same dimension as x, but this time with the
                       mole fractions instead of the mass fractions.

      moltot           (if return_also_moltot==True) the nr of moles of 1 g of this mineral.

    """
    if icomponents is None:
        icomponents,DfGcomponents = identify_component_minerals(mdb,components)
    componmol = np.zeros(len(icomponents))
    for k in range(len(icomponents)):
        i               = icomponents[k]
        formula         = mdb.iloc[i]['Formula']
        mol,mass,charge = dissect_molecule(formula)
        componmol[k]    = 1/mass
    if len(xmass.shape)==1:
        moltot = 0.
    else:
        moltot = np.zeros_like(xmass[...,0])
    for k in range(len(icomponents)):
        moltot += xmass[...,k]*componmol[k]
    xmol = np.zeros_like(xmass)
    for k in range(len(icomponents)):
        xmol[...,k] = xmass[...,k]*componmol[k]/moltot
    if return_also_moltot:
        return xmol,moltot
    else:
        return xmol
    
def interpolate_on_simplex(x,plane_x,plane_mfDfGs):
    """
    Suppose you have a substance at mole-fractional position x (an N-dimensional vector
    with sum_i x[i]=1) between N points with with known Gibbs DfG values. What is the
    linearly interpolated DfG value? The simplest example is a tie line between two
    minerals, in a binary case. For a ternary case, it is a tie-surface. For a quaternary
    case it is a tie-tetrad. In general it is a tie-simplex.

    Arguments:

      x[0:N]              Mole fractions of substance in terms of components. Sum should be 1.
      plane_x[0:N,0:N]    Mole fractions of N substances with known mfDfG values. Left index
                          is the index of the N substances. Right index is same as x[0:N],
                          where plane_x[:,:].sum(axis=-1)==1. That is: plane_x[i] is a vector
                          like x, summing to 1.
      plane_mfDfGs[0:N]   The mass-fraction-weighted Delta_f G values of all the points of
                          plane_x. The mass-fraction-weighted means, e.g., that with 0.333 mole
                          of SiO2 and 0.667 mole of MgO (in total 1 mole worth of components)
                          you can create 0.333 mole of Mg2SiO4. So mfDfG=0.333*DfG for Mg2SiO4
                          where DfG is the Delta_f G for 1 mole of Mg2SiO4.

    Returns:

      mfDfG               The interpolated value of mfDfG at this point on the simplex.
    """
    from scipy import linalg
    N = len(x)
    assert N==len(plane_x), 'Error in linear interpolation on a simplex: plane_x does not have right nr of points.'
    assert np.abs(x.sum()-1)<1e-6, 'Error in linear interpolation on a simplex: x does not sum to 1'
    for i in range(len(x)):
        assert N==len(plane_x[i]), 'Error in linear interpolation on a simplex: plane_x does not have right dimension.'
        assert np.abs(plane_x[i].sum()-1)<1e-6, f'Error in linear interpolation on a simplex: plane_x[{i}][:] does not sum to 1'
    evec = np.zeros((N-1,N-1))
    for i in range(N-1):
        evec[:,i] = plane_x[i][:-1]-plane_x[-1][:-1]
    y = linalg.solve(evec, (x[:-1]-plane_x[-1][:-1]))
    y = np.hstack((y,1-y.sum()))
    return (y*plane_mfDfGs).sum()

def remove_all_minerals_with_DfG_above_component_plane(mdb,components):
    """
    Once we calculated the DfG of all solid (fixed-composition) minerals, see the function
    compute_DfG_with_mole_fraction_weighting(mdb,T) above, we can reject all those that
    have a DfG above the DfG plane spanned by the components, as they are always unstable, because 
    they can always (at the very least) separate into phases with component composition.

    Arguments:

      mdb              The mineral database (see read_minerals_and_liquids())
      components       List of the formulae of the components, e.g. ['SiO2','MgO','Al2O3'].

    Returns:

      modifies the mdb database in-place.
    
    """
    nm          = len(mdb)
    nem         = len(components)
    iend,DfGend = identify_component_minerals(mdb,components)
    plane_x     = np.zeros((nem,nem))
    mdb['ok']   = True
    for iem in range(nem):
        plane_x[iem,:] = mdb.loc[iend[iem]]['x']
    for i,row in mdb.iterrows():
        if(i not in iend):
            x   = row['x']
            DfG = interpolate_on_simplex(x,plane_x,DfGend)
            if(row['mfDfG']>=DfG):
                mdb.at[i,'ok'] = False
    mdb = mdb[mdb['ok']].copy().reset_index(drop=True).drop('ok',axis=1)
    return mdb

def generate_grid_on_simplex(N,nx,return_integer_grid=False):
    """
    If we, for instance, want to create a heatmap on a ternary figure, we need
    a uniform grid in x[0], x[1], x[2], in which x.sum()==1. Since this grid is
    triangular and lives on a triangle (or more precisely in N-dimensions: on a
    simplex), we cannot simply create an N-dimensional array, because that would
    contain points that do not sum to 1. Here we generate a 1D array of x-points
    containing only those x-points which are summing up to 1.

    NOTE: If you want to make ternary plots in mass fraction (instead of mole
          fraction), you should first use generate_grid_on_simplex() to make
          a regular grid for xmass, then use convert_mass_fraction_into_mole_fraction()
          to convert that to xmole. Reason: The liquid stuff only works in
          mole fractions.

    Arguments:

      N       The dimension of the space, i.e., the number of components
      nx      The nr of x points in each dimension

    Returns:

      x       A 2D array x[0:ntot,N] where
                ntot = nx for N==2,
                ntot = (nx+1)*nx//2 for N==3,
                ntot = (nx+2)*(nx+1)*nx//2//3 for N==4,
                ntot = (nx+3)*(nx+2)*(nx+1)*nx//2//3//4 for N==5, etc.
              is the total number of points on the simplex.

      grid    Same as x, but in integer form (only if return_integer_grid==True)
    """
    assert N>=2, 'Error: Invalid N'
    assert nx>=2, 'Error: Invalid nx'
    ntot = 1
    for n in range(2,N+1):
        ntot *= (nx+n-2)
    for n in range(2,N+1):
        ntot = ntot//(n-1)

    def grid_recursive(N,iprev,nx):
        if type(iprev) is list: iprev=np.array(iprev,dtype=int)
        xlist  = []
        itot   = iprev.sum()
        imax   = nx-itot
        for k in range(imax):
            ll = np.array(list(iprev)+[k],dtype=int)
            if len(ll)==0: breakpoint()
            if len(iprev)>=N-2:
                xlist.append(ll)
            else:
                xlist += grid_recursive(N,ll,nx)
        return xlist

    grid = np.stack(grid_recursive(N,[],nx))
    assert grid.shape[0]==ntot, 'Error: Somehow the number of grid points is strange'

    x        = np.zeros((ntot,N))
    x[:,:-1] = grid[:,:]
    x[:,-1]  = nx-1-grid[:,:].sum(axis=-1)
    grid     = x.copy().astype(int)
    x       /= nx-1
    x[:,-1]  = 1-x[:,:-1].sum(axis=-1)  # Once more normalization

    if return_integer_grid:
        return x,grid
    else:
        return x

def make_dict_from_grid_G(grid,G):
    """
    The ternary library https://github.com/marcharper/python-ternary needs
    values on a dictionary with keys being tuples of integers e.g.
    d[(1,4,3)]=-1.394 etc. This is computed here from a grid of
    integers made with generate_grid_on_simplex() and the corresponding
    values G. 
    """
    scale = 0
    nxx = len(grid)
    d   = {}
    for i in range(nxx):
        t    = tuple(grid[i,:])
        scale = max(scale,grid[i,:].max())
        d[t] = G[i]
    return d,scale

def put_simplex_grid_back_onto_regular_grid(grid,Gfloat=None,index=None,usenan=True):
    """
    After you have done the Gibbs convex hull stuff to find the
    minimum Gibbs energy, it might be a bit hard to analyze the
    results, given that the results are on this triangular/simplex
    coordinate system, being a list of points instead of a
    regular grid. Most matplotlib functions do not work with non
    regular grids. So to put everything back on a regular x-grid,
    you can use this function. Note that the regular x grid will
    be (1) only in the first N-1 coordinates, because the Nth one
    is dependent on the N-1 others by x[-1]=1-x[:-1], and (2)
    will have many xpoints that are invalid, i.e., those that
    have sum>1. When we put the Gfloat and/or index values onto
    that grid, there will be many values that are NaN (for Gfloat)
    and -1 (for index).

    Arguments:

      grid      Array of integers grid[npts,N] with integers ranging
                from 0 to nx-1, where nx is the number of grid points
                in each direction.

    Optional:

      Gfloat    The floating points values (presumably Gibbs energy?)
                you want to put onto the regular grid. Each grid point
                should have a value on Gfloat.

      index     Same as Gfloat, but for an integer.

      usenan    If True, then set all G values at invalid points to NaN
    """
    N     = grid.shape[-1]
    nx    = grid.max()+1
    npts  = len(grid)
    x     = np.linspace(0,1,nx)
    G     = None
    idx   = None
    answer= [x]
    shape = list(nx*np.ones(N-1).astype(int))
    if Gfloat is not None:
        G = np.zeros(shape)
        if usenan:
            G[:] = np.nan
        for k in range(npts):
            i    = tuple(grid[k][:-1])
            G[i] = Gfloat[k]
        answer.append(G)
    if index is not None:
        idx = np.zeros(shape)-1
        for k in range(npts):
            i      = tuple(grid[k][:-1])
            idx[i] = index[k]
        answer.append(idx)
    return answer

def compute_DfG_with_mole_fraction_weighting(mdb,T,no_mfDfG=False):
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
        DfG                = get_mu0_at_T(mdb,row['Abbrev'],T)
        mdb.at[i,'DfG']    = DfG
        if not no_mfDfG:
            mdb.at[i,'mfDfG']  = DfG * row['moles']

#-----------------------------------------------------------------------
#              Points in (x_0,...,x_{N-2},DeltaG) space
#-----------------------------------------------------------------------

def convert_points_from_mole_fraction_to_mass_fraction(mdb,components,pts):
    """
    Binary, ternary and higher order phase diagrams are often plotted with
    mass fractions instead of mole fractions on the axes. Given a list of
    points pts where each p in pts is [x[0],x[1],...,x[N-2],mfDfG] with
    N being the order of the phase diagram (N=2 is binary, N=3 is ternary,
    etc) and x[N-1]=1-x[0]-...-x[N-2], this function will return a new
    list ptsm where each p in ptsm is [xm[0],xm[1],...,xm[N-2],mfDfG]
    where xm are the mass fractions.

    Arguments:
    
      mdb              The mineral database (see read_minerals_and_liquids())
      components       List of the formulae of the components, e.g. ['SiO2','MgO','Al2O3'].
      pts              The list of points

    Returns:

      ptsm             Array of points converted into mass fraction.

    """
    icomponents,DfGcomponents = identify_component_minerals(mdb,components)
    componmass = np.zeros(len(icomponents))
    for k in range(len(icomponents)):
        i               = icomponents[k]
        formula         = mdb.iloc[i]['Formula']
        mol,mass,charge = dissect_molecule(formula)
        componmass[k]   = mass
    if type(pts) is list:
        ptsar = np.stack(pts)
    else:
        ptsar = pts
    x        = np.zeros_like(ptsar)
    x[:,:-1] = ptsar[:,:-1]
    x[:,-1]  = 1-x[:,:-1].sum(axis=-1)
    xm,mtot  = convert_mole_fraction_into_mass_fraction(mdb,components,x,icomponents=icomponents,return_also_mtot=True)
    ptsm     = ptsar.copy()
    ptsm[:,:-1] = xm[:,:-1]        # Replace the mole fraction with mass fraction
    ptsm[:,-1]  = ptsm[:,-1]/mtot  # Also correct the Gibbs energy from per mole to per gram
    return ptsm

def convert_points_from_mass_fraction_to_mole_fraction(mdb,components,pts):
    """
    The inverse of convert_points_from_mole_fraction_to_mass_fraction(). This is
    necessary if you want to plot liquid quantities into ternary plots, because
    the liquid functions all need mole fractions instead of mass fractions, but
    if you start with a regular grid in mole fraction, it is no longer regular
    in mass fraction, so it won't plot correctly on a ternary plot. So if you
    want to plot ternary plots with liquids, you should first make a regular
    grid in xmass (mass fraction) using generate_grid_on_simplex(), then use
    convert_points_from_mass_fraction_to_mole_fraction() or the function
    convert_mass_fraction_into_mole_fraction() to convert these to xmol (mole
    fractions), which you can then feed into the liquid functions.

    Arguments:
    
      mdb              The mineral database (see read_minerals_and_liquids())
      components       List of the formulae of the components, e.g. ['SiO2','MgO','Al2O3'].
      pts              The list of points

    Returns:

      ptsm             Array of points converted into mass fraction.

    """
    icomponents,DfGcomponents = identify_component_minerals(mdb,components)
    if type(pts) is list:
        ptsar = np.stack(pts)
    else:
        ptsar = pts
    x        = np.zeros_like(ptsar)
    x[:,:-1] = ptsar[:,:-1]
    x[:,-1]  = 1-x[:,:-1].sum(axis=-1)
    xmol,moltot  = convert_mass_fraction_into_mole_fraction(mdb,components,x,icomponents=icomponents,return_also_moltot=True)
    ptsmol        = ptsar.copy()
    ptsmol[:,:-1] = xmol[:,:-1]        # Replace the mole fraction with mass fraction
    ptsmol[:,-1]  = ptsmol[:,-1]/moltot  # Also correct the Gibbs energy from per mole to per gram
    return ptsmol

#-----------------------------------------------------------------------
#                             Plotting
#-----------------------------------------------------------------------

def plot_1D_x_G_simplex(mdb,ldb,simplices,Gscale=1e3):
    colors = {'allcryst':'C1','liquid':'C0','cryst_1_liq_1':'C4','crystals':'C3'}
    linest = {'allcryst':None,'liquid':None,'cryst_1_liq_1':'--','crystals':'o'}
    for s in simplices:
        color = colors[s['stype']]
        lst   = linest[s['stype']]
        if lst is not None:
            plt.plot(s['x'][:,0],s['G'],lst,color=color)
            if 'cryst' in s['stype'] and 'liq' in s['stype']:
                for i in range(len(s['names'])):
                    if s['names'][i]=='liq':
                        plt.plot([s['x'][i][0]],[s['G'][i]],marker='o',fillstyle='none',color='black')
        else:
            plt.plot(s['x'][:,0],s['G'],color=color)
    for i,m in mdb.iterrows():
        plt.plot([m['x'][0]],[m['mfDfG00']/Gscale],'o',color=colors['crystals'])
    plt.xlabel('x')
    plt.ylabel('G')

def ternary_image(x0,x1,G,nxs=300):
    """
    Given a function G as a function of a 2D grid (x0,x1) on a ternary diagram,
    wich x2=1-x0-x1, given as a function of an array on a regular grid on (x0,x1),
    this will return an array Gint on a 2D image plane, where (x,y) are such that
    y = x1 * sqrt(3)/2 and x = x0 + x1/2.

    Arguments:

      x0     1D grid in zeroth fraction x coordinate
      x1     1D grid in first fraction x coordinate
      G      The tabular G function as a function of x0 and x1

    Option:

      nxs    The nr of pixels in x-direction

    Returns:

      Gint   The triangular image (= array of size nxs, nxs*sqrt(3)/2
    """
    from scipy.interpolate import RegularGridInterpolator
    nys         = int(nxs * np.sqrt(3)/2)
    xs          = np.linspace(0,1,nxs)
    ys          = np.linspace(0,1,nys)
    xpts        = np.zeros((nxs,nys,2))
    xpts[...,1] = ys[None,:] / (np.sqrt(3)/2)
    xpts[...,0] = xs[:,None] - xpts[...,1]/2
    interp      = RegularGridInterpolator((x0, x1), G, bounds_error=False)
    Gint        = interp(xpts)
    mask        = xpts[...,0] + xpts[...,1] > 1.00000001
    Gint[mask]  = np.nan
    return Gint

def plot_ternary_image_bitmap(x0,x1,G,nxs=300,scale=1.,yfact=1,tax=None,components=None):
    """
    Plot the ternary image ("heatmap"), using bitmap. Advantage: Fast.
    Disadvantage: Can have grainy edge near x0+x1=1.

    Arguments:

      x0       1D grid in zeroth fraction x coordinate
      x1       1D grid in first fraction x coordinate
      G        The tabular G function as a function of x0 and x1

    Option:

      nxs      The nr of pixels in x-direction
      scale    If 1, then axes are from 0 to 1, if 100, from 0 to 100
      yfact    Depending on your settings of the axis, the y-stretching
    """
    import ternary
    Gint = ternary_image(x0,x1,G,nxs=nxs)
    if tax is None:
        figure, tax = ternary.figure(scale=scale)
    tax.boundary(linewidth=1.0)
    tax.ax.imshow(Gint.T,extent=[0,scale,0,scale*yfact],origin='lower')
    if components:
        tax.right_corner_label(components[0])
        tax.top_corner_label(components[1])
        tax.left_corner_label(components[2])
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    tax.show()
    return tax

def plot_ternary_image(d,scale,components=None,cmap=None):
    """
    Plot the ternary image ("heatmap"), using the "ternary" package.
    Advantage: smooth. Disadvantage: Can be slow.

    Arguments:

      d        Dict obtained from make_dict_from_grid_G(grid,G)
      scale    Must be nx-1

    Option:

      components    Component names.
      cmap          Color map
    """
    import ternary
    figure, tax = ternary.figure(scale=scale)
    print('Making ternary heatmap')
    tax.heatmap(d, cmap=cmap)
    print('Finished ternary heatmap')
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple=scale//10, color="blue")
    #tax.ticks(axis='brl', clockwise=True, linewidth=1, multiple=scale//10)  # Somehow the axis does not work
    if components is not None:
        tax.right_corner_label(components[0])
        tax.top_corner_label(components[1])
        tax.left_corner_label(components[2])
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    tax.show()
    return tax

def ternary_fill(x,stype,scale=1,ax=None,done=False):
    colors   = {'allcryst':'C1','liquid':'C0','cryst_1_liq_1':'C4','cryst_1_liq_2':'C9','cryst_2_liq_1':'C6','crystals':'C3','inmisc_liquids':'cyan'}
    lincols  = {'allcryst':'C5','cryst_1_liq_2':'C0'}
    color    = colors[stype]
    if stype in lincols:
        lincol = lincols[stype]
    else:
        lincol = None
    y        = np.zeros((4,2))
    y[:-1,0] = x[:,0] + 0.5*x[:,1]
    y[:-1,1] = x[:,1]*np.sqrt(3)/2.
    y[-1,:]  = y[0,:]
    if ax is None:
        ax = plt.sca()
    if done:
        ax.fill(y[:,0]*scale,y[:,1]*scale,color=color)
    else:
        ax.fill(y[:,0]*scale,y[:,1]*scale,color=color,label=stype)
    if lincol is not None:
        ax.plot(y[:,0]*scale,y[:,1]*scale,color=lincol,linewidth=0.5)

def plot_ternary_phases(simplices,scale=1,components=None):
    import ternary
    ddone   = {'allcryst':False,'liquid':False,'cryst_1_liq_1':False,'cryst_1_liq_2':False,'cryst_2_liq_1':False,'crystals':False,'inmisc_liquids':False}
    #linest = {'allcryst':None,'liquid':None,'cryst_1_liq_1':'--','crystals':'o'}
    figure, tax = ternary.figure(scale=scale)
    print('Making ternary phases map')
    ax = tax.ax
    for isim in range(len(simplices['id'])):
        x     = simplices['x'][isim]
        ternary_fill(x,simplices['stype'][isim],scale=scale,ax=ax,done=ddone[simplices['stype'][isim]])
        ddone[simplices['stype'][isim]] = True
    print('Finished ternary phases map')
    if components is not None:
        tax.right_corner_label(components[0])
        tax.top_corner_label(components[1])
        tax.left_corner_label(components[2])
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    ax.legend()
    tax.show()
    return tax

#----------------------------------------------------------------------------------
#                                Plotting stuff
#----------------------------------------------------------------------------------

def extract_1d_cut_from_liquid(xstart,xend,Gfunc,nx=100):
    if type(xstart) is list: xstart=np.stack(xstart)
    if type(xend) is list: xend=np.stack(xend)
    s = np.linspace(0,1,nx)
    x = xstart[None,:] * (1-s[:,None]) + xend[None,:] * s[:,None]
    G = np.zeros(nx)
    for i in range(nx):
        G[i] = Gfunc(x[i])
    return s,x,G

def get_tie_lines_x_values_for_a_group(simplices,iselection,group,stride=1):
    x = []
    icnt = 0
    for i in group:
        isel = iselection[i]
        if icnt==0:
            x.append(simplices['xtieline'][isel])
        icnt += 1
        if icnt>=stride:
            icnt = 0
    x = np.stack(x)
    return x

def plot_tie_lines(xx,color=None):
    for x in xx:
        plt.plot(x[:,0]+0.5*x[:,1],(np.sqrt(3)/2)*x[:,1],'.-',color=color)

def plot_binodal_curve(xx,color=None):
    xl = xx[:,0,0]
    xr = xx[:,1,0][::-1]
    yl = xx[:,0,1]
    yr = xx[:,1,1][::-1]
    x  = np.hstack([xl,xr,[xl[0]]])
    y  = np.hstack([yl,yr,[yl[0]]])
    plt.plot(x+0.5*y,(np.sqrt(3)/2)*y,color=color)

def plot_simplex(simplices,isel,fillcolor=None,linecolor=None,thick=None):
    xx = simplices['x'][isel]
    xx = np.vstack((xx,xx[0,:]))
    x  = xx[:,0]+0.5*xx[:,1]
    y  = np.sqrt(3)/2*xx[:,1]
    if fillcolor is not None:
        plt.fill(x,y,color=fillcolor)
    if linecolor is not None:
        plt.plot(x,y,color=linecolor,linewidth=thick)


#----------------------------------------------------------------------------------
#                            Miscellaneous stuff
#----------------------------------------------------------------------------------
def latexify_chemical_formula(s):
    i = 0
    s = s+' '
    for k in range(1000):
        try:
            if s[i].isnumeric():
                s=s[:i]+r'$_{'+s[i:]
                i+=4
                while s[i].isnumeric(): i+=1
                s=s[:i]+r'}$'+s[i:]
                i+=2
        except:
            break
        i+=1
    s=s.strip()
    return s
