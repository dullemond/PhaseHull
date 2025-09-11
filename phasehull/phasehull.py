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
from scipy.spatial import ConvexHull

class CrystalDatabase(object):
    """
    Class for a database of the physical properties of fixed-composition phases, 
    i.e., crystals with a well-defined stoichiometry. Each of these will be a single
    point on the phase diagram.
    """
    def __init__(self,dbase,resetfunc=None):
        """
        Provide or read a database of fixed stoichiometry phases.

        Arguments:

          dbase        Either a string containing the name of the .csv or fixed-width-format
                       file containing the database, or a Pandas DataFrame of the database.

        Optional:

          resetfunc    A function with arguments (T,P), i.e. temperature (in Kelvin) and
                       pressure (in bar) that recomputes the Gibbs energy (in the mfDfG
                       column)
        """
        if type(dbase) is str:
            if dbase[-4:]=='.csv':
                dbase = pd.read_csv(dbase)
            elif dbase[-4:]=='.fwf':
                dbase = pd.read_fwf(dbase)
            else:
                raise ValueError(f'Do not know how to read {dbase}')
        elif type(dbase)!=pd.DataFrame:
            raise  ValueError(f'Error: dbase must be a pandas DataFrame')
        self.dbase  = dbase
        self.reset  = resetfunc

class Liquid(object):
    """
    Class for the physical properties of a liquid, or more general for a
    non-stoichiometric material (e.g., an alloy or a glas), which has a
    Gibbs energy for any value of fractional abundance vector x.
    """
    def __init__(self,name,endmembers,Gfunc,kwforGfunc=None,resetfunc=None):
        """
        Arguments:

          name         The name of this liquid or alloy. Can be as simple as e.g.,
                       'liquid' (for a melt) or 'solid' (for an alloy or glas).
                       Only important if one has more than one continuous
                       non-stoichiometric phases.

          endmembers   A list of names of the endmembers out of which this liquid
                       or alloy is composed. Important to assure the appropriate
                       order of the x-values when this class is used in other
                       parts of the code. For example, if endmembers =
                       ['MgO','SiO2','CaO'] and the fractional abundances are
                       given as [0.2,0.7,0.1], then it means that the liquid has
                       0.2 mole fraction of MgO, 0.7 mole fraction of SiO2 and
                       0.1 mole fraction of CaO. Particularly important if other
                       parts of the code use only a subset of these endmembers,
                       or a different order of them.

          Gfunc        A Python function for Gibbs(x) where x is either a 1D array
                       [x[0],x[1]...x[M-1]] with M = len(endmembers) or x is a
                       2D array x[N,M] with N is the number of sampling points
                       in the M-dimensional phase space. The function Gfunc must
                       be able to accept these 2D x arrays (i.e., 1D array of
                       arrays x[0:M]), for efficiency, so that a single call to
                       Gfunc returns a 1D array or G values at all x-points.

        Optional:

          kwforGfunc   A dictionary of possible keyword arguments for the Gfunc,
                       in case your function needs additional information to be
                       passed on.

          resetfunc    A function with arguments (T,P), i.e. temperature (in Kelvin) and
                       pressure (in bar) that reconfigures the Gfunc to compute
                       the Gibbs energy for the new T and P.
        """
        self.name         = name
        self.endmembers   = np.array(endmembers)
        self.Gfunc        = Gfunc
        self.kwforGfunc   = kwforGfunc
        self.reset        = resetfunc

    def call_Gfunc(self,endmem,x):
        """
        Interface to the Gibbs function provided by the user, allowing it to
        be used for a subset or different order of endmembers.

        Arguments:

          endmem       List of names of the endmembers in the correct order
                       in which they are in x. All endmem must be part of the
                       self.endmembers list.

          x            Array x[N,M] of mole fractions such that x.sum(axis=1)==1.
                       Here M is len(endmem), and N is the number of sampling
                       points.

        Returns:

          G            An array G[N] of Gibbs energy values.
        """
        assert set(endmem)<=set(self.endmembers), f'Error: Not all endmembers {endmem} are included in liquid {self.name}'
        if len(x.shape)==1:
            x = np.array([x,])
        assert len(x.shape)==2, 'Error: x must be an array x[N,M].'
        assert len(endmem)==x.shape[-1], 'Error: The number of endmem must equal the x.shape[1].'
        M   = len(self.endmembers)
        N   = x.shape[0]
        idx = []
        for e in endmem:
            idx.append(np.where(self.endmembers==e)[0][0])
        xx  = np.zeros((N,M))
        for i in range(len(endmem)):
            xx[:,idx[i]] = x[:,i]
        if self.kwforGfunc is not None:
            G   = self.Gfunc(xx,**self.kwforGfunc)
        else:
            G   = self.Gfunc(xx)
        return G

    def call_reset(self,T,P):
        """
        To be able to change temperature T or pressure P of this liquid, you have to provide
        a function called 'reset()'. If that function is part of an external class, and stores
        its information there, then all is fine. But what if you prefer to store the information
        (e.g., updated coefficients of the G(x) function model) in the Liquid class? The way to
        do this is to make your function reset() return a dictionary, that will then be stored
        in the Liquid class as self.kwforGfunc.

        The function call_reset() is a wrapper around your reset() function, allowing to 'catch'
        the output of your reset() function and store it in self.kwforGfunc. 
        """
        if self.reset is not None:
            self.kwforGfunc = self.reset(T,P)


class PhaseHull(object):
    """
    Class for the PhaseHull basic algorithm. This class performs the PhaseHull computations, and makes the
    basic classifications of the simplices. It does not contain any physical/chemical data. All that has
    to be provided as arguments to this class.
    """
    def __init__(self,endmembers,crystals=None,liquids=None,T=None,P=None,nres0=30,nrefine=4,nfact=2,nspan=2, \
                 min_nr_tielines=2,nocompute=False,incl_ptnames=False,incl_xvals=True,incl_Gvals=True,        \
                 incl_Gcen=False,incl_xcen=False,incl_xtie=False,mrcrit=10.):
        """
        Arguments:

          endmembers   List of names of the endmembers in the correct order
                       in which they are in the x fractional composition vector
                       to be used.

          crystals     A list of instances of the CrystalDataBase class. Usually just one.

          liquids      A list of instances of the Liquid class. Typically 1 (liquid) or 2 (liquid,solid).
                       Notice that liquid simply means: non-stoichiometric solution. It does not have
                       to be truly liquid. For instance, alloys are also continuous mixed phases, so they
                       would also be a 'liquid' in this sense.

          nres0        If liquids are present, nres0 determines the base grid resolution: it is the
                       number of grid spacings between 0 and 1.

          nrefine      If liquids are present, nrefine determines the nr of cycles of grid refinement.

          nfact        If liquids are present, nfact gives the refinement factor per refinement step.

          nspan        If liquids are present, nspan gives the width of the refinement zone around the
                       binodal curves.

          min_nr_tielines   Is a criterion for grouping of tie simplices.

          T            Temperature in Kelvin

          P            Pressure in bar

        There are a set of options for the kind of information to be stored for each simplex:
        
          incl_ptnames     Include the names of all the points of the simplex
          incl_xvals       Include the x-coordinates of all the points of the simplex
          incl_Gvals       Include the G values of all the points of the simplex
          incl_xcen        Include the x center of the simplex
          incl_Gcen        Include the G center of the simplex
          incl_xtie        For those simplices that are tie lines, include the tie line x values

        Calling PhaseHull will automatically start the computation of the phase diagram for the
        given temperature T and pressure P for which the crystals and liquid(s) are given. If a liquid
        or other non-stoichiometric material is included, then also a grid refinement is performed.
        This will create a set of every refined calculations. The corresponding results are all
        in the lists self.the<something> where <something> is e.g. pts, ids, simplices, hulls etc.
        The final (highest resulution) one is always the last one, which can be easiest obtained
        with the [-1] index, as in pts = self.thepoints[-1], and simplices = self.thesimplices[-1].

        Result:

          PhaseHull's product is a set of N simplices (= line elements in a binary system, triangles
          in a ternary system, tetrads in a quaternary system etc) that fill the entire phase space
          in the x-coordinates (= the region between 0 and 1 in a binary system, the triangle between
          [1,0,0], [0,1,0] and [0,0,1] for a ternary system etc). The number of simplices N can be
          any number >= 1, and tends to be larger, the more complex the phase diagram is. The
          information about this set of simplices is located in PhaseHull.thesimplices[-1]. So you
          have

            phull       = PhaseHull(....)
            simplices   = phull.thesimplices[-1]

          This is a dictionary containing a set of arrays/lists of length N (one value for each simplex).
          These arrays or lists are:

            simplices['x']          np.array((N,nendm,nendm)) where nendm is the number of endmembers, 
                                    such that simplices['x'].sum(axis=-1)==1. The last index is 
                                    therefore the index of the coordinates. The middle index counts
                                    the corners of the simplex, and the first index is, as said above,
                                    the index counting the simplices.
            simplices['stype']      The list of physical types of the simplices. E.g. if
                                    simplices['stype'][5]=='cryst_1_liq_1', then simplex 5 is of the
                                    type of a coexistence of one crystals with one liquid.
            simplices['id']         Just in case the order of the N simplices gets mixed up, this is
                                    the original integer index of the simplices.
            simplices['id_qhull']   The index of the simplex as it was in the Qhull results. Note that
                                    not all Qhull simplices are included here, because Qhull also
                                    returned the simplices at the top (rather than the bottom) of the
                                    convex hull.
            simplices['ipts']       np.array((N,nendm)) of the indices of the points at the corners. 
                                    The coordinates of the points can be found in PhaseHull.thepoints[-1].
            simplices['G']          np.array((N,nendm)) of the G values at the corners. Only if
                                    incl_Gvals is set.
            simplices['neighbors']  np.array((N,nendm)) of the indices of the neighboring simplices.
                                    A neighbor shares a face with another simplex.

          Sometimes there may be fewer or more contents than these. With simplices.keys() you can
          get a list of all the contents of this dictionary.

        In cases with a large number of simplices (e.g., when you have one or more liquids), it might
        become a bit difficult to get an overview. Here are some methods to get a better overview:

          PhaseHull.select_simplices_of_a_given_kind(stype):
                   Returns the indices (for the above simplices arrays) of all simplices of the
                   kind stype, where stype is a string such as e.g., 'allcryst' or 'cryst-1-liq-1'.

          PhaseHull.get_tie_lines_simplices():
                   Returns the indices of all simplices that are tie lines.

          PhaseHull.get_tie_line_groups():
                   Returns a list of groups of tie lines (for ternary or larger systems). This is
                   useful because in ternary or larger systems, tie lines are usually in bundles.

          ...more to come...

        If you want to redo the computation for a different temperature or pressure, you can call:

          PhaseHull.reset(T,P)

        which then also automatically starts the computation, unless you set nocompute=True as keyword.
        """
        self.mrcrit       = mrcrit
        self.endmembers   = endmembers
        self.nendmembers  = len(endmembers)
        #self.tieline_types= ['inmisc_liquids','cryst_1_liq_1','cryst_1_liq_2','cryst_1_liq_3']
        self.tieline_types= ['tieline_c0l2','tieline_c0l3','tieline_c1l2','tieline_c1l3']
        if self.nendmembers>3:
            print('Note: At this moment the tie line identification has not been tested for nr of endmembers>3.')
        if crystals is not None:
            if type(crystals) is not list:
                crystals      = [crystals]
        else:
            crystals      = []
        self.crystals     = crystals
        if liquids is not None:
            if type(liquids) is not list:
                liquids       = [liquids]
        else:
            liquids       = []
        self.liquids      = liquids
        self.T            = T
        self.P            = P
        if T is not None or P is not None:
            for c in self.crystals:
                c.reset(T,P)
            for l in self.liquids:
                l.call_reset(T,P)
        self.nres0        = nres0
        self.nrefine      = nrefine
        self.nfact        = nfact
        self.nspan        = nspan
        self.min_nr_tielines = min_nr_tielines
        self.incl_ptnames = incl_ptnames
        self.incl_xvals   = incl_xvals
        self.incl_Gvals   = incl_Gvals
        self.incl_Gcen    = incl_Gcen
        self.incl_xcen    = incl_xcen
        self.incl_xtie    = incl_xtie
        if not nocompute:
            self.compute()

    def reset(self,T,P=1,nocompute=False):
        """
        PhaseHull is designed to create a phase diagram for a given (fixed)
        temperature T [Kelvin] and pressure P [bar], in the space of x (the
        composition given by fractional abundances). To reset to a different
        temperature and pressure, use this function.

        Arguments:

          T       Temperature in Kelvin

          P       Pressure in bar
        
        """
        self.T = T
        self.P = P
        for c in self.crystals:
            c.reset(T,P)
        for l in self.liquids:
            l.call_reset(T,P)
        if not nocompute:
            self.compute()

    def compute(self):
        """
        This runs all the basic computations for the phase diagram.
        It is the core of PhaseHull.
        """
        self.setup_base_grid()
        self.thehulls     = []
        self.thesimplices = []
        self.do_all_refinement_steps()
        self.get_tie_lines_simplices()
        if self.nendmembers==3:
            self.get_tie_line_groups()
        if len(self.crystals)>0:
            self.find_stable_crystals()

    def create_index_for_the_simplices(self):
        """
        If you know the index of a point, or pair of point indices, or triple of points,
        which make up a corner, ribbon, wall etc of a simplex of the convex hull, you
        can find the simplex back using this index. For instance, to find all simplices
        that contain point 6:

          isims = self.index_for_simplices[-1][1][6]

        provided, of course, that point 6 is indeed on the convex hull. Let us start with
        a given simplex with index isim=10.

          import phasehull as ph
          phull = ph.PhaseHull(....fill in here....)
          simplices = phull.thesimplices[-1]      # Get the highest resolution grid
          phull.create_index_for_the_simplices()  # Create the index
          isim = 10
          ipts = list(simplices['ipts'][isim])
          ipts.sort()
          isims_points = phull.index_for_simplices[-1][1][ipts[0]]            # Only for point 0
          isims_walls  = phull.index_for_simplices[-1][2][(ipts[0],ipts[1])]  # Only for the pair points 0 and 1

        isims_points are the indices of all simplices that contain point 0.
        isims_walls are the indices of all simplices that contain points 0 and 1.
        """
        self.index_for_simplices = []
        for ilevel in range(len(self.thesimplices)):  # The refinement levels
            simplices = self.thesimplices[ilevel]
            indices   = {}
            for idim in range(1,self.nendmembers+1):    # The indices based on points, line ribbons, triangle walls, tetrads etc.
                indices[idim] = {}
                for isim in range(len(simplices['id'])):
                    ipts = simplices['ipts']
                    if idim<self.nendmembers:
                        walls = self.get_walls(ipts[isim],depth=self.nendmembers-idim)
                        for index in walls:
                            if index in indices[idim]:
                                indices[idim][index].append(isim)
                            else:
                                indices[idim][index] = [isim]
                    else:
                        index = list(ipts[isim])
                        if len(index)>1:
                            index.sort()
                            index = tuple(index)
                        else:
                            index = index[0]
                        indices[idim][index] = isim
        self.index_for_simplices.append(indices)

    def add_to_index(self,indices,index,isim):
        if type(index) is tuple:
            if len(index)==1:
                index = index[0]  # No tuples of one element
        if index in indices:
            indices[index].append(isim)
        else:
            indices[index] = [isim]

    def get_walls(self,ipts,depth=1):
        walls = []
        for i in range(len(ipts)):
            w = list(ipts).copy()
            w.remove(w[i])
            w.sort()
            if depth>1:
                ww = self.get_walls(w,depth=depth-1)
                for w in ww:
                    walls.append(w)
            else:
                w = tuple(w)
                if len(w)==1:
                    w = w[0]      # No tuples of one element
                walls.append(w)
        walls = list(set(walls))
        walls.sort()
        return walls

    def select_simplices_of_a_given_kind(self,stype):
        """
        The product of PhaseHull is a set of simplices (in a binary model = line elements,
        in a ternary model = triangles, in a quaternary model = tetrad) describing the
        bottom of the convex hull. PhaseHull will try to classify the physical meaning of
        each of these simplices. For instance, a tiny simplex is likely part of a continuous
        liquid; if the analytic G(x) lies below that simplex, then this simplex is clearly
        a discrete part of the liquid. A very elongated simplex, in particular when the
        analytic G(x) lies above it, is very likely a tie line. Currently implemented
        type (called 'stype') are:

           liquid            Fully liquid
           allcryst          A coexistence of 2 (for binary) or 3 (for 
                             ternary) (etc) crystals.
           cryst_1_liq_1     A coexistence of 1 crystal phase with
                             1 liquid phase (only for binary diagrams).
           cryst_2_liq_1     A coexistence of 2 crystal phases with
                             1 liquid phase (only for ternary diagrams).
           cryst_1_liq_2     A coexistence of 1 crystal phase with
                             2 liquid phases (only for ternary diagrams),
                             but not a tie line.
           tieline_c1l2      Like cryst_1_liq_2, but with the physical
                             meaning of a tie line between 1 crystal and
                             1 liquid point (only for ternary diagrams).
                             This is a very elongated simplex. Its
                             meaning is that of a binary phase line 
                             in a ternary diagram. The two fluid points
                             are actually a single fluid, but it becomes 
                             two very-nearby fluid points due to the
                             discretization.
           tieline_c0l3      A coexistence of two liquid phases that
                             are inmiscible  (only for ternary diagrams).
                             In reality 3 liquids are connected, two
                             being very close together.
           inmisc_liquids_3phase   Like 'inmisc_liquids' but for
                                   three phases. In a ternary diagram
                                   these occasionally show up as big
                                   single triangles.
    
        But in the near future more will likely be implemented. If you want to
        select all simplices of one of these stypes, you can use this method.

        Arguments:

          stype   The type you wish to select.

        Returns:

          isel    A list of indices of the simplices selected.

        Usage of isel, for example:

          simplices = self.thesimplices[-1]
          x         = simplices['x'][isel]
        
        """
        simplices  = self.thesimplices[-1]
        nsim       = len(simplices['id'])
        nendm      = simplices['x'][0].shape[-1]
        iselection = []
        simindices = set()
        simplices['neighbors_'+stype] = [None for _ in range(nsim)]
        for isim in range(nsim):
            if simplices['stype'][isim]==stype:
                iselection.append(isim)
                simindices.add(isim)
        for isel in iselection:
            idxnew = []
            for i in range(len(simplices['neighbors'][isel])):
                k = simplices['neighbors'][isel][i]
                if k in simindices:
                    idxnew.append(k)
            simplices['neighbors_'+stype][isel] = idxnew
        return iselection

    def check_if_two_neighboring_tie_lines_are_parallel(self,i,k):
        # If they are parallel, they should share mutuall contain each others peak vertex
        simplices  = self.thesimplices[-1]
        assert 'tiepeak' in simplices, 'Error: Cannot check parallel neighbors without tiepeak info'
        ipt_peak_i = simplices['ipts'][i][simplices['tiepeak'][i]]
        ipt_peak_k = simplices['ipts'][k][simplices['tiepeak'][k]]
        return ipt_peak_i in simplices['ipts'][k] and ipt_peak_k in simplices['ipts'][i]

    def find_stable_crystals(self):
        """
        Once the convex hull has been computed, we can figure out which of the original crystals
        is stable, i.e. lies on the bottom of the convex hull. This will add a column 'stable' to the
        crystal database with True (stable) or False (unstable).
        """
        self.find_crystals_among_points()
        cdb = self.crystals[-1].dbase
        cdb['stable'] = False
        for i,row in cdb.iterrows():
            if row['ipt'] in self.thehulls[-1].vertices:
                cdb.at[i,'stable'] = True

    def find_crystals_among_points(self):
        """
        Before starting the convex hull algorithm, all points, liquid or crystal or other types of
        phases, are joined into a single big point cloud, self.thepoints[-1]. It can be useful to
        know which of these points correspond to which crystals. This function adds a column to
        the crystal database with the point indices.
        """
        ids = np.array(self.theids[-1])
        cdb = self.crystals[-1].dbase
        cdb['ipt'] = -1
        for i,row in cdb.iterrows():
            w               = np.where(ids==row['Abbrev'])[0]
            if len(w)==1:
                cdb.at[i,'ipt'] = w[0]
            elif len(w)==0:
                cdb.at[i,'ipt'] = -1   # Signalling: This mineral is not in the point cloud
            else:
                cdb.at[i,'ipt'] = -2   # Signalling: This mineral has multiple points

    def compute_tie_lines_from_simplices(self,iselection):
        """
        In PhaseHull tie lines are, in fact, very narrow tie simplices. Their width is of
        the order of the finest grid resolution, while their length can be anything. In
        a ternary diagram (2D) they are needle-like triangles. The 'blunt' side (the 'tie base')
        consists (in 2D) of two points, while the 'sharp' side (the 'tie peak') has only one 
        point. The tie line it represents is best approximated by the line going from the
        tie peak to the middle of the tie base.

        The present function takes the indices of all simplices that have been identified
        as being tie lines. It will then, for each of them, compute the best approximated
        tie line simplices['xtieline'], and identify the 'tiepeak' and the 'tiebase'.
        """
        nsel      = len(iselection)
        simplices = self.thesimplices[-1]
        if 'xtieline' not in simplices:
            simplices['xtieline'] = [None for _ in range(len(simplices['id']))]
        if 'tiepeak' not in simplices:
            simplices['tiepeak'] = [None for _ in range(len(simplices['id']))]
        if 'tiebase' not in simplices:
            simplices['tiebase'] = [None for _ in range(len(simplices['id']))]
        for isel,isim in enumerate(iselection):
            lmin = []
            for i,x in enumerate(simplices['x'][isim]):
                l = ((x[None,:]-simplices['x'][isim])**2).sum(axis=-1)
                l[l==0] = 1e99
                lmin.append(l.min())
            lmin  = np.array(lmin)
            ipeak = lmin.argmax()
            xpeak = simplices['x'][isim][ipeak]
            xav   = np.zeros(len(x))
            n     = 0
            for i,x in enumerate(simplices['x'][isim]):
                if i!=ipeak:
                    xav += x
                    n += 1
            xav /= n
            ibase = []
            for i,ipt in enumerate(simplices['ipts'][isim]):
                if i!=ipeak:
                    ibase.append(ipt)
            simplices['xtieline'][isim] = np.stack([xpeak,xav])
            simplices['tiepeak'][isim] = ipeak
            simplices['tiebase'][isim] = np.stack(ibase)

    def remove_false_tie_line_neighbors(self,iselection,stype):
        """
        Sometimes Qhull creates tie line simplices that are joined by their short side (the
        two liquid points for a ternary system), instead by their long sides. This is rare,
        but it can happen when two tie line fans touch at one location. This leads to the
        tie line grouping algorithm to falsely group both tie line fans into the same group.
        So we should try to identify these joins, and remove them from the tie line neighbor
        list (not from the full neighbor list).
        """
        simplices = self.thesimplices[-1]
        for isel,isim in enumerate(iselection):
            for k in simplices['neighbors_'+stype][isim]:
                if(not self.check_if_two_neighboring_tie_lines_are_parallel(isim,k)):
                    print(f'Removing false tie line neighbors {isim} {k}')
                    simplices['neighbors_'+stype][isim].remove(k)
                    if isim in simplices['neighbors_'+stype][k]:
                        simplices['neighbors_'+stype][k].remove(isim)

    def get_tie_lines_simplices(self,stypes=None):
        """
        From the convex hull result, identify all simplices that are, physically, tie lines.
        These are then gathered in the self.tie_line_simplices dictionary, where the keys are
        the various types of tie lines. By self.tie_line_simplices.keys() you can get a list of
        all present types.
        """
        if stypes is None:
            stypes = self.tieline_types
        self.tie_line_simplices = {}
        for stype in stypes:
            dummy = self.select_simplices_of_a_given_kind(stype)
            if len(dummy)>0:
                self.tie_line_simplices[stype] = dummy
                self.compute_tie_lines_from_simplices(self.tie_line_simplices[stype])
                self.remove_false_tie_line_neighbors(self.tie_line_simplices[stype],stype)

    def get_tie_line_groups(self):
        """
        Tie lines, at least those that connect to a liquid, usually come in bundles of nearly parallel
        ones. They can form multiple distinct groups. This function tries to identify all groups
        and properly identify the group members. This can be done because, in PhaseHull, these
        tie lines are, in fact, tie simplices with a non-zero width. The neighboring tie simplices
        will thus be exact neighbors, which can be identified as such. This is done with the
        function self.sort_tie_lines().
        """
        assert self.nendmembers==3, 'Error: get_tie_line_groups() only works for ternary diagrams.'
        self.tie_line_groups       = {}
        for stype in self.tie_line_simplices:
            self.tie_line_groups[stype] = []
            groups = self.sort_tie_lines(self.tie_line_simplices[stype],stype)
            for g in groups:
                if len(g)>=self.min_nr_tielines:
                    self.tie_line_groups[stype].append(g)

    def get_x_values_of_tie_lines(self,stypes=None,stride=1):
        if stypes is None:
            stypes = self.tieline_types
        assert self.nendmembers==3, 'Error: get_x_values_of_tie_lines() only works for ternary diagrams.'
        xx = {}
        for stype in stypes:
            if stype in self.tie_line_groups:
                xx[stype] = []
                for i,g in enumerate(self.tie_line_groups[stype]):
                    x = self.get_tie_lines_x_values_for_a_group(g,stride=stride)
                    xx[stype].append(x)
        return xx

    def get_x_values_of_binodal_curves(self,stypes=None,stride=1,fullx=False):
        """
        Binodal curves in ternary plots are the curves created from the endpoints of tie lines in, e.g.,
        a liquid with an inmiscibility region. This function can also get the endpoints of tie lines
        starting at fixed-composition points. This is then called the liquidus.

        Arguments:

           stypes     The simplex types for which to compute these endpoints. Default is the
                      types listed in self.tieline_types.

           stride     To reduce the output data volume, you can set stride to a value >1, meaning
                      that only every other `stride` point will be returned.

           fullx      If True, it will return not just the 2D coordinates of these points, but also
                      the third (dependent) coordinate which is 1-minus-the other two coordinates.
        """
        if stypes is None:
            stypes = self.tieline_types
        assert self.nendmembers==3, 'Error: get_x_values_of_binodal_curves() only works for ternary diagrams.'
        xb = {}
        for stype in stypes:
            if stype in self.tie_line_groups:
                xb[stype] = []
                for g in self.tie_line_groups[stype]:
                    xx = self.get_tie_lines_x_values_for_a_group(g,stride=stride)
                    xl = xx[:,0,0]
                    xr = xx[:,1,0][::-1]
                    yl = xx[:,0,1]
                    yr = xx[:,1,1][::-1]
                    x  = np.hstack([xl,xr,[xl[0]]])
                    y  = np.hstack([yl,yr,[yl[0]]])
                    if fullx:
                        z = 1-x-y
                        xb[stype].append(np.stack([x,y,z]).T)
                    else:
                        xb[stype].append(np.stack([x,y]).T)
        return xb

    def get_tie_lines_x_values_for_a_group(self,group,stride=1):
        """
        A group is a group of tie line simplices that belong together. In 2D ternary diagrams
        these tie lines typically form "fans" of near-parallel tie lines, sometimes diverging
        from a point (a crystal) but not always (if they are e.g. binodals of an inmiscibility
        region). This function returns the endpoints of these tie lines (both sides).
        
        Arguments:

           group      The group of tie lines that belong together (are adjacent to each other
                      in the phase diagram). 
        
           stride     To reduce the output data volume, you can set stride to a value >1, meaning
                      that only every other `stride` point will be returned.

        Returns:

           x          Array of x coordinates of the endpoints of this set of tie lines.
        """
        simplices = self.thesimplices[-1]
        x = []
        icnt = 0
        for i in group:
            isel = i
            if icnt==0:
                xtl = simplices['xtieline'][isel]
                assert xtl is not None, f'Error: Tie line for simplex {i} is None...'
                x.append(xtl)
            icnt += 1
            if icnt>=stride:
                icnt = 0
        x = np.stack(x)
        return x

    def get_ipts_liquidus_for_one_group_2d(self,stype,igroup):
        """
        Get the point indices (in the convex hull point cloud) of the points that
        lie on the liquidus, i.e., the end point of tie lines.

        Arguments:

           stype      The simplex type for which to compute this. For the liquidus
                      in a ternary (2D) plot belonging to (i.e. starting from) a
                      certain crystal (fixed composition solid), stype should be
                      'tieline_c1l2'.

           igroup     The index of the tie line group for which to collect these.

        Returns:

           ipts       The indices of the end points of these tie lines along the
                      liquidus.
        """
        simplices = self.thesimplices[-1]
        isims     = self.tie_line_groups[stype][igroup]
        ibases    = []
        for isim in isims:
            ibases.append(tuple(simplices['tiebase'][isim]))
        chains    = self.get_chains(ibases)
        iptss     = []
        for ch in chains:
            c    = np.stack(ch)
            ipts = list([c[0,0]])+list(c[:,1])
            iptss.append(ipts)
        return iptss

    def get_x_liquidus_for_one_group_2d(self,stype,igroup,smooth=False):
        """
        Get the point x locations (in the convex hull point cloud) of the points that
        lie on the liquidus, i.e., the end point of tie lines.

        Arguments:

           stype      The simplex type for which to compute this. For the liquidus
                      in a ternary (2D) plot belonging to (i.e. starting from) a
                      certain crystal (fixed composition solid), stype should be
                      'tieline_c1l2'.

           igroup     The index of the tie line group for which to collect these.

           smooth     Since the liquidus points are the points of the short edge
                      of the "needle-like" simplices, these can be a bit jagged,
                      like a saw. By setting smooth, this alternating back-and-forth
                      is averaged out.

        Returns:

           xx         Array of x-coordinates of the end points of these tie lines 
                      along the liquidus.
        """
        xpts  = self.thepoints[-1]
        iptss = self.get_ipts_liquidus_for_one_group_2d(stype,igroup)
        xx    = []
        for ipts in iptss:
            if smooth:
                x        = np.zeros((len(ipts)+1,self.nendmembers))
                x[0,:-1] = xpts[ipts[0]][:-1]
                for i in range(len(ipts)-1):
                    x[i+1,:-1] = 0.5 * ( xpts[ipts[i]][:-1] + xpts[ipts[i+1]][:-1] )
                x[-1,:-1] = xpts[ipts[-1]][:-1]
                x[:,-1] = 1-x[:,:-1].sum(axis=-1)
            else:
                x     = np.zeros((len(ipts),self.nendmembers))
                for i in range(len(ipts)):
                    x[i,:-1] = xpts[ipts[i]][:-1]
                x[:,-1] = 1-x[:,:-1].sum(axis=-1)
            xx.append(x)
        return xx

    def complete_x(self,x):
        """
        If x is (or maybe is) only the first n-1 elements where n is the number of endmembers,
        then this function will return a new x that contains the complete set of endmembers
        that sums up to 1. Note that x can either be one vector of an array of vectors.

        Arguments:

          x      The x vector x[0:nendmembers-1] or array of vectors x[0:nx,0:nendmembers-1]

        Returns:

          x      The x vector x[0:nendmembers] or array of vectors x[0:nx,0:nendmembers], such
                 that x.sum(axis=-1)==1.
        """
        if type(x) is list: x=np.array(x)
        if x.shape[-1]==self.nendmembers-1:
            if len(x.shape)==2:
                nx   = x.shape[0]
            elif len(x.shape)==1:
                nx   = 1
            else:
                raise ValueError('Error: Dimensions of x are somehow wrong. Must be a 1D array of x vectors.')
            xnew = np.zeros((nx,self.nendmembers))
            xnew[:,:-1] = x[:,:]
            xnew[:,-1]  = 1-x.sum(axis=-1)
            x = xnew
        return x

    def polygon_from_simplex(self,isim):
        """
        For plotting of the simplices in a ternary diagram you often need the top, left, right x
        values for a full closed polygon. This is returned here.
        """
        simplices = self.thesimplices[-1]
        x = simplices['x'][isim]
        x = np.vstack((x,x[0,:]))
        t = x[:,0]
        l = x[:,1]
        r = x[:,2]
        return t,l,r

    def get_liquidus_ipts_and_x_2d(self):
        """
        In a ternary diagram (2D) the liquidus is the closed curve at the edge of
        the domain(s) of the purely liquid phase. The edge is then usually either
        the edge of the diagram or the start/end of a set of tie lines.

        This function returns two results:

            liquidus_ipts    A list of lists of indices of the points along the
                             liquidus curve. The last one is equal to the first
                             one, to show that the curve is closed.
            liquidus_x       A list of lists of x[0:nendmember] vectors, also here
                             the last in the list equals the first, making it easier
                             to plot a closed curve or polygon.

        Note that these are lists-of-lists, because there can be multiple disjoint
        regions of liquid phase. 
        """
        simplices = self.thesimplices[-1]
        isims     = self.select_simplices_of_a_given_kind('liquid')
        nface     = self.nendmembers-1
        liquidus_walls  = []
        ## liquidus_edges  = [] # Something for ternary liquidi
        for isim in isims:
            walls = []
            for i,ineigh in enumerate(simplices['neighbors'][isim]):
                cpts = list(set(simplices['ipts'][isim]).intersection(set(simplices['ipts'][ineigh])))
                assert len(cpts)==nface, f'Face has wrong nr of points {cpts}'
                cpts.sort()
                tpts = tuple(cpts)
                walls.append(tpts)
                if simplices['stype'][ineigh]!='liquid':
                    # Neighbor with a non-fluid simplex == liquidus wall
                    liquidus_walls.append(tpts)
            if len(simplices['neighbors'][isim])<self.nendmembers:
                # Some walls are at the edge of the domain
                dummy = []
                for i in range(self.nendmembers):
                    l = list(simplices['ipts'][isim]).copy()
                    l.remove(l[i])
                    l.sort()
                    dummy.append(tuple(l))
                walls = list(set(dummy)-set(walls))
                for w in walls:
                    liquidus_walls.append(w)
        self.liquidus_walls = liquidus_walls.copy()
        chains = self.get_chains(liquidus_walls)
        liquidus_x    = []
        liquidus_ipts = []
        xpts  = self.thepoints[-1][:,:-1]
        for c in chains:
            cn   = np.stack(c)
            ipts = np.hstack((cn[0,0],cn[:,1],cn[0,0]))
            x    = np.zeros((len(ipts),self.nendmembers))
            for i in range(len(ipts)):
                x[i,:-1] = xpts[ipts[i]]
            x[:,-1] = 1-x[:,:-1].sum(axis=-1)
            liquidus_x.append(x)
            liquidus_ipts.append(ipts)
        self.liquidus_ipts  = liquidus_ipts
        self.liquidus_x     = liquidus_x
        return liquidus_ipts,liquidus_x

    def define_a_relevelling_plane(self,endmembers=None,useliq=False,x=None,G=None):
        """
        The Gibbs free energy landscape is generally on a "steep slope" because the endmembers
        typically have very different Delta_f G. The Gibbs energies of the intermediate composition
        phases or phase-combinations are, typically, only slightly below the surface spanned by the
        endmembers. This makes it hard to analyze. The self.define_a_relevelling_plane() allows you 
        to pre-define the function self.get_G_plane(x) that returns the G values of a plane going
        through the endmembers (or through any set of len(endmembers) pairs of x and G) so that
        you can (for plotting, for instance) subtract those values from the G values obtained
        from the PhaseHull algorithm. In this way, these new Gibbs energies are 0 at the endmembers
        (or the set of X,G pairs given), and (likely/mostly) <0 between them, giving a much clearer
        view of the Gibbs free energy landscape within the simplex spanned by the endmembers (or
        set of x,G pairs). Only for convenience.

        Arguments (either/or):

          endmembers      If set, the endmembers to use to define the plane.
          useliq          If set, use the endmembers of the liquids, not those of the crystals
                          to define the plane.

          x, G            If set, the set of len(endmembers) to use to define the plane.

        Returns:

          nothing, but you can now use self.get_G_plane(x) to get the values of G at points x
          on that plane, which you can subtract from the G values you get from the PhaseHull()
          algorithm.
        """
        if endmembers is not None:
            assert x is None and G is None, 'Error: Cannot set both endmembers and x,G'
            nendm = len(endmembers)
            x = np.zeros(nendm)
            G = np.zeros(nendm)
            if useliq:
                liq = self.liquids[-1]
                for i in range(nendm):
                    x[:] = 0
                    x[i] = 1.
                    G[i] = liq.call_Gfunc(endmembers,x[i,:])
            else:
                cryst = self.crystals[-1]
                db    = cryst.dbase
                iendmembers,Gzero = self.identify_endmember_minerals(db,endmembers)
                Gzero = np.array(Gzero)
            self.relevel_G = Gzero
        else:
            assert x is not None and G is not None, 'Error: You must set either endmembers or x,G'
            from scipy import linalg
            x     = np.array(x)
            G     = np.array(G)
            nendm = x.shape[-1]
            evec  = np.zeros((nendm-1,nendm-1))
            for i in range(nendm-1):
                evec[:,i] = x[i,:-1]-x[-1,:-1]
            einv     = linalg.inv(evec)
            xx       = np.zeros([nendm,nendm])
            for i in range(nendm):
                xx[i,i] = 1.
            y        = np.zeros([nendm,nendm])
            y[:,:-1] = (einv[None,:,:]*(xx[:,None,:-1]-xx[-1,None,:-1])).sum(axis=-1)
            y[:,-1]  = 1-y[:,:-1].sum(axis=-1)
            Gzero    = np.zeros(nendm)
            for i in range(nendm):
                Gzero += y[:,i]*G[i]
            self.relevel_G = Gzero

    def G_relevelling_plane(self,x):
        """
        After calling define_a_relevelling_plane() you can use G_relevelling_plane(x) to obtain
        the G value of this plane at any point x. By subtracting this from the G values you obtain
        from the PhaseHull() algorithm, you get "re-levelled G values", where the endmembers have
        G=0. 
        """
        if x.shape[-1] == len(self.endmembers)-1:
            xx = x
            sh = list(xx.shape)
            sh[-1] += 1
            x  = np.zeros(sh)
            x[...,:-1] = xx[...,:]
            x[...,-1]  = 1-xx[...,:].sum(axis=-1)
        assert x.shape[-1] == len(self.endmembers), 'Error: Dimension of x incorrect'
        if hasattr(self,'relevel_G'):
            if len(x.shape)==1:
                return (self.relevel_G*x).sum()
            else:
                return (self.relevel_G[...,:]*x).sum(axis=-1)
        else:
            return np.zeros(x.shape[:-1])

    # ------ Inner working stuff for PhaseHull -----

    def call_Gfunctions(self,endmembers,x):
        """
        PhaseHull can handle multiple continua, for instance liquid and solid metal alloy. When these functions
        are called, the lowest of the continua is taken at any point x. Also the id-number of this continuum
        is then chosen for that point. Returned is G and the id number of the continuum, for every x.

        Argument:

          endmembers   List of names of the endmembers in the correct order
                       in which they are in x.

          x            The fractional abundances. x can be a 2D array x[N,M].

        Returns:

          G            The (lowest of) the G value(s) at each x of the liquids or alloys.

          idlm         The identification number of which liquid has lowest G at that x.
                       These id numbers start with -1 (for the first liquid in the self.liquids)
                       and -2 (for the second liquid), etc.
        """
        assert len(self.liquids)>0, 'Error: No continua (liquids, alloys or glasses) available.'
        G    = []
        idl  = []

        # Loop over all liquids (usually just one, and note that 'liquid' here means
        # a continuum, and can therefore also be a non-fixed composition solid, such
        # as an alloy).
        
        for iliq in range(len(self.liquids)):
            G.append(self.liquids[iliq].call_Gfunc(endmembers,x))  # Call the function that computes G
            idl.append(-iliq-1)                                    # Store the ID of this liquid
        G    = np.stack(G).T
        imin = G.argmin(axis=-1)                 # Determine which of the liquids has the lowest G
        Gmin = np.zeros(len(G))                  # Since x can be an array of points
        idlm = np.zeros(len(G),dtype=int)        # we find and identify the lowest G at each point
        for i in range(len(G)):
            im      = imin[i]
            Gmin[i] = G[i,im]
            idlm[i] = idl[im]
        return Gmin,idlm

    def create_points_for_liquids(self,xgrid,Ggrid):
        """
        From a set of points xgrid with a G value at each point Ggrid,
        construct the points we shall use for the convex hull algorithm. This is
        used for setting up a grid of points to map continua (i.e., liquids,
        which include also alloys or glasses). These have no fixed
        stoichiometry, hence they have a Gibbs energy at all (or a continuous
        subspace) of the x phase space.
        """
        pts = np.zeros((len(Ggrid),self.nendmembers))
        ids = np.zeros((len(Ggrid)),dtype=int)
        for i in range(len(Ggrid)):
            pts[i,:-1] = xgrid[i][:-1].copy()    # For the convex hull we only need the independent x values, hence [:-1]
            pts[i,-1]  = Ggrid[i][0]             # The last value of the point coordinates is the G value
            ids[i]     = Ggrid[i][1]             # The ids tell which liquid this point belongs to, if multiple continua are used.
        pts = np.stack(pts)
        return pts,ids

    def create_points_for_crystals(self):
        """
        Crystals (in PhaseHull) are fixed-composition points in x space.
        They are discrete points, each of which belongs to an entry in
        the mineral database of self.crystals. This function creates
        the corresponding points for the convex hull algorithm.

        Note that we use mfDfG, not DfG here. The reason is that from one
        mole of endmembers, you do not always get 1 mole of crystal. Example
        from half a mole of MgO and half a mole of SiO2 (in total these
        comprise 1 mole of endmembers) you get only half a mole of MgSiO3.
        So we should use 0.5 times the Gibbs energy of MgSiO3. This is
        what mfDfG is, in relation to DfG. 
        """
        pts   = []
        ids   = []
        nendm = len(self.endmembers)

        # Not always we have such fixed-composition minerals
        
        if len(self.crystals)>0:

            # Get the crystal database

            assert len(self.crystals)==1, 'Error: At the moment, only one crystal database is allowed.'
            mdb       = self.crystals[0].dbase

            # At the endmember locations only 1 crystal is allowed, so pick the lowest energy one

            mdb       = self.remove_non_lowest_endmembers_from_db(mdb)

            # Loop over all remaining crystals
            
            for i,row in mdb.iterrows():
                pt    = np.zeros(nendm)
                pt[:-1] = row['x'][:-1]   # For the convex hull we only need the independent x values, hence [:-1]
                pt[-1]  = row['mfDfG']    # The last value of the point coordinates is the G value, weighted by moles
                pts.append(pt)
                name    = row['Abbrev']   # Add the label of which crystal this is to this point, for later reference.
                ids.append(name)
        return pts,ids

    def remove_non_lowest_endmembers_from_db(self,mdb):
        """
        Sometimes, if the crystal database contains multiple crystals at (1,0,0)
        or another corner point, then Qhull finds simplices connecting both
        (or more) points that are all at the same x. This function eliminates
        any multiplicity of crystals at the same corner by choosing only the
        one with the lowest energy.
        """
        mdb   = mdb.reset_index(drop=True)
        iincl = list(np.arange(len(mdb)))
        for k in range(self.nendmembers):
            ii = []
            GG = []
            for i,m in mdb.iterrows():
                if m['x'][k]==1:
                    ii.append(i)
                    GG.append(m['mfDfG'])
            assert len(ii)>0, f'No Endmember found in direction {k}'
            ikeep = ii[np.argmin(GG)]
            for i in ii:
                if i!=ikeep:
                    iincl.remove(i)
        mdb = mdb.iloc[iincl].reset_index(drop=True)
        return mdb

    def setup_base_grid(self):
        """
        For liquids or non-fixed-composition solids (alloys, glasses) we need a grid
        of x values. This grid can be refined if necessary, but first we set up a
        regular grid with self.nres0 grid points along each axis.

        Because we explicitly envision the use of recursive grid refinement, we
        allow for successive layers of refinement, while keeping the data of the
        coarser refinement levels in memory. The data for these grids are stored
        in a series of arrays, with names all starting with "the". The most important
        ones are:

            self.thepoints      The points used for the convex hull
            self.theids         The identifications of the points used for the convex hull

        The highest resolution grid is always the last. So the set of points of
        the base grid are self.thepoints[0], while the set of points of the
        most-refined grid are self.thepoints[-1] (where -1 stands for last).
        """

        # For each level of refinement, all data is stored in self.the<something>.
        # To get the highest-resolution data, always choose self.the<something>[-1]
        self.thepoints       = []  # The points used for the convex hull
        self.theids          = []  # The identifications of the points used for the convex hull
        self.thepoints_liq   = []  # Only for convenience
        self.theids_liq      = []  # Only for convenience
        self.thepoints_cryst = []  # Only for convenience
        self.theids_cryst    = []  # Only for convenience

        # The liquids
        pts_liq = None
        ids_liq = None
        if(len(self.liquids)>0):
            nres              = self.nres0
            self.gridsnres    = [nres]
            igrid             = self.make_integer_grid(nres)  # Call the actual grid builder
            self.igrids       = []
            self.igrids.append(igrid)
            self.xgrids       = []
            xgrid             = igrid/nres
            self.xgrids.append(xgrid)
            Ggrid             = []
            for x in xgrid:
                Ggrid.append(self.call_Gfunctions(self.endmembers,x))
            pts_liq,ids_liq = self.create_points_for_liquids(xgrid,Ggrid)
            self.thepoints_liq.append(pts_liq)
            self.theids_liq.append(ids_liq)

        # The fixed stoichiometry crystals
        pts_cryst = None
        ids_cryst = None
        if(len(self.crystals)>0):
            pts_cryst,ids_cryst = self.create_points_for_crystals()
            self.thepoints_cryst.append(np.stack(pts_cryst))
            self.theids_cryst.append(ids_cryst)

        # Combine the points
        pts = []
        ids = []
        if pts_liq is not None:
            pts += list(pts_liq)
            ids += list(ids_liq)
        if pts_cryst is not None:
            pts += list(pts_cryst)
            ids += list(ids_cryst)
        pts = np.stack(pts)
        self.thepoints.append(pts)
        self.theids.append(ids)

    def make_integer_grid(self,nres):
        """
        The function that constructs the base grid in integer form. Note
        that for 3 enmembers the grid is triangular, for 4 endmembers
        the grid is a tetrad, etc. The integer grid is constructed in
        a way that the full allowed space is uniformly mapped with
        nres points along each axis. 
        """
        ixgrid = set()
        if self.nendmembers==2:
            for ix0 in range(0,nres+1):
                if ix0>=0 and ix0<=nres:
                    k = nres - ix0
                    ixgrid.add((ix0,k))
        elif self.nendmembers==3:
            for ix0 in range(0,nres+1):
                for ix1 in range(0,nres+1):
                    if ix0>=0 and ix1>=0 and ix0+ix1<=nres:
                        k = nres - ix0 - ix1
                        ixgrid.add((ix0,ix1,k))
        elif self.nendmembers==4:
            for ix0 in range(0,nres+1):
                for ix1 in range(0,nres+1):
                    for ix2 in range(0,nres+1):
                        if ix0>=0 and ix1>=0 and ix2>=0 and ix0+ix1+ix2<=nres:
                            k = nres - ix0 - ix1 - ix2
                            ixgrid.add((ix0,ix1,ix2,k))
        else:
            raise ValueError(f'Unfortunately at the moment we cannot handle nr of endmembers = {self.nendmembers}')
        ixgrid = list(ixgrid)
        ixgrid.sort()
        ixgrid = np.array(ixgrid)
        return ixgrid

    def make_x_grid(self,nres):
        """
        Wrapper around make_integer_grid(), where the integer values are rescaled back
        to values between 0 and 1.
        """
        ixgrid = self.make_integer_grid(nres)
        x      = ixgrid / nres
        return x

    def do_convex_hull_algorithm(self,igrid):
        """
        The main routine of PhaseHull: this will call the convex hull algorithm
        for the computation of the phase diagram. The resulting set of simplices
        of the convex hull will then be analyzed: First, only the simplices at the
        bottom of the hull will be selected. Then each of the remaining simplices
        will be classified: is this simplex an element of a liquid, or is it a
        tie line, or is it a coexistence of crystals, or an inmiscibility simplex?

        The results are all listed in a dictionary called "simplices", which is then
        appended to the self.thesimplices list. This means in self.thesimplices[0]
        you have the results of the convex hull algorithm for the lowest-resolution
        (base grid) computation, while (if applicable) self.thesimplices[1] is the
        result of the next-higher refined grid, all the way to self.thesimplices[-1]
        which is the result of the highest refinement level.

        The simplices dictionary consists of a set of simple arrays, the index of
        which is the index of the simplex. So if the convex hull algorithm found
        230 simplices at the basegrid, then simplices=self.thesimplices[0] is a
        dictionary of arrays, each of which has length 230. By checkout out
        simplex.keys() you can see which arrays are available. The most important
        are:

          simplices['stype']    The physical meaning of the simplices. E.g. if
                                simplices['stype'][5]=='cryst_1_liq_1', then simplex
                                number 5 represents a coexistence of one crystals
                                with one liquid.
          simplices['ipts']     The indices of the points at its corners.
          simplices['x']        The coordinate of the corner points.
          simplices['G']        The G values at the corner points.
        """
        pts       = self.thepoints[igrid]
        idlist    = self.theids[igrid]
        answer    = self.find_lower_convex_hull_of_x_G_points(pts)
        hull      = answer['hull']
        self.thehulls.append(hull)
        simplices = self.classify_simplices(pts,idlist,hull)
        self.thesimplices.append(simplices)

    def do_one_refinement_step(self):
        """
        To correctly reproduce delicate details of the phase diagram it is
        usually important to have very high grid resolution close to certain
        special locations, e.g. close to the liquidus or the binodal curve.
        But to avoid wasting computing power on uninteresting regions, we
        employ adaptive grid refinement.

        This function takes the results of the previous refinement step,
        figures out where further refinement can be beneficial, adds gridpoints
        there, and then redoes the convex hull computation.
        """
        assert len(self.liquids)>0, 'Error: No refinement necessary if no liquids/alloys/glasses available'
        assert len(self.thepoints)==len(self.thesimplices), f'First call do_convex_hull_algorithm({len(self.thepoints)-1})'
        igrid     = len(self.gridsnres)
        nres      = self.gridsnres[-1]
        nresnew   = nres*self.nfact
        print(f'Iteration {igrid} at dx = 1/{nresnew}')
        self.gridsnres.append(nresnew)

        # The liquids
        pts_liq   = self.thepoints_liq[igrid-1]
        ids_liq   = self.theids_liq[igrid-1]
        simplices = self.thesimplices[igrid-1]
        ptsnew_liq,idsnew_liq = self.refine_near_binodals_2d(pts_liq,simplices,nres+1,nfact=self.nfact,nspan=self.nspan)
        self.thepoints_liq.append(ptsnew_liq)
        self.theids_liq.append(idsnew_liq)

        # The fixed stoichiometry crystals
        ptsnew_cryst = None
        if(len(self.crystals)>0):
            ptsnew_cryst = self.thepoints_cryst[igrid-1]
            idsnew_cryst = self.theids_cryst[igrid-1]
            self.thepoints_cryst.append(np.stack(ptsnew_cryst))
            self.theids_cryst.append(idsnew_cryst)

        # Combine the points
        pts = []
        ids = []
        if ptsnew_liq is not None:
            pts += list(ptsnew_liq)
            ids += list(idsnew_liq)
        if ptsnew_cryst is not None:
            pts += list(ptsnew_cryst)
            ids += list(idsnew_cryst)
        pts = np.stack(pts)
        self.thepoints.append(pts)
        self.theids.append(ids)

        # Call the convex hull algorithm
        self.do_convex_hull_algorithm(igrid)

    def do_all_refinement_steps(self):
        #print(f'First low resolution calculation at dx = 1/{self.nres0}')
        self.do_convex_hull_algorithm(0)
        if (len(self.liquids)>0) and len(self.endmembers)>2:
            for iter in range(self.nrefine):
                self.do_one_refinement_step()

    def find_lower_convex_hull_of_x_G_points(self,pts):
        """
        Find the convex hull of all the points in N-1 x-space dimensions +
        1 mfDfG space dimension. We are interested the lower facets (simplices),
        as they represent, for each x, the lowest Gibbs energy mfDfG. The
        convex hull, however, also has an upper facet (or multiple upper
        facets). They can be recognized by checking the last (=mfDfG) index
        of the facet normal vectors. We remove these, so that only the lowest
        energy facet for each x position is left: the bottom of the hull.
    
        Arguments:
    
          pts       The points in [x[:-1],mfDfG]-space. Pts is an array
                    with dimensions [npoints,N], where N is the number of
                    endmembers. But of these N elements, only N-1 are x
                    values, because the last x can be found from the
                    sum to unity condition: x[-1]=1-x[:-1].sum(). This
                    frees up space for the Gibbs free energy as the
                    last value. So x = pts[:,:-1] while mfDfG = pts[:,-1].
    
        Returns:
    
          answer    A dictionary containing:
    
                     simplices    Like ConvexHull.simplices, but only
                                  the ones of the bottom of the hull.
                     equations    Like ConvexHull.equations, but only
                                  the ones of the bottom of the hull.
                     vertices     Like ConvexHull.vertices
                     hull         The original ConvexHull instance.
    
        Note: Since the simplices and equations are a subset of the
              hull.simplices and hull.equations, the indices of them
              are not to be used with hull.neighbors. The indices in
              hull.neighbors refer to hull.simplice instead.
        """
        points    = np.stack(pts)
        hull      = ConvexHull(points)
        include   = hull.equations[:,-2]<0
        simplices = hull.simplices[include,:]
        equations = hull.equations[include,:]
        vertices  = hull.vertices
        answer    = {}
        answer['simplices'] = simplices
        answer['equations'] = equations
        answer['vertices']  = vertices
        answer['hull']      = hull
        return answer

    def classify_simplices(self,pts,idlist,hull,mrcrit=None):
        """
        Interpret the output of the convex hull computation in terms
        of the meaning of each simplex. Each simplex will get a label
        (the 'stype' key of the dict) specifying the physical meaning.
        For now only physical meanings for up to ternary diagrams are
        included, but this is easily extendable to quaternary diagrams
        and higher. Types:
    
           liquid            Fully liquid
           allcryst          A coexistence of 2 (for binary) or 3 (for 
                             ternary) (etc) crystals.
           cryst_1_liq_1     A coexistence of 1 crystal phase with
                             1 liquid phase (only for binary diagrams).
           cryst_2_liq_1     A coexistence of 2 crystal phases with
                             1 liquid phase (only for ternary diagrams).
           cryst_1_liq_2     A coexistence of 1 crystal phase with
                             2 liquid phases (only for ternary diagrams),
                             but not a tie line.
           tieline_c1l2      Like cryst_1_liq_2, but with the physical
                             meaning of a tie line between 1 crystal and
                             1 liquid point (only for ternary diagrams).
                             This is a very elongated simplex. Its
                             meaning is that of a binary phase line 
                             in a ternary diagram. The two fluid points
                             are actually a single fluid, but it becomes 
                             two very-nearby fluid points due to the
                             discretization.
           tieline_c0l3      A coexistence of two liquid phases that
                             are inmiscible  (only for ternary diagrams).
                             In reality 3 liquids are connected, two
                             being very close together.
           inmisc_liquids_3phase   Like 'inmisc_liquids' but for
                                   three phases. In a ternary diagram
                                   these occasionally show up as big
                                   single triangles.
    
        Arguments:
    
          pts       The points in [x[:-1],mfDfG]-space. Pts is an array
                    with dimensions [npoints,N], where N is the number of
                    endmembers. But of these N elements, only N-1 are x
                    values, because the last x can be found from the
                    sum to unity condition: x[-1]=1-x[:-1].sum(). This
                    frees up space for the Gibbs free energy as the
                    last value. So x = pts[:,:-1] while mfDfG = pts[:,-1].
    
          idlist    List of the same length as pt with the identity of
                    each point. For instance, if you have 3 solid
                    crystals (say 'SiO2','MgO', and 'MgSiO3'), and
                    200 liquid points, then the idlist would look like
                    this: ['SiO2','MgO', 'MgSiO3', -1, -1, ... , -1]
                    where -1 has the meaning of the liquid.
    
          hull      The object in the answer['hull'] from the function
                    find_lower_convex_hull_of_x_G_points(pts)
    
        Returns:
    
          simplices     A list of dicts with information for each simplex.
        """
        if mrcrit is None: mrcrit=self.mrcrit
        if hasattr(self,'gridsnres'):
            nx = self.gridsnres[-1]
        pts   = np.stack(pts)
        npts  = len(pts)
        nendm = pts.shape[-1]
        simplices = {'id':[],'id_qhull':[],'stype':[],'ipts':[],'neighbors_qhull':[]}
        if self.incl_ptnames:
            simplices['ptnames'] = []
        if self.incl_xvals:
            simplices['x'] = []
        if self.incl_Gvals:
            simplices['G'] = []
        if self.incl_xcen:
            simplices['xcen'] = []
        if self.incl_Gcen:
            simplices['Gcen'] = []
        if self.incl_xtie:
            simplices['xtie'] = []
        isimnew = 0
        simindices = {}

        # Loop over all simplices of the convex hull

        for isim,simplex in enumerate(hull.simplices):

            # But select only the simplices who's normal
            # vector point downward (bottom of the convex
            # hull)

            if(hull.equations[isim,-2]<0):
                names    = []
                x        = []
                G        = []
                allcryst = True
                allcont  = True

                # Loop over all corner points of this simplex
                # and store their x coordinates and G values,
                # as well as their IDs/names. If the name of a points
                # is an integer <0, then the point represents
                # a liquid (or continuous solid) point. If the
                # name is a string, then it represents a fixed
                # composition solid (which we call crystal in
                # PhaseHull). 

                for ipt in simplex:
                    name = idlist[ipt]
                    names.append(name)
                    x.append(pts[ipt,:-1])
                    G.append(pts[ipt,-1])
                    #if type(name) is int: allcryst = False
                    if type(name) is str:
                        allcont  = False
                    else:
                        allcryst = False
                assert not (allcryst and allcont), 'Error: Inconsistency in crystal/continuum.'

                # Since the x values of these points are only
                # the independent x values (one fewer than the
                # number of endmembers), we reconstruct the full
                # x values using x[last] = 1 - sum x[all but last].

                xx        = np.zeros((len(x),len(x[0])+1))
                xx[:,:-1] = x
                xx[:,-1]  = 1-xx.sum(axis=-1)

                # Store the current information in arrays 
                
                simplices['id'].append(isimnew)
                simplices['id_qhull'].append(isim)
                simplices['ipts'].append(simplex)
                if self.incl_ptnames: simplices['ptnames'].append(names)
                if self.incl_xvals:   simplices['x'].append(xx)
                if self.incl_Gvals:   simplices['G'].append(G)

                # Now classify the simplex: Is it a pure liquid (or continuous
                # solid/alloy, which is equivalent in PhaseHull) or is it a coexistence
                # of crystals (fixed composition points in the phase diagram)?
                # This is the stype of this simplex.
                
                if allcryst:
                    # All corners of this simplex are crystal solids
                    # A tie simplex
                    simplices['stype'].append('allcryst')
                elif allcont:
                    # All corners of this simplex are liquid/alloy
                    if nx is not None:
                        inmisc,x,l,mr = self._test_if_simplex_is_inmiscible_fluids(pts,isimnew,simplices['ipts'],nx=nx,return_all=True)
                    else:
                        inmisc        = self._test_if_simplex_is_inmiscible_fluids(pts,isimnew,simplices['ipts'],nx=nx,return_all=False)
                        mr            = None
                    if(inmisc):                                  # If we are in an inmiscibility part of the liquid, 
                        nliq  = self.nendmembers                 # then do some more work.
                        stype = f'tieline_c0l{nliq}'             # Normally if a liquid has inmiscibility, the simplex is a tie line
                        if mr is not None and self.nendmembers==3:
                            if mr>mrcrit:                        # In a ternary one can also have inmiscibility of three compositions.
                                stype = 'inmisc_liquids_3phase'  # This is a bit tricky to find: we use the shape of the simplex.
                        # Check if this simplex connects different liquids (if you have more than 1 continuum)
                        if len(set(names))>1:
                            stype += '_crossliq'
                    else:
                        stype = 'liquid'                         # If not inmiscible, then this is a normal liquid part
                    simplices['stype'].append(stype)
                else:
                    # Some corners are liquid, some are crystal solids
                    nliq   = 0
                    ncryst = 0
                    for i in range(len(simplex)):
                        if type(names[i]) is str:
                            ncryst += 1
                        else:
                            nliq   += 1
                    if self.nendmembers>2:
                        if self._test_if_simplex_is_tie_line(pts,isimnew,simplices['ipts']):
                            simplices['stype'].append(f'tieline_c{ncryst}l{nliq}')
                        else:
                            simplices['stype'].append(f'cryst_{ncryst}_liq_{nliq}')
                    else:
                        simplices['stype'].append(f'tieline_c{ncryst}l{nliq}')
                simplices['neighbors_qhull'].append(hull.neighbors[isim].copy())
                simindices[isim] = isimnew
                isimnew += 1

        simplices['neighbors'] = []
        for isimnew in range(len(simplices['stype'])):
            idxnew = []
            for i in range(len(simplices['neighbors_qhull'][isimnew])):
                k = simplices['neighbors_qhull'][isimnew][i]
                if k in simindices:
                    idxnew.append(simindices[k])
            simplices['neighbors'].append(idxnew)

        # Convert all lists to numpy arrays for easier handling
        simplices['id']              = np.array(simplices['id'])
        simplices['id_qhull']        = np.array(simplices['id_qhull'])
        simplices['ipts']            = np.array(simplices['ipts'])
        simplices['stype']           = np.array(simplices['stype'])
        #simplices['neighbors']       = np.array(simplices['neighbors'])
        #simplices['neighbors_qhull'] = np.array(simplices['neighbors_qhull'])
        if self.incl_ptnames:  simplices['ptnames'] = np.array(simplices['ptnames'])
        if self.incl_xvals:    simplices['x']       = np.array(simplices['x'])
        if self.incl_Gvals:    simplices['G']       = np.array(simplices['G'])
        if self.incl_xtie:     simplices['xtie']    = np.array(simplices['xtie'])
        if self.incl_xcen:     simplices['xcen']    = np.array(simplices['xcen'])
        if self.incl_Gcen:     simplices['Gcen']    = np.array(simplices['Gcen'])
        return simplices

    def _test_if_simplex_is_inmiscible_fluids(self,pts,isimnew,sim_ipts,nx=None,return_all=False):
        # Inmiscible liquids occur when the G-surface is not convex,
        # but instead has a "wiggle". If the present simplex lies (apart
        # from its corner points) below the true liquid G function, then
        # this simplex is a tie line or tie simplex of inmiscible liquids.
        # This is what is tested here.
        N   = len(pts[0])                                          # Nr of endmembers 
        npt = len(pts)                                             # Nr of points available (not all are part of the convex hull)
        x   = np.zeros((N+1,N))                                    # Molar fractions of the corner points of this simplex
        for i in range(N):
            x[i,:-1] = pts[sim_ipts[isimnew][i]][:-1]
            x[i,-1]  = 1-x[i,:-1].sum(axis=-1)
        x[-1,:] = x[0,:]                                           # Copy the first x to the extra x, useful for the algorithm
        xcen = x[:-1,:].mean(axis=0)                               # The center of this simplex
        Gcont,idcont = self.call_Gfunctions(self.endmembers,xcen)  # Compute the real G at this center
        Gcen = 0.                                                  # Now compute the mean of the G values at the corners of the simplex
        for i in range(N):
            Gcen += pts[sim_ipts[isimnew][i]][-1]
        Gcen /= N
        if return_all and nx is not None:
            l   = np.zeros(N)                                      # Compute the distance between successive points on the simplex
            for i in range(N):                                     # The method here is not the most correct one, but it works.
                l[i] = ((x[i,:]-x[i+1,:])**2).sum()**0.5
            dx = 1/nx
            mr = l.min()/dx                                        # Note: Here use min()
            return Gcen<Gcont,x,l,mr                               # In addition to inmiscibility, return also further information
        else:
            return Gcen<Gcont                                      # If Gcen<Gcont, then this simplex is a simplex of inmiscibility

    def _test_if_simplex_is_tie_line(self,pts,isimnew,sim_ipts,mrcrit=None,nx=None):
        if mrcrit is None: mrcrit=self.mrcrit
        assert self.nendmembers>2, 'Internal Error: Should not call _test_if_simplex_is_tie_line for binaries'
        if nx is None:
            if hasattr(self,'gridsnres'):
                nx = self.gridsnres[-1]
            else:
                raise ValueError('If nx is not set, you must have self.gridsnres set')
        N   = len(pts[0])
        npt = len(pts)
        x   = np.zeros((N+1,N))
        for i in range(N):
            x[i,:-1] = pts[sim_ipts[isimnew][i]][:-1]
            x[i,-1]  = 1-x[i,:-1].sum(axis=-1)
        x[-1,:] = x[0,:]
        l   = np.zeros(N)
        for i in range(N):
            l[i] = ((x[i,:]-x[i+1,:])**2).sum()**0.5
        dx = 1/nx
        mr = l.min()/dx   # Note: Here use min()
        return mr<=mrcrit

    def refine_near_binodals_2d(self,pts,simplices,nx,nfact=2,nspan=1,return_newnx=False):
        eps      = 1e-3
        nend     = len(pts[0])
        assert nend==3, 'Error: refine_near_binodals_2d() only works with 3 endmembers.'
        tie_lines = []
        for isim in range(len(simplices['id'])):
            #
            # ADD MORE OF THESE TRIGGERS FOR TERNARY OR HIGHER
            #
            if simplices['stype'][isim]=='tieline_c0l3' or \
               simplices['stype'][isim]=='tieline_c1l2':
                tie_lines.append(isim)
        nxnew    = (nx-1)*nfact+1
        gridold  = (np.stack(pts)[:,:-1]*(nx-1)+eps).astype(int)
        check    = (np.stack(pts)[:,:-1]*(nx-1)+1-eps).astype(int)
        assert np.all(gridold==check), f'Error: Current grid is finer than 1/{nx-1}'
        gridnew  = gridold * nfact
        gridnset = set([tuple(g) for g in gridnew])
        gaddnew  = set()
        for ib in tie_lines:
            for x in simplices['x'][ib]:
                gr = (x[:nend-1]*(nx-1)+eps).astype(int) * nfact
                # Next lines only work for nend==3
                gadd  = set()
                for ix0 in range(gr[0]-nspan*nfact,gr[0]+nspan*nfact+1):
                    for ix1 in range(gr[1]-nspan*nfact,gr[1]+nspan*nfact+1):
                        if ix0>=0 and ix1>=0 and ix0+ix1<nxnew:
                            gadd.add((ix0,ix1))
                gaddnew = gaddnew.union(gadd)
        gridnset = gridnset.union(gaddnew)
        grnew    = np.stack(list(gridnset))
        xnew     = np.zeros((len(grnew),nend))
        xnew[:,:-1] = grnew.astype(float)/(nxnew-1)
        xnew[:,-1]  = 1-xnew[:,:-1].sum(axis=-1)
        Gnew     = np.zeros(len(grnew))
        idsnew   = np.zeros(len(grnew),dtype=int)
        for i in range(len(grnew)):
            Gnew[i],idsnew[i] = self.call_Gfunctions(self.endmembers,xnew[i])
        ptsnew   = xnew
        ptsnew[:,-1] = Gnew[:]
        if return_newnx:
            return ptsnew,idsnew,nxnew
        else:
            return ptsnew,idsnew

    def _follow_neighbors_till_end(self,iselection,istart,stype,maxstep=10000):
        icurr = istart
        simplices = self.thesimplices[-1]
        neigh = simplices['neighbors_'+stype][icurr]
        if(len(neigh)!=2):
            if(len(neigh)==1):
                return icurr,0
            else:
                return icurr,None
        iprev = neigh[0]  # Random choice of the two
        for istep in range(maxstep):
            assert iprev in neigh, 'Weird error: neighbors not bidirectional'
            if iprev==neigh[0]:
                inext = neigh[1]
                ii    = 0
            else:
                inext = neigh[0]
                ii    = 1
            iprev = icurr
            icurr = inext
            neigh = simplices['neighbors_'+stype][icurr]
            if(len(neigh)==1):
                assert iprev==neigh[0], 'Weird error: neighbors not bidirectional at endpoint'
                return icurr,0
            elif(len(neigh)>2):
                return iprev,ii
            if icurr==istart:
                return icurr,0
        raise ValueError('Did not find the start/end of the neighbor series')

    def _next_neighbor(self,iselection,iprev,icurr,stype):
        simplices = self.thesimplices[-1]
        neigh = simplices['neighbors_'+stype][icurr]
        if iprev<0:
            if(len(neigh))==1:
                return neigh[0]
            else:
                iprev = -(iprev+1)
                assert iprev in neigh, 'Weird error: iprev not in neigh'
                return iprev
        if len(neigh)!=2:
            return None  # Reached end
        assert iprev in neigh, 'Weird error: neighbors not bidirectional'
        if iprev==neigh[0]:
            return neigh[1]
        else:
            return neigh[0]
    
    def sort_tie_lines(self,iselection,stype,nrgroupmax=100,maxstep=10000):
        groups = []
        simplices = self.thesimplices[-1]
        left   = iselection.copy()
        for igroup in range(nrgroupmax):
            if len(left)==0:
                break
            icurr,ii = self._follow_neighbors_till_end(iselection,left[-1],stype,maxstep=10000)
            group = []
            if ii is None:
                left.remove(left[-1])
            else:
                assert icurr in left, f'Something went wrong in left. icurr={icurr}'
                iprev = -simplices['neighbors_'+stype][icurr][ii]-1
                for istep in range(maxstep):
                    group.append(icurr)
                    left.remove(icurr)
                    inext = self._next_neighbor(iselection,iprev,icurr,stype)
                    if inext is None:
                        break
                    if inext not in left:
                        break
                    iprev = icurr
                    icurr = inext
            if len(group)>0:
                groups.append(group)
        for group in groups:
            for i,isim in enumerate(group):
                xcurr = simplices['xtieline'][isim]
                if i>0:
                    dx0110 = ((xcurr[0,:]-xprev[1,:])**2).sum() + ((xcurr[1,:]-xprev[0,:])**2).sum()
                    dx0011 = ((xcurr[0,:]-xprev[0,:])**2).sum() + ((xcurr[1,:]-xprev[1,:])**2).sum()
                    if dx0110<dx0011:
                        xswap         = xcurr.copy()
                        xcurr[0,:]    = xswap[1,:]
                        xcurr[1,:]    = xswap[0,:]
                        simplices['xtieline'][isim] = xcurr
                xprev  = xcurr
        return groups

    def identify_endmember_minerals(self,mdb,endmembers):
        """
        This function determines which of the crystals in the mineral database
        are the true endmembers with the lowest DfG.

        Arguments:

          mdb              The mineral database (see read_minerals_and_liquids())
          endmembers       List of the formulae of the endmembers, e.g. ['SiO2','MgO','Al2O3'].

        Returns:

          iendmembers      List of integer indices of the mdb database pointing to the true
                           endmembers. If the mdb database has different versions of the endmembers
                           (e.g. alpha-quartz or beta-quartz for SiO2), then the version is
                           picked that has (for the given T) the smallest value of DfG.
          DfGendmembers    List of DfG values of these endmembers.
          
        """
        iendmembers   = np.zeros(len(endmembers),dtype=int)
        DfGendmembers = np.zeros(len(endmembers))+1e90
        for k,e in enumerate(endmembers):
            ms = mdb[mdb['Formula']==e]
            assert(len(ms)>0), f'Error: Could not find endmember mineral {e} among minerals'
            DfG  = 1e90
            iend = -1
            for i,row in ms.iterrows():
                if(row['DfG']<DfG):
                    iend = i
                    DfG  = row['DfG']
            assert i>-1, f'Error: Could not find endmember mineral {e} among minerals (stranger version)'
            iendmembers[k]   = iend
            DfGendmembers[k] = DfG
        return iendmembers,DfGendmembers

    def convert_mole_fraction_into_mass_fraction(self,endmemmass,x,G=None):
        """
        If you have a mole fraction x (such that x.sum(axis=-1)==1), or an array
        of them (again such that x.sum(axis=-1)==1, so x[...,:]), then you can
        convert them into mass fractions xm (again such that xm.sum(axis=-1)==1)
        with this function.

        Arguments;

          endmemmass       List of the masses (in atomic units) of the endmembers.
          x                Mole fraction x values. E.g. x = np.array([0.2,0.3,0.5]) or an
                           array of them, e.g. x = np.array([[0.2,0.3,0.5],[0.1,0.4,0.5]])

        Optional:

          G                If G is passed on, then also the adjusted value of G
                           (from J/mole to J/mass) is returned

        Returns:

          xm               Array of the same dimension as x, but this time with the
                           mass fractions instead of the mole fractions.

          mtot             The mass of 1 mole of this mineral.

          Gm               If G was passed on, this is the new version of G in
                           units of J/mass.
        """
        nendm = len(endmemmass)
        if len(x.shape)==1:
            mtot = 0.
        else:
            mtot = np.zeros_like(x[...,0])
        for k in range(nendm):
            mtot += x[...,k]*endmemmass[k]
        xm = np.zeros_like(x)
        for k in range(len(endmemmass)):
            xm[...,k] = x[...,k]*endmemmass[k]/mtot
        if G is not None:
            Gm = G/mtot
            return xm,mtot,Gm
        else:
            return xm,mtot

    def convert_mass_fraction_into_mole_fraction(self,endmemmass,xmass,Gmass=None):
        """
        The inverse of convert_mole_fraction_into_mass_fraction().

        If you have a mass fraction x (such that x.sum(axis=-1)==1), or an array
        of them (again such that x.sum(axis=-1)==1, so x[...,:]), then you can
        convert them into mole fractions xmol (again such that xmol.sum(axis=-1)==1)
        with this function.

        Arguments;

          endmemmass       List of the masses (in atomic units) of the endmembers.
          xmass            Mass fraction x values. E.g. x = np.array([0.2,0.3,0.5]) or an
                           array of them, e.g. x = np.array([[0.2,0.3,0.5],[0.1,0.4,0.5]])

        Optional:

          Gmass            If Gmass is passed on, then also the adjusted value of G
                           (from J/mass to J/mole) is returned.

        Returns:

          xmol             Array of the same dimension as x, but this time with the
                           mole fractions instead of the mass fractions.

          moltot           (if return_also_moltot==True) the nr of moles of 1 g of this mineral.

          Gmol             If Gmass was passed on, this is the new version of G in
                           units of J/mol.
        """
        nendm = len(endmemmass)
        endmemmol = 1/np.array(endmemmass)
        if len(xmass.shape)==1:
            moltot = 0.
        else:
            moltot = np.zeros_like(xmass[...,0])
        for k in range(len(iendmembers)):
            moltot += xmass[...,k]*endmemmol[k]
        xmol = np.zeros_like(xmass)
        for k in range(nendm):
            xmol[...,k] = xmass[...,k]*endmemmol[k]/moltot
        if Gmass is not None:
            Gmol = Gmass/moltot
            return xmol,moltot,Gmol
        else:
            return xmol,moltot

    def get_chain(self,tpairs,istart=0):
        assert type(tpairs[0]) is tuple, 'Error: tpairs must be a list of tuples.'
        npairs = np.stack(tpairs.copy())
        lpairs = [list(npairs[:,0]),list(npairs[:,1])]
        n      = len(tpairs)
        pstart = tuple(npairs[istart])
    
        tpairs.remove(pstart)
        lpairs[0].pop(istart)
        lpairs[1].pop(istart)
    
        # Forward
        p      = pstart
        m      = p[0]
        index  = istart
        chainf = [p]
        for k in range(n):
            j  = p[1]
            if j in lpairs[0]:
                index = lpairs[0].index(j)
                m     = lpairs[1][index]
                lpairs[0].pop(index)
                lpairs[1].pop(index)
                po    = tuple([j,m])
            elif j in lpairs[1]:
                index = lpairs[1].index(j)
                m     = lpairs[0][index]
                lpairs[0].pop(index)
                lpairs[1].pop(index)
                po    = tuple([m,j])
            else:
                break
            p     = tuple([j,m])
            chainf.append(p)
            tpairs.remove(po)
    
        # Backward
        p      = pstart
        m      = p[1]
        index  = istart
        chainb = []
        for k in range(n):
            j  = p[0]
            if j in lpairs[0]:
                index = lpairs[0].index(j)
                m     = lpairs[1][index]
                lpairs[0].pop(index)
                lpairs[1].pop(index)
                po    = tuple([j,m])
            elif j in lpairs[1]:
                index = lpairs[1].index(j)
                m     = lpairs[0][index]
                lpairs[0].pop(index)
                lpairs[1].pop(index)
                po    = tuple([m,j])
            else:
                break
            p     = tuple([m,j])
            chainb.append(p)
            tpairs.remove(po)
    
        chain = chainb[::-1] + chainf
        return chain,tpairs
    
    def get_chains(self,tpairs):
        chains = []
        while len(tpairs)>0:
            chain,tpairs = self.get_chain(tpairs)
            chains.append(chain)
        return chains

    def find_simplex_for_given_x(self,x,return_G=False,ilevel=-1):
        """
        Given a value, or a list of values, of x, this function
        returns the index (integer) of the simplex of the convex
        hull on which this point lies.

        Arguments:

          x         Location x[0:N] or array of locations x[0:nx,0:N]
                    where the interpolation should be done.

        Returns:

          isims     List of integers representing the indices of the
                    self.thesimplices[-1][:] simplices at the bottom
                    of the convex hull.

        Note: The algorithm is not at all optimal: For all x we try all
              simplices at the bottom of the hull, and select the highest
              G value, which is the correct one. Explanation: the highest 
              crossing with the planes through all simplices is, in fact, 
              the crossing with the simplex inside of which x is. since the
              simplices are the bottom of the hull, these highest points
              are, in fact, the lowest value, at x, of the volume enclosed
              by the convex hull. It works. But a much more efficient
              method would search for the right simplex by walking through
              the simplices using the hull.neighbors array. For arrays of
              contiguous x points, one can use the previous simplex as an
              initial guess for finding the current simplex. This is, however,
              not efficient in Python and would require an external C or
              Fortran library. However, for the current demonstration
              purposes, the present method is sufficient.
        """
        hull = self.thehulls[ilevel]
        if type(x) is list:
            x = np.stack(x)
        if len(x.shape)==1:
            x = x[None,:]
        mask    = hull.equations[:,-2]<0     # Select bottom of the hull
        nvec    = hull.equations[mask,:-1]   # Normal vectors of the simplices (at bottom)
        offset  = hull.equations[mask,-1]    # Offset of simplices from origin (at bottom)
        if type(x) is list:
            x = np.stack(x)
        N       = x.shape[-1]                # Nr of endmembers
        nx      = x.shape[0]                 # Nr of x vectors to interpolate at
        ns      = len(offset)                # Nr of simplices at bottom of the hull
        heights = np.zeros((nx,ns))
        for i in range(N-1):
            heights -= x[:,i,None]*nvec[:,i][None,:]
        heights -= offset[None,:]
        heights /= nvec[:,-1][None,:]
        simid    = np.argmax(heights, axis=-1)   # Find which simplex the x hits at the bottom of the hull
        if return_G:
            G = np.zeros(nx)
            for ix in range(nx):
                G[ix] = heights[ix,simid[ix]] # Get the line-simplex crossing height at that simplex
            return simid,G
        else:
            return simid

    def map_phase_diagram(self,nres=100,ilevel=-1,colormap=None):
        simplices       = self.thesimplices[ilevel]
        answer          = {}
        answer['x']     = self.make_x_grid(nres)
        answer['isim']  = self.find_simplex_for_given_x(answer['x'])
        answer['stype'] = []
        if colormap is not None:
            answer['color'] = []
        for isim in answer['isim']:
            stype = simplices['stype'][isim]
            answer['stype'].append(stype)
            if colormap is not None:
                if stype in colormap:
                    color = colormap[stype]
                else:
                    color = np.nan
                answer['color'].append(color)
        return answer
