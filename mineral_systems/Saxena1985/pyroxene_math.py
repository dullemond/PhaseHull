# This code provides the math behind the pyroxene solid solution
# quadrilateral spanned by En, Fs, Hd and Di, on the ternary
# spanned by En, Fs and Wo. 
#
# This code is written almost entirely by claude.ai under supervision of
# C.P. Dullemond, May 2026.

import numpy as np

def ternary_to_quad(xt_En, xt_Fs, xt_Wo, z):
    """
    Transform ternary coordinates (En-Fs-Wo) to pyroxene quadrilateral
    coordinates (En-Fs-Di-Hd).

    The ternary system uses end-members:
      Enstatite   (En): Mg2Si2O6  -> ternary vertex (1, 0, 0)
      Ferrosilite (Fs): Fe2Si2O6  -> ternary vertex (0, 1, 0)
      Wollastonite(Wo): Ca2Si2O6  -> ternary vertex (0, 0, 1)

    The quadrilateral occupies the region Wo <= 0.5 of the ternary, with
    corners:
      En at (1.0, 0.0, 0.0)
      Fs at (0.0, 1.0, 0.0)
      Di at (0.5, 0.0, 0.5)
      Hd at (0.0, 0.5, 0.5)

    Because the system is underdetermined (3 ternary coords, 4 quad coords),
    an extra parameter z = xq_Di is required to uniquely fix the solution.

    Parameters
    ----------
    xt_En : float
        Mole fraction of Enstatite in ternary coordinates.
    xt_Fs : float
        Mole fraction of Ferrosilite in ternary coordinates.
    xt_Wo : float
        Mole fraction of Wollastonite in ternary coordinates.
    z : float
        Mole fraction of Diopside in the quadrilateral (= xq_Di).
        This is the extra degree of freedom needed to make the system
        determined. Must satisfy 0 <= z <= 2 * xt_Wo.

    Returns
    -------
    xq_En : float
    xq_Fs : float
    xq_Di : float
    xq_Hd : float
    """
    xq_Di = z
    xq_Hd = 2.0 * xt_Wo - z
    xq_En = xt_En - 0.5 * z
    xq_Fs = xt_Fs - xt_Wo + 0.5 * z

    return xq_En, xq_Fs, xq_Di, xq_Hd

def z_valid_range(xt_En, xt_Fs, xt_Wo):
    """
    Compute the valid range of z (= xq_Di) for a given point in ternary
    coordinates, such that all four quadrilateral coordinates lie in [0, 1].

    Parameters
    ----------
    xt_En : float
        Mole fraction of Enstatite in ternary coordinates.
    xt_Fs : float
        Mole fraction of Ferrosilite in ternary coordinates.
    xt_Wo : float
        Mole fraction of Wollastonite in ternary coordinates.

    Returns
    -------
    z_min : float
        Minimum valid value of z.
    z_max : float
        Maximum valid value of z.

    Notes
    -----
    If z_min > z_max, the point lies outside the pyroxene quadrilateral.
    """
    if np.isscalar(xt_En) and np.isscalar(xt_Fs):
        z_min = max(0.0,
                    2.0 * (xt_Wo - xt_Fs),
                    2.0 * xt_Wo - 1.0,
                    2.0 * (xt_En - 1.0))
    
        z_max = min(1.0,
                    2.0 * xt_Wo,
                    2.0 * xt_En,
                    2.0 * (1.0 - xt_Fs + xt_Wo))
    else:
        z_min = np.maximum(0.0,
                    np.maximum(2.0 * (xt_Wo - xt_Fs),
                    np.maximum(2.0 * xt_Wo - 1.0,
                    2.0 * (xt_En - 1.0))))
    
        z_max = np.minimum(1.0,
                    np.minimum(2.0 * xt_Wo,
                    np.minimum(2.0 * xt_En,
                    2.0 * (1.0 - xt_Fs + xt_Wo))))

    return z_min, z_max

def quad_to_ternary(xq_En, xq_Fs, xq_Di, xq_Hd):
    """
    Transform pyroxene quadrilateral coordinates (En-Fs-Di-Hd) back to
    ternary coordinates (En-Fs-Wo).

    The mapping follows directly from the positions of the quadrilateral
    end-members in the ternary:
      En at (1.0, 0.0, 0.0)
      Fs at (0.0, 1.0, 0.0)
      Di at (0.5, 0.0, 0.5)
      Hd at (0.0, 0.5, 0.5)

    Unlike the inverse (ternary_to_quadrilateral), this direction is unique:
    no extra parameter is needed.

    Parameters
    ----------
    xq_En : float
        Mole fraction of Enstatite in quadrilateral coordinates.
    xq_Fs : float
        Mole fraction of Ferrosilite in quadrilateral coordinates.
    xq_Di : float
        Mole fraction of Diopside in quadrilateral coordinates.
    xq_Hd : float
        Mole fraction of Hedenbergite in quadrilateral coordinates.

    Returns
    -------
    xt_En : float
        Mole fraction of Enstatite in ternary coordinates.
    xt_Fs : float
        Mole fraction of Ferrosilite in ternary coordinates.
    xt_Wo : float
        Mole fraction of Wollastonite in ternary coordinates.
    """
    xt_En = xq_En + 0.5 * xq_Di
    xt_Fs = xq_Fs + 0.5 * xq_Hd
    xt_Wo = 0.5 * xq_Di + 0.5 * xq_Hd

    return xt_En, xt_Fs, xt_Wo

def z_random_mixing(xt_En, xt_Fs, xt_Wo):
    """
    Compute the value of z (= xq_Di) corresponding to a purely random
    (ideal) cation configuration, given ternary coordinates.

    In a random arrangement, Ca and Mg/Fe are distributed independently
    across their respective sites, so the probability of a Di-type unit
    (Ca on M2, Mg on M1) is the product of the independent site fractions:

        X_Ca  = 2 * xt_Wo               (Ca fraction on M2)
        X_Mg  = xt_En / (xt_En + xt_Fs) (Mg fraction among non-Ca sites)
        z     = X_Ca * X_Mg

    Parameters
    ----------
    xt_En : float
        Mole fraction of Enstatite in ternary coordinates.
    xt_Fs : float
        Mole fraction of Ferrosilite in ternary coordinates.
    xt_Wo : float
        Mole fraction of Wollastonite in ternary coordinates.

    Returns
    -------
    z : float
        The value of xq_Di for a random cation configuration.

    Notes
    -----
    Undefined (division by zero) when xt_En = xt_Fs = 0, i.e. at the
    Wollastonite vertex, which lies outside the quadrilateral anyway.
    """
    X_Ca = 2.0 * xt_Wo
    X_Mg = xt_En / (xt_En + xt_Fs)
    return X_Ca * X_Mg

def quad_to_site_occupancy(xq_En, xq_Fs, xq_Di, xq_Hd):
    """
    Compute M1 and M2 site occupancies for a pyroxene in the
    Diopside - Hedenbergite - Enstatite - Ferrosilite quadrilateral.

    Note however that the xq_Di, xq_En, xq_Hd, xq_Fs are the
    more fundamental parameters, as these are the representations
    of the site pairings. The site occupancies are not necessarily
    a sufficient description to know how the site occupancies are
    paired up. But in this particular case they, in fact, are,
    because Ca is always on the same (M2) site, and MgFe pairs
    do not exist (are thermodynamically unstable). 
    
    Parameters
    ----------
    xq_Di : float  Molar fraction of Diopside       (M2=Ca, M1=Mg)
    xq_En : float  Molar fraction of Enstatite      (M2=Mg, M1=Mg)
    xq_Hd : float  Molar fraction of Hedenbergite   (M2=Ca, M1=Fe)
    xq_Fs : float  Molar fraction of Ferrosilite    (M2=Fe, M1=Fe)

    Returns
    -------
    dict with M1 and M2 site occupancies (each summing to 1).
    """
    assert abs(xq_Di + xq_En + xq_Hd + xq_Fs - 1.0) < 1e-9, \
        "Molar fractions must sum to 1."

    # M2 site
    xm2_Ca = xq_Di + xq_Hd
    xm2_Mg = xq_En
    xm2_Fe = xq_Fs

    # M1 site
    xm1_Mg = xq_Di + xq_En
    xm1_Fe = xq_Hd + xq_Fs

    return {
        "xm2_Ca": xm2_Ca,
        "xm2_Mg": xm2_Mg,
        "xm2_Fe": xm2_Fe,
        "xm1_Mg": xm1_Mg,
        "xm1_Fe": xm1_Fe,
    }

def site_occupancy_to_quad(xm1_Mg, xm1_Fe, xm2_Ca, xm2_Mg, xm2_Fe):
    """
    Invert M1/M2 site occupancies back to molar fractions in the
    Diopside - Hedenbergite - Enstatite - Ferrosilite quadrilateral.

    Parameters
    ----------
    xm1_Mg : float  M1 occupancy of Mg
    xm1_Fe : float  M1 occupancy of Fe
    xm2_Ca : float  M2 occupancy of Ca
    xm2_Mg : float  M2 occupancy of Mg
    xm2_Fe : float  M2 occupancy of Fe

    Returns
    -------
    dict with molar fractions xq_Di, xq_En, xq_Hd, xq_Fs (summing to 1).
    """
    assert abs(xm1_Mg + xm1_Fe - 1.0) < 1e-9, "M1 occupancies must sum to 1."
    assert abs(xm2_Ca + xm2_Mg + xm2_Fe - 1.0) < 1e-9, "M2 occupancies must sum to 1."

    xq_En = xm2_Mg
    xq_Fs = xm2_Fe
    xq_Di = xm1_Mg - xm2_Mg
    xq_Hd = xm1_Fe - xm2_Fe

    assert abs(xq_Di + xq_En + xq_Hd + xq_Fs - 1.0) < 1e-9, \
        "Resulting molar fractions do not sum to 1 — input occupancies may be inconsistent."

    return {
        "xq_Di": xq_Di,
        "xq_En": xq_En,
        "xq_Hd": xq_Hd,
        "xq_Fs": xq_Fs,
    }

if __name__=='__main__':

    xt_En = 0.6
    xt_Fs = 0.1
    xt_Wo = 1-xt_En-xt_Fs
    assert xt_Wo<=0.5, 'Error: out of range of the quadrilateral'

    zmin,zmax = z_valid_range(xt_En, xt_Fs, xt_Wo)
    zran      = z_random_mixing(xt_En, xt_Fs, xt_Wo)
    assert zran>=zmin
    assert zran<=zmax
    print(zmin,zmax,zran)

    z = zran
    xq_En, xq_Fs, xq_Di, xq_Hd = ternary_to_quad(xt_En, xt_Fs, xt_Wo, z)
    print(xq_En, xq_Fs, xq_Di, xq_Hd)
    print(quad_to_ternary(xq_En, xq_Fs, xq_Di, xq_Hd))
    print(quad_to_site_occupancy(xq_En, xq_Fs, xq_Di, xq_Hd))
