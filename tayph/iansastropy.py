def integral_smallplanet_nonlinear(z, p, cn, lower, upper):
    import numpy as np
    """Return the integral in I*(z) in Eqn. 8 of Mandel & Agol (2002).
    -- Int[I(r) 2r dr]_{z-p}^{1}, where:

    :INPUTS:
         z = scalar or array.  Distance between center of star &
             planet, normalized by the stellar radius.

         p = scalar.  Planet/star radius ratio.

         cn = 4-sequence.  Nonlinear limb-darkening coefficients,
              e.g. from Claret 2000.

         lower, upper -- floats. Limits of integration in units of mu

    :RETURNS:
         value of the integral at specified z.

         """
    # 2010-11-06 14:12 IJC: Created
    # 2012-03-09 08:54 IJMC: Added a cheat for z very close to zero

    #import pdb

    #z = np.array(z, copy=True)
    #z[z==0] = zeroval
    #a = (z - p)**2
    lower = np.array(lower, copy=True)
    upper = np.array(upper, copy=True)
    return eval_int_at_limit(upper, cn) - eval_int_at_limit(lower, cn)

def eval_int_at_limit(limit, cn):
    import numpy as np
    """Evaluate the integral at a specified limit (upper or lower)"""
    # 2013-04-17 22:27 IJMC: Implemented some speed boosts; added a
    #                        bug; fixed it again.

    # The old way:
    #term1 = cn[0] * (1. - 0.8 * np.sqrt(limit))
    #term2 = cn[1] * (1. - (2./3.) * limit)
    #term3 = cn[2] * (1. - (4./7.) * limit**1.5)
    #term4 = cn[3] * (1. - 0.5 * limit**2)
    #goodret = -(limit**2) * (1. - term1 - term2 - term3 - term4)

    # The new, somewhat faster, way:
    sqrtlimit = np.sqrt(limit)
    sqlimit = limit*limit
    total = 1. - cn[0] * (1. - 0.8 * sqrtlimit)
    total -= cn[1] * (1. - (2./3.) * limit)
    total -= cn[2] * (1. - (4./7.) * limit*sqrtlimit)
    total -= cn[3] * (1. - 0.5 * sqlimit)
    ret = -(sqlimit) * total

    return ret


def occultnonlin_small(z,p, cn):
    """Nonlinear limb-darkening light curve in the small-planet
    approximation (section 5 of Mandel & Agol 2002).

    :INPUTS:
        z -- sequence of positional offset values

        p -- planet/star radius ratio

        cn -- four-sequence nonlinear limb darkening coefficients.  If
              a shorter sequence is entered, the later values will be
              set to zero.

    :NOTE:
       I had to divide the effect at the near-edge of the light curve
       by pi for consistency; this factor was not in Mandel & Agol, so
       I may have coded something incorrectly (or there was a typo).

    :EXAMPLE:
       ::

         # Reproduce Figure 2 of Mandel & Agol (2002):
         from pylab import *
         import transit
         z = linspace(0, 1.2, 100)
         cns = vstack((zeros(4), eye(4)))
         figure()
         for coef in cns:
             f = transit.occultnonlin_small(z, 0.1, coef)
             plot(z, f, '--')

    :SEE ALSO:
       :func:`t2z`
    """
    # 2010-11-06 14:23 IJC: Created
    # 2011-04-19 15:22 IJMC: Updated documentation.  Renamed.
    # 2011-05-24 14:00 IJMC: Now check the size of cn.
    # 2012-03-09 08:54 IJMC: Added a cheat for z very close to zero
    # 2013-04-17 10:51 IJMC: Mild code optimization

    #import pdb
    import numpy as np

    cn = np.array([cn], copy=False).ravel()
    if cn.size < 4:
        cn = np.concatenate((cn, [0.]*(4-cn.size)))

    z = np.array(z, copy=False)
    F = np.ones(z.shape, float)
    eps = np.finfo(float).eps
    zeroval = eps*1e6
    z[z==0] = zeroval # cheat!

    a = (z - p)**2
    b = (z + p)**2
    c0 = 1. - np.sum(cn)
    Omega = 0.25 * c0 + np.sum( cn / np.arange(5., 9.) )

    ind1 = ((1. - p) < z) * ((1. + p) > z)
    ind2 = z <= (1. - p)

    # Need to specify limits of integration in terms of mu (not r)
    aind1 = 1. - a[ind1]
    zind1m1 = z[ind1] - 1.
    #pdb.set_trace()

    #if c_integral_smallplanet_nonlinear:
        #print 'do it the C way'
    #    Istar_edge = _integral_smallplanet_nonlinear.integral_smallplanet_nonlinear(cn, np.sqrt(aind1), np.array([0.])) / aind1
    #    Istar_inside = _integral_smallplanet_nonlinear.integral_smallplanet_nonlinear(cn, np.sqrt(1. - a[ind2]), np.sqrt(1. - b[ind2])) / z[ind2]
    #else:
    Istar_edge = integral_smallplanet_nonlinear(None, p, cn, \
                                                    np.sqrt(aind1), np.array([0.])) / aind1

    Istar_inside = integral_smallplanet_nonlinear(None, p, cn, \
                                              np.sqrt(1. - a[ind2]), \
                                              np.sqrt(1. - b[ind2])) / \
                                              (z[ind2])


    term1 = 0.25 * Istar_edge / (np.pi * Omega)
    term2 = p*p * np.arccos((zind1m1) / p)
    term3 = (zind1m1) * np.sqrt(p*p - (zind1m1*zind1m1))

    F[ind1] = 1. - term1 * (term2 - term3)
    F[ind2] = 1. - 0.0625 * p * Istar_inside / Omega

    #pdb.set_trace()
    return F
