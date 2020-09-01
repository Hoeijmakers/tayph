
def test_ccf():
    """This tests the ccf routine by creating 3 orders synthetic comb of spectral
    lines with equal depths of 1,3 and 5. It proves that the cross-correlation
    truly acts as a weighted average.

    It first defines wavelength grids for a mock template and mock data, which
    both derive from the same comb of spectral lines. This comb is blurred to
    resembles boxes with a narrow width, to represent the template,
    and Gaussian lines with a wider width to represent data.

    The 'data' are repeated into matrices with height 30 to simulate spectral orders,
    and 3 such 'orders' are passed to the cross-correlation function. Multiplied
    with strengths of 1, 3 and 5. The template is kept uniform with line depth 1.
    The expected answer is that the CCF measures an average line with depth 3.
    (actually slightly less because the template has a finite width, so it slightly
    smudges the Gaussian line core, but this is an effect on the 1% level).

    The cross-correlation function is tested again with data with Gaussian noise
    added, to prove that extra input (the noise of each pixel) is accepted, that
    noise on the CCF is returned, and that the resulting noise on the CCF goes
    down by a factor close to the sqrt of the number of points in the template that
    are 1.0. (equal to sqrt(tsums) because the template was constructed to be a set of
    boxes with depth 1).

    """

    # import sys
    # sys.path.insert(0,'../../../tayph')
    # from tayph.ccf import xcor
    # from tayph.operations import smooth
    # from tayph.functions import findgen, rebinreform

    from ..ccf import xcor
    from ..operations import smooth
    from ..functions import findgen, rebinreform
    import matplotlib.pyplot as plt
    import numpy as np
    import astropy.io.fits as fits
    import scipy.interpolate as interpolate
    import math
    from numpy.random import normal
    c=3e5
    snr = 10.0
    #The following tests XCOR on synthetic orders.
    wl = findgen(4096)*0.00115+600
    wlm = findgen(2e6)/10000.0+550.0
    fxm = wlm*0.0
    fxm[[(findgen(400)*4e3+1e3).astype(int)]] = 1.0
    px_scale=wlm[1]-wlm[0]
    dldv = np.min(wlm) / c /px_scale
    T=smooth(fxm,dldv * 4.0)#This makes boxes with a
    T=T/np.max(T)#This sets the lines to 1.
    fxm_b=smooth(fxm,dldv * 20.0,mode='gaussian')
    dspec = interpolate.interp1d(wlm,fxm_b)(wl)
    noise = normal(loc=0.0, scale=1/snr, size=len(wl))
    dspec = dspec/np.max(dspec)
    order = rebinreform(dspec,30)
    noisy_order = rebinreform(dspec+noise,30)
    noisy_order_error = noisy_order*0.0+1/snr
    rv,ccf,tsums=xcor([wl,wl,wl],[order,order*3.0,order*5.0],wlm,T,1.0,100.0)
    rv2,ccf2,ccf2_e,tsums2=xcor([wl,wl,wl],[noisy_order,noisy_order*3.0,noisy_order*5.0],wlm,T,1.0,100.0,list_of_errors=[noisy_order_error,noisy_order_error,noisy_order_error])
    assert np.median(ccf) == 0.0#Prove that at velocities where the data is 0, the CCF is zero.
    assert np.sum(np.abs(tsums-tsums2)) == 0#Prove that running the CCF with and without noise returns the same template normalization.
    assert math.isclose(np.max(ccf),3.0, rel_tol=1e-2)#Prove that the peak of the average line is equal to the average line strength in the data, i.e. that the CCF truly acts as a weighted average operator.
    assert math.isclose(np.mean(ccf2_e),1/snr/np.sqrt(np.mean(tsums2)),rel_tol=1e-2)#Prove that Gaussian noise is correctly propagated through the CCF.


# test_ccf()
