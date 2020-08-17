def findgen(n,integer=False):
    """This is basically IDL's findgen function.
    a = findgen(5) will return an array with 5 elements from 0 to 4:
    [0,1,2,3,4]
    """
    import numpy as np
    from tayph.vartests import typetest,postest
    typetest(n,[int,float],'n in findgen()')
    typetest(integer,bool,'integer in findgen()')
    postest(n,'n in findgen()')
    n=int(n)
    if integer:
        return np.linspace(0,n-1,n).astype(int)
    else:
        return np.linspace(0,n-1,n)


def gaussian(x,A,mu,sig,cont=0.0):
    import numpy as np
    """This produces a gaussian function on the grid x with amplitude A, mean mu
    and standard deviation sig. No testing is done, to keep it fast."""
    return A * np.exp(-0.5*(x - mu)/sig*(x - mu)/sig)+cont


def sigma_clip(array,nsigma=3.0,MAD=False):
    """This returns the n-sigma boundaries of an array, mainly used for scaling plots.

    Parameters
    ----------
    array : list, np.ndarray
        The array from which the n-sigma boundaries are required.

    nsigma : int, float
        The number of sigma's away from the mean that need to be provided.

    MAD : bool
        Use the true standard deviation or MAD estimator of the standard deviation
        (works better in the presence of outliers).

    Returns
    -------
    vmin,vmax : float
        The bottom and top n-sigma boundaries of the input array.
    """
    from tayph.vartests import typetest
    import numpy as np
    typtest(array,[list,np.ndarray],'array in fun.sigma_clip()')
    typetest(nsigma,[int,float],'nsigma in fun.sigma_clip()')
    typetest(MAD,bool,'MAD in fun.sigma_clip()')
    m = np.nanmedian(array)
    if MAD:
        from astropy.stats import mad_std
        s = mad_std(array,ignore_nan=True)
    else:
        s = np.nanstd(array)
    vmin = m-nsigma*s
    vmax = m+nsigma*s
    return vmin,vmax
