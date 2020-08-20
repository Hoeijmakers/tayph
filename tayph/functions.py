__all__ = [
    'selmax',
    'running_MAD_2D',
    'rebinreform',
    'nan_helper',
    'findgen',
    'gaussian',
    'sigma_clip'
]


def selmax(y_in,p,s=0.0):
    """This program returns the p (fraction btw 0 and 1) highest points in y,
    ignoring the very top s % (default zero, i.e. no points ignored), for the
    purpose of outlier rejection."""
    import lib.utils as ut
    import numpy as np
    import copy
    ut.postest(p)
    y=copy.deepcopy(y_in)#Copy itself, or there are pointer-related troubles...
    if s < 0.0:
        raise Exception("ERROR in selmax: s should be zero or positive.")
    if p >= 1.0:
        raise Exception("ERROR in selmax: p should be strictly between 0.0 and 1.0.")
    if s >= 1.0:
        raise Exception("ERROR in selmax: s should be strictly less than 1.0.")
    ut.postest(-1.0*p+1.0)
    #ut.nantest('y in selmax',y)
    ut.dimtest(y,[0])#Test that it is one-dimensional.
    y[np.isnan(y)]=np.nanmin(y)#set all nans to the minimum value such that they will not be selected.
    y_sorting = np.flipud(np.argsort(y))#These are the indices in descending order (thats why it gets a flip)
    N=len(y)
    if s == 0.0:
        max_index = np.max([int(round(N*p)),1])#Max because if the fraction is 0 elements, then at least it should contain 1.0
        return y_sorting[0:max_index]

    if s > 0.0:
        min_index = np.max([int(round(N*s)),1])#If 0, then at least it should be 1.0
        max_index = np.max([int(round(N*(p+s))),2]) #If 0, then at least it should be 1+1.
        return y_sorting[min_index:max_index]


def running_MAD_2D(z,w):
    """Computers a running standard deviation of a 2-dimensional array z.
    The stddev is evaluated over the vertical block with width w pixels.
    The output is a 1D array with length equal to the width of z."""
    import astropy.stats as stats
    import numpy as np
    from tayph.vartests import typetest,dimtest,postest
    typetest(z,np.ndarray,'z in fun.running_MAD_2D()')
    dimtest(z,[0,0],'z in fun.running_MAD_2D()')
    typetest(w,[int,float],'w in fun.running_MAD_2D()')
    postest(w,'w in fun.running_MAD_2D()')
    size = np.shape(z)
    ny = size[0]
    nx = size[1]
    s = findgen(nx)*0.0
    for i in range(nx):
        minx = max([0,i-int(0.5*w)])
        maxx = min([nx-1,i+int(0.5*w)])
        s[i] = stats.mad_std(z[:,minx:maxx],ignore_nan=True)
    return(s)

def rebinreform(a,n):
    """
    This works like the rebin(reform()) trick in IDL, where you use fast
    array manipulation operations to transform a 1D array into a 2D stack of itself,
    to be able to do operations on another 2D array by multiplication/addition/division
    without having to loop through the second dimension of said array.
    """
    import numpy as np
    return(np.transpose(np.repeat(np.expand_dims(a,1),n,axis=1)))


def nan_helper(y):
    """
    Helper function to handle indices and logical indices of NaNs.

    Input:
        - y, 1D (!!) numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    #This comes from https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    # Lets define first a simple helper function in order to make it more straightforward to handle indices and logical indices of NaNs:
    # Now the nan_helper(.) can now be utilized like:
    #
    # >>> y= array([1, 1, 1, NaN, NaN, 2, 2, NaN, 0])
    # >>>
    # >>> nans, x= nan_helper(y)
    # >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    # >>>
    # >>> print y.round(2)
    # [ 1.    1.    1.    1.33  1.67  2.    2.    1.    0.  ]
    import numpy as np
    from tayph.vartests import dimtest
    dimtest(y,[0],'y in fun.nan_helper()')
    return np.isnan(y), lambda z: z.nonzero()[0]

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
    typetest(array,[list,np.ndarray],'array in fun.sigma_clip()')
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
