__all__ = [
    'eval_poly',
    'ladfit',
    'box',
    'voigt',
    'rotation_broadened_line',
    'fit_rotation_broadened_line',
    'selmax',
    'running_MAD_2D',
    'running_MAD',
    'strided_window',
    'running_median_2D',
    'running_std_2D',
    'running_mean_2D',
    'rebinreform',
    'nan_helper',
    'findgen',
    'gaussian',
    'gaussfit',
    'sigma_clip',
    'local_v_star',
    'polysinfit',
    'sysrem'
]


def eval_poly(x,f):
    """This evaluates a polynomial using fitting coefficients f that were found via np.polyfit(x,y).
    Normally you could use numpy.polynomial.polynomial.Polynomlial(fit)(x) for this, but not if fit
    is a 2D array found by using np.polyfit(x,Y) with Y 2D. To deal with this eval_poly evaluates a
    sequence of polynomials for a sequence of fits, on x.  If you want to rip this out of tayph for
    your own purposes, all you need to remove is the test functions.

    Parameters
    ----------
    x : np.ndarray
        A 1D array with x values.

    f : np.ndarray
        A 1D or 2D array containing polynomial coefficients.


    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> a = [1.0,2.0]
    >>> b = [0.0,0.0]
    >>> x = np.linspace(0,5,9)
    >>> Y = np.array([a[0]*x+b[0],a[1]*x+b[1]]).T
    >>> fit = np.polyfit(x,Y)
    >>> prediction = eval_poly(x,fit)
    >>> plt.plot(x,Y,'.',color='black')
    >>> plt.plot(x,prediction)
    >>> plt.show()
    """
    import numpy as np

    #Test functions:
    from tayph.vartests import typetest,dimtest
    typetest(x,np.ndarray)
    typetest(f,np.ndarray)
    dimtest(x,[0])
    #End test functions.

    if np.ndim(f) not in [1,2]:
        raise Exception("f in eval_poly should be 1D or 2D.")

    out = 0
    powers = np.arange(len(f))[::-1]
    X = np.repeat(x[:,None].astype(float),len(f),axis=1)
    return((X**powers @ f))


def ladfit(x,y,t=None):
    """
    This is a wrapper for LAD regression with sklego, implemented in a way such that
    it returns the linear coefficients a,b from y=ax+b.
    x and y need to be one-dimensional arrays.

    LADRegression accepts multi-dimensional arrays but that is for doing multivariate
    linear regression, not independent regression for different datasets y defined on
    the same grid x. So I could not get this to work for a multidimensional Y array
    without looping.

    The algoritm returns the model coefficients [a,b], either with 2 or mx2 elements.
    This behavior is made to match np.polyfit.

    Doing a linear fit with ladfit(x,y.T) is cognate with np.polyfit(x,y,1).
    The extra transpose is there so that ladfit works intuitively on an n x m frame for which the
    x axis is horizontal (e.g. a spectral order).


    Set the parameter t to a threshold number of sigma values that are considered outliers.
    The algorithm will do an initial fit, flag residuals that are t*sigma away from 0.0 as outliers,
    mask these, and do the fit again. Doing this will return a second variable that has the same
    shape as y, filled with 1.0s for values that are good, and 0.0s for outliers. This can be used
    as an outlier mask.

    If you want to rip this out of tayph for your own purposes, all you need to remove is
    the test functions.

    Parameters
    ----------
    x : np.ndarray
        A 1D array with x values.

    y : np.ndarray
        A 1D or 2D array containing dependent variable.


    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import tayph.functions as fun
    >>> a = [1.0,2.0]
    >>> b = [0.0,0.0]
    >>> x = np.linspace(0,5,9)
    >>> Y = np.array([a[0]*x+b[0],a[1]*x+b[1]]).T
    >>> fit=ladfit(x,Y)
    >>> prediction = fun.eval_poly(x,fit)
    >>> plt.plot(x,Y,'.',color='black')
    >>> plt.plot(x,prediction)
    >>> plt.show()
    """
    import numpy as np
    from sklego.linear_model import LADRegression
    import astropy.stats as stats
    #Test functions:
    from tayph.vartests import typetest,dimtest
    typetest(x,np.ndarray)
    typetest(y,np.ndarray)
    dimtest(x,[0])
    #End test functions.

    x = x.reshape(1,-1)

    if np.ndim(y) not in [1,2]:
        raise Exception("y in ladfit should be 1D or 2D.")

    if t:
        import astropy.stats as stats #Need #stats.mad_std
        mask = y * 0.0 + 1.0 #Emtpy array to store outliers. Agnostic as to shape of y.
    if np.ndim(y) > 1:
        yT = y
        a,b = [],[]
        for i in range(len(yT)):
            l = LADRegression().fit(x.T,yT[i])
            ai,bi = l.coef_[0],l.intercept_
            if t:#Do an iteration
                res = yT[i]-ai*x[0]-bi
                outliers = np.abs(res) > t * stats.mad_std(res,ignore_nan=True)
                l = LADRegression().fit(x.T[~outliers],yT[i][~outliers])
                ai,bi = l.coef_[0],l.intercept_
                mask[i,outliers] = 0.0
            a.append(ai)
            b.append(bi)
        if t:
            return(np.array([a,b]),mask)
        else:
            return(np.array([a,b]))
    else:
        l = LADRegression().fit(x.T,y)
        a,b = l.coef_[0],l.intercept_
        if t:#Do an iteration
            res = y-a*x[0]-b
            outliers = np.abs(res) > t * stats.mad_std(res,ignore_nan=True)
            l = LADRegression().fit(x.T[~outliers],y[~outliers])
            a,b = l.coef_[0],l.intercept_
            mask[outliers] = 0.0
            return(np.array([a,b]),mask)
        else:
            return(np.array([a,b]))


def box(x,A,c,w):
    """
    This function computes a box with width w, amplitude A and center c
    on grid x. It would be a simple multiplication of two heaviside functions,
    where it not that I am taking care to interpolate the edge pixels.
    And because I didn't want to do the cheap hack of oversampling x by a factor
    of many and then using np.interpolate, I computed the interpolation manually
    myself. This function guarantees* to preserve the integral of the box, i.e.
    np.sum(box(x,A,c,w))*dwl equals w, unless the box is truncated by the edge.)

    Uncomment the two relevant lines below if you don't believe me and
    want to make sure.
    """
    from tayph.vartests import typetest
    import numpy as np
    #import matplotlib.pyplot as plt
    #import pdb

    typetest(x,np.ndarray,'x in fun.box()')
    typetest(A,[int,float],'x in fun.box()')
    typetest(c,[int,float],'x in fun.box()')
    typetest(w,[int,float],'x in fun.box()')
    A=float(A)
    c=float(c)
    w=float(w)

    y=x*0.0

    #The following manually does the interpolation of the edge pixels of the box.
    #First do the right edge of the box.
    r_dist=x-c-0.5*w
    r_px=np.argmin(np.abs(r_dist))
    r_mindist=r_dist[r_px]
    y[0:r_px]=1.0
    if r_px != 0 and r_px != len(x)-1:#This only works if we are not at an edge.
        dpx=abs(x[r_px]-x[r_px-int(np.sign(r_mindist))])#Distance to the
        #previous or next pixel, depending on whether the edge falls to the left
        #(posive mindist) or right (negative mindist) of the pixel center.

    #If we are an edge, dpx needs to be approximated from the other neigboring px.
    elif r_px == 0:
        dpx=abs(x[r_px+1]-x[0])
    else:
        dpx=abs(x[r_px]-x[r_px-1])

    if w/dpx < 2.0:
        raise Exception("InputError in fun.box: Width too small (< 2x sampling rate).")
    frac=0.5-r_mindist/dpx
    #If mindist = 0, then frac = 0.5.
    #If mindist = +0.5*dpx, then frac=0.0
    #If mindist = -0.5*dpx, then frac=1.0
    y[r_px] = np.clip(frac,0,1)#Set the pixel that overlaps with the edge to the
    #fractional distance that the edge is away from the px center. Clippig needs
    #to be done to take into account the edge pixels, in which case frac can be
    #larger than 1.0 or smaller than 0.0.

    #And do the same for the left part. Note the swapping of two signs.
    l_dist=x-c+0.5*w
    l_px=np.argmin(np.abs(l_dist))
    l_mindist=l_dist[l_px]
    y[0:l_px]=0.0
    if l_px != 0 and l_px != len(x)-1:#This only works if we are not at an edge.
        dpx=abs(x[l_px]-x[l_px-int(np.sign(l_mindist))])
    elif l_px == 0:
        dpx=abs(x[l_px+1]-x[0])
    else:
        dpx=abs(x[l_px]-x[l_px-1])
    frac=0.5+l_mindist/dpx
    y[l_px] = np.clip(frac,0,1)
    #If mindist = 0, then frac = 0.5.
    #If mindist = +0.5*dpx, then frac=1.0
    #If mindist = -0.5*dpx, then frac=0.0

    #print([w,np.sum(y)*dpx]) #<=This line compares the integral.
    #In perfect agreement!!

    #plt.plot(x,y,'.')
    #plt.axvline(x=c+0.5*w)
    #plt.axvline(x=c-0.5*w)
    #plt.ion()
    #plt.show()
    return y*A


def voigt(x,x0,sigma, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWFM gamma
    and Gaussian sigma.


    Adopted from https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/ as a faster
    alternative (approx 6x) to astropy's Voigt1D in default.
    """
    from scipy.special import wofz
    import numpy as np
    return np.real(wofz(((x-x0) + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)





def rotation_broadened_line(RV,*args):
    import numpy as np
    from astropy.modeling.models import Voigt1D
    p=[i for i in args]


    A=p[0]
    vsys=p[1]
    vsini=p[2]
    E=p[3]
    s=p[4]
    g=p[5]
    c=p[6:][::-1]#polynomial, flipped for np.poly.

    vz_grid = RV-RV[:,np.newaxis]#Broadcasting trick here to do the convolution below.
    Dv = np.array(vz_grid/vsini).clip(-1,1)#Compute the Delta-v parameter, clipped to -1,1 to avoid
    #negative values of G. This is 2D.

    # H = gaussian(RV,1.0,vsys,s) / (s*np.sqrt(2*np.pi))
    H = voigt(RV,vsys,2*np.sqrt(2*np.log(2))*s,2*g)
    G0 = (2*(1-E)+0.5*np.pi*E)
    G = (2*(1-E)*np.sqrt((1-Dv**2))+0.5*np.pi*E*(1-Dv**2))# /(np.pi*vsini*(1-E/3)) <--- remove
    #this because it is a constant.
    return(H@G / G0 * A + np.poly1d(c)(RV))







# def rotation_broadened_line_gauss(RV,*args):
#     import numpy as np
#     from astropy.modeling.models import Voigt1D
#     p=[i for i in args]
#     A=p[0]
#     vsys=p[1]
#     vsini=p[2]
#     E=p[3]
#     s=p[4]
#     c=p[5:][::-1]#polynomial, flipped for np.poly.
#
#     vz_grid = RV-RV[:,np.newaxis]#Broadcasting trick here to do the convolution below.
#     Dv = np.array(vz_grid/vsini).clip(-1,1)#Compute the Delta-v parameter, clipped to -1,1 to avoid
#     #negative values of G. This is 2D.
#
#     H = gaussian(RV,1.0,vsys,s) / (s*np.sqrt(2*np.pi))
#     # H = Voigt1D(x_0=vsys, amplitude_L= 2 / np.pi / 2 / g, fwhm_L=2*g, fwhm_G=2*np.sqrt(2*np.log(2))*s)
#     G0 = (2*(1-E)+0.5*np.pi*E)
#     G = (2*(1-E)*np.sqrt((1-Dv**2))+0.5*np.pi*E*(1-Dv**2))# /(np.pi*vsini*(1-E/3)) <--- remove
#     #this because it is a constant.
#     return(H@G / G0 * A + np.poly1d(c)(RV))


def fit_rotation_broadened_line(RV,CCF,degree=1,startparams=[],fixparams=[]):
    """
    This fits a rotation broadened profile following Gray (2005), with a polynomial for the
    continuum. The line profile is fit by a Voigt with a sigma (to approximate some level
    of turbulent broadening and spectral resolution) and gamma, with the analytical profile derived
    by Gray on page 465 of the 2005 edition.

    Returns a list of best fit values for, respectively:
    A, line amplitude,
    vsys, systemic / centroid velocity (km/s, or same as RV-axis),
    vsini, projected stellar equatorial rotation velocity (km/s, or same as RV-axis),
    E, the linear limb darkening coefficient,
    sigma, the Gaussian sigma-width of the unbroadened line profile,
    gamma, the Lorenzian width parameter,
    plus polynomial coefficients describing the continuum, starting with offset, 1st order, etc.
    """
    import scipy.optimize
    import numpy as np
    import tayph.functions as fun
    import tayph.util as ut
    import pdb
    from astropy.modeling.models import Voigt1D


    if len(startparams)==0:
        A_start = np.min(CCF)-np.max(CCF) #Assumes feature in negative direction
        vsys_start = RV[np.argmin(CCF)]
        vsini_start = 40.0#km/s
        E_start = 0.5#Limb darkening parameter
        sigma_start = 1.5#km/s, Gaussian FWHM, for resolution and turbulence.
        gamma_start = 0
        p_start = [np.median(CCF)]+degree*[0]
        # if voigt:
        startparams = [A_start,vsys_start,vsini_start,E_start,sigma_start,gamma_start]+p_start
        # else:
        #     startparams = [A_start,vsys_start,vsini_start,E_start,sigma_start]+p_start
        if len(fixparams) > 0:
            raise Exception('Fix params should not be provided if no startparams are provided.')
    elif len(startparams) < 6:
        raise Exception(f"Error: Start parameters has length less than 6 ({len(startparams)})."
        "Define startparams as A,vsys,vsini,E,sigma,offset and one or more higher degree terms.")
    else:
        p_start = startparams[6:]#Make this for use in setting bounds below.
        if len(fixparams) >0 and len(fixparams) != len(startparams):
            raise Exception("Fix params is set, but should have the same number of "
            "elements as startparms.")
        if len(fixparams)>0:
            for fp in fixparams:
                if fp not in [0,1]:
                    raise Exception("fixparams should be set to a list containing either 0 or 1")
    # pdb.set_trace()
    #https://stackoverflow.com/questions/4092528/how-to-clamp-an-integer-to-some-range

    vz_grid = RV-RV[:,np.newaxis]#Broadcasting trick here to do the convolution below.
    #This can be defined outside of the optimization. It is big, 2D.
    def eval_rotation_broadened_line(p,vz,ccf,vz_grid):
        """The parameters are:
        Amplitude
        vsys
        vsini
        E (limb darkening)
        sigma (R+macroturbulence)
        gamma (lorenzian width)
        polynomial as y=p1+x*p2+x^2*p3+...
        """

        A=p[0]
        vsys=p[1]
        vsini=p[2]
        E=p[3]
        s=p[4]#sigma
        g=p[5]#gamma
        c=p[6:][::-1]#polynomial, flipped for np.poly.

        Dv = np.array(vz_grid/vsini).clip(-1,1)#Compute the Delta-v parameter, clipped to -1,1 to
        #avoid negative values of G.
        # H = gaussian(vz,1.0,vsys,s) / (s*np.sqrt(2*np.pi))


        # H = Voigt1D(x_0=vsys, amplitude_L= 2 / np.pi / 2 / g, fwhm_L=2*g, fwhm_G=2*np.sqrt(2*np.log(2))*s)(vz)
        H = voigt(vz,vsys,2*np.sqrt(2*np.log(2))*s,2*g)
        #G is defined on every position on the star because Dv is 2D.
        G0 = (2*(1-E)+0.5*np.pi*E)
        G = (2*(1-E)*np.sqrt((1-Dv**2))+0.5*np.pi*E*(1-Dv**2))# /(np.pi*vsini*(1-E/3)) <--- remove
        #this because it is a constant. This is 2D.
        diff = H@G / G0 * A + np.poly1d(c)(vz) - ccf #H@G/G0 is normalised from 0 to 1.
        return(diff)

    # def eval_rotation_broadened_line_gauss(p,vz,ccf,vz_grid):
    #     """The parameters are:
    #     Amplitude
    #     vsys
    #     vsini
    #     E (limb darkening)
    #     sigma (R+macroturbulence)
    #     polynomial as y=p1+x*p2+x^2*p3+...
    #     """
    #
    #     A=p[0]
    #     vsys=p[1]
    #     vsini=p[2]
    #     E=p[3]
    #     s=p[4]#sigma
    #     c=p[5:][::-1]#polynomial, flipped for np.poly.
    #
    #     Dv = np.array(vz_grid/vsini).clip(-1,1)#Compute the Delta-v parameter, clipped to -1,1 to
    #     #avoid negative values of G.
    #     # H = gaussian(vz,1.0,vsys,s) / (s*np.sqrt(2*np.pi))
    #
    #     H = gaussian(RV,1.0,vsys,s) / (s*np.sqrt(2*np.pi))
    #     #G is defined on every position on the star because Dv is 2D.
    #     G0 = (2*(1-E)+0.5*np.pi*E)
    #     G = (2*(1-E)*np.sqrt((1-Dv**2))+0.5*np.pi*E*(1-Dv**2))# /(np.pi*vsini*(1-E/3)) <--- remove
    #     #this because it is a constant. This is 2D.
    #     diff = H@G / G0 * A + np.poly1d(c)(vz) - ccf #H@G/G0 is normalised from 0 to 1.
    #     return(diff)

    # if voigt:

    bounds = ([-np.inf,np.min(RV),0,0,0,0]+[i*0-np.inf for i in p_start],
    [0,np.max(RV),np.max(RV)-np.min(RV),1,np.inf,np.inf]+[i*0+np.inf for i in p_start])

    # if len(fixparams) > 0:


    result = scipy.optimize.least_squares(eval_rotation_broadened_line,startparams,
    args = (RV,CCF,vz_grid),bounds=bounds,method='trf')
    # else:
    #     result = scipy.optimize.least_squares(eval_rotation_broadened_line_gauss,startparams,
    #     args = (RV,CCF,vz_grid),bounds=([-np.inf,np.min(RV),0,0,0]+[i*0-np.inf for i in p_start],
    #     [0,np.max(RV),np.max(RV)-np.min(RV),1,np.inf]+
    #     [i*0+np.inf for i in p_start]),method='trf')


    return(result['x'])#These are the best-fit parameters for use in rotation_broadened_line().








def selmax(y_in,p,s=0.0):
    """This program returns the p (fraction btw 0 and 1) highest points in y,
    ignoring the very top fraction s (default zero, i.e. no points ignored), for the purpose of
    outlier rejection."""
    import tayph.util as ut
    from tayph.vartests import postest,dimtest
    import numpy as np
    import copy
    postest(p)
    y=copy.deepcopy(y_in)#Copy itself, or there are pointer-related troubles...
    if s < 0.0:
        raise Exception("ERROR in selmax: s should be zero or positive.")
    if p >= 1.0:
        raise Exception("ERROR in selmax: p should be strictly between 0.0 and 1.0.")
    if s >= 1.0:
        raise Exception("ERROR in selmax: s should be strictly less than 1.0.")
    postest(-1.0*p+1.0)
    #ut.nantest('y in selmax',y)
    dimtest(y,[0])#Test that it is one-dimensional.
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



def strided_window(a,L,pad=False):
    """This function computes the rolling window over a rectangular (i.e. long in X)
    array as a sequence of views into the original array, eliminating the need to index.
    This comes from https://stackoverflow.com/questions/44305987/sliding-windows-along-last-axis-of-a-2d-array-to-give-a-3d-array-using-numpy-str

    The output is what looks like a 3D array (though it doesn't take the memory of
    a 3D array because it's just a sequence of views stacked along the third axis)
    and you can do an operation in that direction without having to do the indexing
    needed to look up the values in that window from the large array.

    These windows are stacked along the 0th axis.

    If pad is set to False, the window only goes from the left edge to the right edge without going over.
    If set to True, the array is first padded with columns of NaNs such that the window effectively diminishes in size
    at the edges if e.g. nanmeans or nanmedians or equavalent nan-ignoring algorithms are applied later.

    """
    import copy
    import numpy as np
    if pad:
        ny,nx=a.shape
        a=np.hstack([np.full((ny,int(0.5*L)),np.nan),a,np.full((ny,int(0.5*L)+(L%2)-1),np.nan)])#This pads half a window worth of columns of NaNs to the edges of the array to make the window diminish at the edges.
    s0,s1 = a.strides
    m,n = a.shape
    return np.lib.stride_tricks.as_strided(a, shape=(m,n-L+1,L), strides=(s0,s1,s1)).transpose(1, 0, 2)

def running_median_2D(D,w):
    """This computes a running median on a 2D array in a window with width w that
    slides over the array in the horizontal (x) direction."""
    import numpy as np
    import numpy
    from tayph.vartests import typetest,dimtest,postest
    typetest(D,np.ndarray,'z in fun.running_mean_2D()')
    typetest(w,[int,float,numpy.int64],'w in fun.running_mean_2D()')
    postest(w,'w in fun.running_mean_2D()')
    ny,nx=D.shape
    m2=strided_window(D,w,pad=True)
    s=np.nanmedian(m2,axis=(1,2))
    return(s)

def running_std_2D(D,w):
    """This computes a running standard deviation on a 2D array in a window with width w that
    slides over the array in the horizontal (x) direction."""
    import numpy as np
    import numpy
    from tayph.vartests import typetest,dimtest,postest
    typetest(D,np.ndarray,'z in fun.running_mean_2D()')
    typetest(w,[int,float,numpy.int64],'w in fun.running_mean_2D()')
    postest(w,'w in fun.running_mean_2D()')
    ny,nx=D.shape
    m2=strided_window(D,w,pad=True)
    s=np.nanstd(m2,axis=(1,2))
    return(s)

def running_mean_2D(D,w):
    """This computes a running mean on a 2D array in a window with width w that
    slides over the array in the horizontal (x) direction."""
    import numpy as np
    import numpy
    from tayph.vartests import typetest,dimtest,postest
    typetest(D,np.ndarray,'z in fun.running_mean_2D()')
    typetest(w,[int,float,numpy.int64],'w in fun.running_mean_2D()')
    postest(w,'w in fun.running_mean_2D()')
    ny,nx=D.shape
    m2=strided_window(D,w,pad=True)
    s=np.nanmean(m2,axis=(1,2))
    return(s)




def running_MAD_2D(z,w,verbose=False,parallel=False):
    """Computers a running standard deviation of a 2-dimensional array z.
    The stddev is evaluated over the vertical block with width w pixels.
    The output is a 1D array with length equal to the width of z.
    This is very slow on arrays that are wide in x (hundreds of thousands of points).

    In the case that z is a very wide array (e.g. a stitched 1D spectrum with ~1e5 points, the
    speedup by setting parallel=True on my laptop with 8 cores is a factor of 2).
    """
    import astropy.stats as stats
    import numpy as np
    from tayph.vartests import typetest,dimtest,postest
    import tayph.util as ut
    import numpy


    typetest(z,np.ndarray,'z in fun.running_MAD_2D()')
    dimtest(z,[0,0],'z in fun.running_MAD_2D()')
    typetest(w,[int,float,numpy.int64],'w in fun.running_MAD_2D()')
    postest(w,'w in fun.running_MAD_2D()')
    size = np.shape(z)
    ny = size[0]
    nx = size[1]
    dx1=int(0.5*w)
    dx2=int(int(0.5*w)+(w%2))#To deal with odd windows.

    if parallel:
        from joblib import Parallel, delayed
        import multiprocessing
        ncores = multiprocessing.cpu_count()
    else:
        s = np.arange(0,nx,dtype=float)*0.0#If not parallel, we make this array for output.

    def compute_mad(i):#This is what goes into the forloop:
        minx = max([0,i-dx1])#This here is only a 3% slowdown.
        maxx = min([nx,i+dx2])
        return(stats.mad_std(z[:,minx:maxx],ignore_nan=True))#This is what takes 97% of the time.
        if verbose and not parallel: ut.statusbar(i,nx)

    if parallel:
        with Parallel(n_jobs=ncores) as P:
            s = np.array(P(delayed(compute_mad)(i) for i in range(nx)))
    else:
        s = np.array([compute_mad(i) for i in range(nx)])


    # for i in range(nx):
    #     minx = max([0,i-dx1])#This here is only a 3% slowdown.
    #     maxx = min([nx,i+dx2])
    #     s[i] = stats.mad_std(z[:,minx:maxx],ignore_nan=True)#This is what takes 97% of the time.
    #     if verbose: ut.statusbar(i,nx)
    return(s)

def running_MAD(z,w,parallel=False):
    """Computers a running standard deviation of a 1-dimensional array z.
    The stddev is evaluated over a range with width w pixels.
    The output is a 1D array with length equal to the width of z."""
    import astropy.stats as stats
    import numpy as np
    import numpy
    from tayph.vartests import typetest,dimtest,postest

    typetest(z,np.ndarray,'z in fun.running_MAD()')
    typetest(w,[int,float,numpy.int64],'w in fun.running_MAD()')
    postest(w,'w in fun.running_MAD()')
    nx = len(z)
    s = np.arange(0,nx,dtype=float)*0.0
    dx1=int(0.5*w)
    dx2=int(int(0.5*w)+(w%2))#To deal with odd windows.

    if parallel:
        from joblib import Parallel, delayed
        import multiprocessing
        ncores = multiprocessing.cpu_count()
    else:
        s = np.arange(0,nx,dtype=float)*0.0#If not parallel, we make this array for output.
    def compute_mad(i):
        minx = max([0,i-dx1])
        maxx = min([nx,i+dx2])
        return(stats.mad_std(z[minx:maxx],ignore_nan=True))

    if parallel:
        with Parallel(n_jobs=ncores) as P:
            s = np.array(P(delayed(compute_mad)(i) for i in range(nx)))
    else:
        s = np.array([compute_mad(i) for i in range(nx)])

    # for i in range(nx):
    #     minx = max([0,i-dx1])
    #     maxx = min([nx,i+dx2])
    #     s[i] = stats.mad_std(z[minx:maxx],ignore_nan=True)
    return(s)





def rebinreform(a,n):
    """
    This works like the rebin(reform()) trick in IDL, where you use fast
    array manipulation operations to transform a 1D array into a 2D stack of itself,
    to be able to do operations on another 2D array by multiplication/addition/division
    without having to loop through the second dimension of said array.

    This is likely depricated and may not even be used.
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
        >>> import numpy as np
        >>> y = np.array([0,0.1,0.2,np.nan,0.4,0.5])
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


def gaussian(x,*args):
    import numpy as np
    """
    This is an alternative to producing a gaussian function on the grid x with amplitude A, mean mu
    and standard deviation sig. Its primary usage is to be called by the gaussfit
    function below; but can be used generally in the same way as IDL's gaussian function.
    No testing is done to keep it fast.

    Parameters
    ----------
    x : np.ndarray
        The grid.

    *args : floats
        The Gaussian+continuum parameters. The first three values are the Gaussian amplitude,
        mean and standard deviation. If *args has a length longer than this, a polynomial
        will be added with degree len(*args)-4.


    Returns
    -------
    y : np.ndarray
        If p = *args, the output is a gaussian with parameters set by p[0:3], plus a polynomial continuum
        as p[3]  +  x * p[4]  +  x**2 * p[5]  + ...


    Example
    -------
    >>> import numpy as np
    >>> x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    >>> p=[5,10,1.0,0.0,0.5]
    >>> y=gaussian(x,p)
    """
    p=[i for i in args]
    if len(p) > 3:
        return p[0] * np.exp(-0.5*(x - p[1])/p[2]*(x - p[1])/p[2])+np.poly1d(p[3:][::-1])(x)
    else:
        return p[0] * np.exp(-0.5*(x - p[1])/p[2]*(x - p[1])/p[2])




def gaussfit(x,y,nparams=3,startparams=None,yerr=None,fixparams=None,minvalues=None,maxvalues=None,verbose=False):
    """
    This is a wrapper around lmfit to fit a Gaussian plus a polynomial continuum
    using the LM least-squares minimization algorithm. It is designed to be
    interacted with in a similar way as IDL's Gaussfit routine, though not
    exactly the same.

    The user provides data as np.arrays that form x and y pairs. The Gaussian model
    has a minimum number of 3 parameters: The amplitude, mean and standard deviation.
    In addition, the user can request additional parameters that describe the continuum
    as a polynomial with increasing degree. The model is therefore defined as
    follows:

        y = A*exp(-(x - mu)**2/(sqrt(2)sigma)**2) + c1 + c2*x + c3*x**2 + ...

    If initial guesses for the parameters are unknown or omitted, this function
    will guess start parameters albeit coarsely. The code will determine whether
    the peak is positive or negative by comparing the distance between the maximum and
    the median with the distance between the minimum and the median. If the former is larger,
    the Gaussian is considered to be positive.
    The amplitude is then guessed as the maximum - the minimum of the data y and the
    mean is guessed as the location of that maximum. Vice versa if the peak was
    established to be a minimum. The standard deviation is estimated as a fifth
    of the length of the data array.

    The nparams keyword will determine the number of free parameters to fit
    (3 for the Gaussian plus however many for the continuum).

    If startparams is set, the nparams keyword is overruled; and initial parameters
    need to be applied in the order [A,mu,sigma,c1,c2,...].

    By default, the least-squares optimizer assumes unweighted data points.
    If error bars on y are known, provide them with the yerr keyword (in
    the same way as you would supply them to plt.errorbar(x,y,yerr=...)).

    Parameter values can also be fixed (if startparams is set) or bounded.
    To fix parameters, set fixparams to a list (with same length as startparams)
    filled with 0's and 1's. Each parameter that is 1 will be fixed.

    For putting minimum and/or maximum bounds on the parameters, similarly
    provide lists with the same length as startparams, with the minimum/maximum
    of each parameter. To set a bound on only one parameter, the others that you
    wish to remain unbounded need to be set to np.inf or -np.inf.

    NaNs in y are omitted by the lmfit.minimizer routine. NaNs in x are not accepted.

    Lmfit is a powerful package for all of your curvefitting needs, and
    includes emcee functionality as well. Do check it out if you wish to fit
    models that are more complex than simple Gaussians, or wish to have
    access to more advanced statistics: https://lmfit.github.io.
    If you want to add more advanced functionality or build variations on this
    wrapper, a very useful page is their examples page:
    https://lmfit.github.io/lmfit-py/examples/index.html

    Parameters
    ----------
    x : np.ndarray
        The grid onto which the data is defined. NaNs are not allowed.

    y : np.ndarray
        The data. NaNs are allowed and will be omitted by the minimizer.

    nparams : int, optional.
        The number of parameters of the model. The model is defined as above, with
        a minimum of 3 parameters.

    startparmas : list, optional.
        The starting parameters at which the fit is initialized, in the order
        [A,mu,sigma,c1,c2,...]. This overrules the nparams keyword.

    yerr : np.ndarray, optional.
        The standard deviation on each of the values of y, by which the least-squares
        minimization will be weighted.

    fixparams : list, optional
        The parameters to fix, in the form of [0,1,0,1,1]; in the same order as
        startparams.

    minvalues : list, optional
        The minimum bounds of the fitting parameters, in the form of [-np.inf,10,5,-np.inf,-np.inf].

    maxvalues : list, optional
        The maximum bounds of the fitting parameters, in the form of [np.inf,10,5,np.inf,np.inf].

    verbose : bool, optional
        If set to True, the lmfit package will talk to you.

    Returns
    -------
    r : list
        The best-fit parameters, in the same order as startparams.
    re: list
        The 1-sigma confidence intervals on the best-fit parameters.



    Example
    -------
    >>> import numpy as np
    >>> x=findgen(100)
    >>> p=[10,50,6.0,0.0,0.05]
    >>> y=gaussian(x,*p)
    >>> f,e=gaussfit(x,y,startparams=[8.0,47.0,5.0,0.0,0.0],fixparams=[0,0,0,1,0.0])
    """


    import tayph.util as ut
    from tayph import typetest,minlength,dimtest,nantest
    from lmfit import Minimizer, Parameters, report_fit
    import sys
    import numpy as np
    import matplotlib.pyplot as plt

    typetest(x,np.ndarray,'x in fun.gaussfit()')
    typetest(y,np.ndarray,'y in fun.gaussfit()')
    typetest(nparams,int,'nparams in fun.gaussfit()')
    dimtest(x,[0],'x in fun.gaussfit()')
    dimtest(y,[len(x)],'y in fun.gaussfit()')
    nantest(x,'x in fun.gaussfit()')
    typetest(verbose,bool,'verbose in fun.gaussfit()')

    if startparams is None:
        if fixparams:#If startparams is not set, fixparams would fix the initial guesses which are likely not accurate, so there is no reason to this and I hereby protect the user against doing this to themselves.
            raise InputError("in fun.gaussfit(): startparams needs to be provided if you wish to fix any fitting parameters.")
        N = nparams
        if nparams <3:
            raise ValueError(f"in fun.gaussfit: nparams needs to be 3 or greater ({nparams}).")
        A = np.nanmax(y)-np.nanmin(y)
        if np.abs(np.nanmax(y)-np.median(y)) > np.abs(np.nanmin(y)-np.median(y)):#then we have a positive signal
            mu = x[np.nanargmax(y)]
        else:#or a negative signal.
            mu = x[np.nanargmin(y)]
            A*=(-1.0)
        sig= np.std(x)/5.0

        startparams = [A,mu,sig]
        paramnames = ['A','mu','sigma']
        if nparams > 3:
            for i in range(3,nparams):
                paramnames.append(f"c{i-2}")
                startparams.append(0.0)
    else:
        typetest(startparams,[list,np.ndarray],'startparams in fun.gaussfit()')
        minlength(startparams,3,varname='startparams in fun.gaussfit()',warning_only=False)
        nparams = len(startparams)#Startparams overrides the nparams keyword.
        paramnames = ['A','mu','sigma']
        if nparams > 3:
            for i in range(3,nparams):
                paramnames.append(f"c{i-2}")
    if isinstance(startparams,list):
        startparams=np.array(startparams)

    if yerr is not None:
        typetest(yerr,np.ndarray,'yerr in fun.gaussfit()')
        dimtest(yerr,[len(x)],'yerr in fun.gaussfit()')
        nantest(yerr,'yerr in fun.gaussfit()')
        weight = 1.0/yerr#Weigh by the variance as per https://en.wikipedia.org/wiki/Weighted_least_squares
        #Note that there is no square here, because the minimizer function squares.
    else:
        weight = x*0.0+1.0#Force weight to 1.0, i.e. unweighted least-sq.


    if fixparams is None:
        fixparams = startparams*0.0#Set all fix parameters to zero. I.e. none will be fixed.
    else:
        dimtest(fixparams,[len(startparams)],'fixparams in fun.gaussfit()')
        nantest(fixparams,'fixparams in fun.gaussfit()')

    if minvalues is None:
        minvalues = startparams*0.0-np.inf
        minvalues[2] = 0 #stddev needs to be positive.
    else:
        dimtest(fixparams,[len(startparams)],'minvalues in fun.gaussfit()')
        nantest(fixparams,'minvalues in fun.gaussfit()')

    if maxvalues is None:
        maxvalues = startparams*0.0+np.inf
    else:
        dimtest(fixparams,[len(startparams)],'maxvalues in fun.gaussfit()')
        nantest(fixparams,'maxvalues in fun.gaussfit()')



    def fcn2min(params, x, y, w):
        """
        Model the gaussian and subtract data. The square of the outcome of this
        will be minimized.
        """
        A = params['A']
        mu = params['mu']
        s = params['sigma']
        trigger = 0
        i=1
        p = [A,mu,s]
        while trigger == 0:
            try:
                p.append(params[f'c{i}'])
            except KeyError:
                trigger = 1
            i+=1
        model = gaussian(x,*p)
        return((model - y)*w)


    # pass the Parameters
    params = Parameters()
    for i in range(len(paramnames)):
        params.add(paramnames[i],value=startparams[i])
        params[paramnames[i]].min=minvalues[i]#These are +/- np.inf by default
        params[paramnames[i]].max=maxvalues[i]#So they will trivially be overwritten by infs if min/maxvalues are not set.
        if fixparams[i] == 1:
            params[paramnames[i]].vary = False

    minner = Minimizer(fcn2min, params, fcn_args=(x, y, weight),nan_policy='omit')
    result = minner.minimize()

    # write error report
    if verbose:
        report_fit(result)

    # calculate final result
    final = y + result.residual

    out_res = []
    out_err = []
    for i in range(len(paramnames)):
        out_res.append(result.params[paramnames[i]].value)
        out_err.append(result.params[paramnames[i]].stderr)
    return(out_res,out_err)




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


def local_v_star(phase,aRstar,inclination,vsini,l):
    """This is the rigid-body, circular-orbt approximation of the local velocity occulted
    by the planet as it goes through transit, as per Cegla et al. 2016. No tests are done
    to keep this fast."""
    import numpy as np
    xp = aRstar * np.sin(2.0*np.pi*phase)
    yp = (-1.0)*aRstar * np.cos(2.0*np.pi*phase) * np.cos(np.deg2rad(inclination))
    x_per = xp*np.cos(np.deg2rad(l)) - yp*np.sin(np.deg2rad(l))
    return(x_per*vsini)

def polysin(x,*p):
    """
    Evaluates a function of the form:
    y =  A*sin(2*pi*x/B+C) + an*x**n + ... + a1*x + a0
    """
    import numpy as np
    return(p[0]*np.sin(2*np.pi*x/p[1]+p[2]) + np.poly1d(p[3:])(x))




def vertical_polyfilter(C,deg):
    """This takes a block matrix and fits a polynomial of degree deg to the non-NaN values in each
    column. This can be used to clean up a CCF or an order of residual structures. NaNs can be used
    to mask out regions that need to be protected. If the degree is high, this operation can be
    quite agressive. In the future, I might want to enhance this by including different fitting
    functions, or making adjacent fits to be correlated with each other."""
    import copy
    import numpy as np

    C_out = copy.deepcopy(C)
    fits = copy.deepcopy(C)
    for i in range(len(C[0])):
        x = np.arange(len(C))
        y = C[:,i]
        sel = ~np.isnan(y)
        yf = np.poly1d(np.polyfit(x[sel],y[sel],deg))(x)
        C_out[:,i] = y-yf
        fits[:,i] = yf
    return(C_out,fits)








def polysinfit(x,y,polydeg,lmfit=True,polyprimer=True,return_primer=False,stepsize=0,w=None,startparams=None):
    """This fits a polynomial modulated by a sine-wave to a 1D array of points. With a solution
    tp the hyper-sensitivity to the intial guess frequency by unsym: https://stackoverflow.com/
    questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy


    polyprimer is ignored if startparams is set.
    """
    from scipy.optimize import curve_fit
    from scipy.interpolate import interp1d
    from lmfit import Minimizer, Parameters, report_fit
    import numpy as np
    import pdb
    import matplotlib.pyplot as plt

    x = np.array(x)
    y = np.array(y)

    if polyprimer and polydeg>0:
        primer = np.polyfit(x,y,polydeg,w=w)
        p = np.poly1d(primer)(x)#First-guess polynomial
        yy=y-p
    else:
        yy=y*1.0

    if stepsize == 0:
        dx = (np.max(x)-np.min(x))/len(x)
    else:
        dx = stepsize*1.0

    if w is None:
        w=x*0.0+1.0#Equal weights.
    elif len(x[w<=0]) > 0:
        # plt.plot(x,yy)
        try:
            yy[w<=0]=interp1d(x[w>0],y[w>0],bounds_error=False,fill_value='extrapolate')(x[w<=0])
        except:
            pdb.set_trace()
        # plt.plot(x,yy)
        # plt.show()


    if startparams is None:
        ff = np.fft.fftfreq(len(x),dx)   # assume uniform spacing
        Fy = abs(np.fft.fft(yy))
        guess_freq = abs(ff[np.argmax(Fy[1:])+1])   # excluding the zero frequency
        guess_amp = np.std(yy) * 2.**0.5
        guess_offset = np.mean(yy)

        if polyprimer and polydeg>0:
            startparams = [guess_amp,1/guess_freq,0] + list(primer)
        else:
            startparams = [guess_amp,1/guess_freq,0]+[0]*polydeg+[guess_offset]
    if not lmfit:
        pars,cov = curve_fit(polysin, x, y, p0=startparams)
        return(pars)
    else:

        # pass the Parameters
        params = Parameters()
        nparams = len(startparams)#Startparams overrides the nparams keyword.
        paramnames = ['A','B','C']
        if nparams > 3:
            for i in range(3,nparams):
                paramnames.append(f"c{i-2}")
        if isinstance(startparams,list):
            startparams=np.array(startparams)

        for i in range(len(paramnames)):
            params.add(paramnames[i],value=startparams[i])

        def fcn2min(params, x, y, w):
            """
            Model the polysin and subtract data. The square of the outcome of this
            will be minimized.
            """
            A = params['A']
            B = params['B']
            C = params['C']
            trigger = 0
            i=1
            p = [A,B,C]
            while trigger == 0:
                try:
                    p.append(params[f'c{i}'])
                except KeyError:
                    trigger = 1
                i+=1
            model = polysin(x,*p)
            return((model - y)*w)

        minner = Minimizer(fcn2min, params, fcn_args=(x, y, w),nan_policy='omit')
        result = minner.minimize()

        # write error report
        verbose=False
        if verbose:
            report_fit(result)

        # calculate final result
        final = y + result.residual

        out_res = []
        for i in range(len(paramnames)):
            out_res.append(result.params[paramnames[i]].value)

        if return_primer:
            return(out_res,startparams)
        else:
            return(out_res)




def sysrem(data,sigma,N,init=0,v=False,c_limit=1e-3,a_limit=1e-4,return_steps=False):
    """This runs Sysrem on a 2D frame of spectra; i.e. a time series of spectra or of a spectral order or of a cross-correlation function.
    These spectra need to be normalised before input; and need to be provided with an equivalent 2D array that describes the
    1-sigma error on each spectral point.

    Input:
        data:   A 2D array of spectral flux values. The x-dimension corresponds to the wavelength/velocity axis.
                The y-dimension corresponds to time, orbital phase or simply the exposure number.
                This array must be normalised such that all columns have mean = 1.0 and all rows also have mean = 1.0.
        sigma:  A 2D array equivalent to data, carrying the 1-sigma errors (with normalisation propagated).
        N:      int, the number of sysrem passes applied.
        init:   list, dimension corresponding to the height (number of spectra in the time-series) of data.
                This can be used to prime sysrem, e.g. assuming correlation with airmass. If set to 0, this is ignored
                and no time-correlation is assumed (default).
        v:      bool, verbose output to terminal.
        c_limit:float, the relative convergence limit on c.
        a_limit:float, the relative convergence limit on a.

    Output:
    Depending on whether return_steps is set, the routine will either output the corrected data, or a list of all the correction steps.
    If N is set to zero, the data itself is automatically returned.
        img:    A 2D array corresponding to the input array data, with sysrem applied N times.

        steps:  list. A list of copies of data with length equal to N+1, from each of which is subtracted a subsequent sysrem pass. The last element in this
                list contains the final output of sysrem."""
    import pdb
    import numpy as np
    import copy

    img = copy.deepcopy(data-np.nanmean(data))
    rms = copy.deepcopy(sigma)
    rms[rms<0]=np.nanmax(rms)
    rms[(np.isfinite(rms) == False)]=np.nanmax(rms)#Make sure there are no zero or NAN error values

    nexp,npx=np.shape(img)

    steps = [img]
    sq_rms = rms**2

    if N == 0:
        if v: print('N was set to zero. Returning data.')
        return(data)

    for t in range(N):
        a=findgen(nexp)*0.0+1.0
        c=findgen(npx)*0.0
        #If the weight keyword is set (i.e. to airmass),
        if type(init) == list:
            a=init*1.0
        k=1#Just a counter to track the number of steps until convergence.
        a0=a_limit*2#Greater than demanded, so we always start.
        c0=c_limit*2

        #Preparing vectorization of sysrem:
        invrms = 1.0/sq_rms
        imgrel = img/sq_rms

        c = (a@imgrel)/((a**2)@invrms)#Initialize c. This matrix multiplication is equivalent to:
        # for i in range(npx):
        #     c[i]=np.sum(imgrel[:,i]*a)/np.sum(a**2/sq_rms[:,i])


###########BEGIN SYSREM#########################
        if v: print('Running sysrem iteration %s'%t)
        while ((np.sum(np.abs(c0-c))/np.sum(np.abs(c0)) >= c_limit) or (np.sum(np.abs(a0-a))/np.sum(np.abs(a0)) >= a_limit)): #UNTIL CONVERGES
            c0=copy.deepcopy(c)
            a0=copy.deepcopy(a)
            #First I will code it as a double for-loop. Then I will try to vectorise it with numpy.

            a=(imgrel@c)/(invrms@(c**2))#CALCULATE NEW a FOR NEXT CORRELATING THING
            c = (a@imgrel)/((a**2)@invrms)

            #THIS IS EQUIVALENT TO THE FOLLOWING (uncomment to check by hand if you don't trust my word):
            # for j in range(nexp):
            #     a[j]=np.sum(imgrel[j,:]*c)/np.sum(c**2/sq_rms[j,:])
            # for i in range(npx):
            #     c[i]=np.sum(imgrel[:,i]*a)/np.sum(a**2/sq_rms[:,i])

            #I clocked the execution time to be a factor of 70 shorter this way.
            #BRING IT ON!
            #(thanks to Matteo for the advice of using the @ operator)

            #Can this be made faster?
            # cor =img*0.0
            # for i in range(nexp):
            #     cor[i]=c*a[i]
            #The following is much slower:
            #  cor=fun.rebinreform(c,nexp)*a[:,None]
            # And this is the answer: 30% faster than the forloop.
            # Nuts! The @ operator literally made this 113 times faster....!
            cor=np.outer(a,c)
            k=k+1
        if v: print("---Number of steps: %s,  c: %s (%s),  a: %s (%s)"%(k,np.sum(np.abs(c0-c))/np.sum(np.abs(c0)),c_limit,np.sum(np.abs(a0-a))/np.sum(np.abs(a0)),a_limit))
        img=img-cor
        steps.append(img)
    if return_steps:
        return(steps)
    else:
        return(img)
