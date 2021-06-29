__all__ = [
    "smooth",
    "envelope",
    "normalize_orders",
    "bin",
    "bin_avg",
    "convolve",
    "derivative",
    "constant_velocity_wl_grid",
    "blur_rotate",
    "airtovac",
    "vactoair",
    "clean_block",
    "fits_cleaner"
]



def smooth(fx,w,mode='box',edge_degree=1):
    """
    This function takes a spectrum, and blurs it using either a
    Gaussian kernel or a box kernel, which have a FWHM width of w px everywhere.
    Meaning that the width changes dynamically on a constant d-lambda grid.
    Set the mode to gaussian or box. Because in box, care is taken to correctly
    interpolate the edges, it is about twice slower than the Gaussian.
    This interpolation is done manually in the fun.box function.
    """

    import numpy as np
    import tayph.util as ut
    from tayph.vartests import typetest,postest
    import tayph.functions as fun
    from matplotlib import pyplot as plt
    typetest(w,[int,float],'w in ops.smooth()')
    typetest(fx,np.ndarray,'fx in ops.smooth()')
    typetest(mode,str,'mode in ops.smooth()')
    typetest(edge_degree,int,'edge_degree in ops.smooth()')
    postest(w,'w in ops.smooth()')

    if mode not in ['box','gaussian']:
        raise Exception(f'RuntimeError in ops.smooth(): Mode should be set to "top" or "bottom" ({mode}).')
    truncsize=4.0#The gaussian is truncated at 4 sigma.
    shape=np.shape(fx)

    sig_w = w / 2*np.sqrt(2.0*np.log(2)) #Transform FWHM to Gaussian sigma. In km/s.
    trunc_dist=np.round(sig_w*truncsize).astype(int)

    #First define the kernel.
    kw=int(np.round(truncsize*sig_w*2.0))
    if kw % 2.0 != 1.0:#This is to make sure that the kernel has an odd number of
    #elements, and that it is symmetric around zero.
        kw+=1

    kx=np.arange(kw,dtype=float) #fun.findgen(kw)
    kx-=np.mean(kx)#This must be centered around zero. Doing a hardcoded check:
    if (-1.0)*kx[-1] != kx[0]:
        print(kx)
        raise Exception("ERROR in box_smooth: Kernel could not be made symmetric somehow. Attempted kernel grid is printed above. Kernel width is %s pixels." % kw)


    if mode == 'gaussian':
        k=fun.gaussian(kx,1.0,0.0,sig_w)

    if mode == 'box':
        k=fun.box(kx,1.0,0.0,w)
        kx=kx[k > 0.0]
        k=k[k > 0.0]
        if (-1.0)*kx[-1] != kx[0]:
            print(kx)
            raise Exception("ERROR in box_smooth: Kernel could not be made symmetric AFTER CROPPING OUT THE BOX, somehow. Attempted kernel grid is printed above. Kernel width is %s pixels." % kw)

    k/=np.sum(k)

    return(convolve(fx,k,edge_degree))



def envelope(wlm,fxm,binsize,selfrac=0.05,mode='top',threshold=''):
    """
    This program measures the top or bottom envelope of a spectrum (wl,fx), by
    chopping it up into bins of size binsze (unit of wl), and measuring the mean
    of the top n % of values in that bin. Setting the mode to 'bottom' will do the
    oppiste: The mean of the bottom n% of values. The output is the resulting wl
    and flux points of these bins.

    Example: wle,fxe = envelope(wl,fx,1.0,selfrac=3.0,mode='top')
    """
    import numpy as np
    import tayph.util as ut
    import tayph.functions as fun
    from tayph.vartests import typetest,dimtest,nantest,postest
    from matplotlib import pyplot as plt
    typetest(wlm,np.ndarray,'wlm in ops.envelope()')
    typetest(fxm,np.ndarray,'fxm in ops.envelope()')
    dimtest(wlm,[len(fxm)])
    typetest(binsize,[int,float],'binsize in ops.envelope()')
    typetest(selfrac,float,'percentage in ops.envelope()')
    typetest(mode,str,'mode in envelope')
    nantest(fxm,'fxm in ops.envelope()')
    nantest(wlm,'wlm in ops.envelope()')
    postest(wlm,'wlm in ops.envelope()')
    postest(binsize,'binsize in ops.envelope()')

    if mode not in ['top','bottom']:
        raise Exception(f'RuntimeError in ops.envelope(): Mode should be set to "top" or "bottom" ({mode}).')

    binsize=float(binsize)
    if mode == 'bottom':
        fxm*=-1.0

    wlcs=np.array([])#Center wavelengths
    fxcs=np.array([])#Flux at center wavelengths
    i_start=0
    wlm_start=wlm[i_start]
    for i in range(0,len(wlm)-1):
        if wlm[i]-wlm_start >= binsize:
            wlsel = wlm[i_start:i]
            fxsel = fxm[i_start:i]
            maxsel = fun.selmax(fxsel,selfrac)
            wlcs=np.append(wlcs,np.mean(wlsel[maxsel]))
            fxcs=np.append(fxcs,np.mean(fxsel[maxsel]))
            i_start=i+1
            wlm_start=wlm[i+1]

    if isinstance(threshold,float) == True:
        #This means that the threshold value is set, and we set all bins less than
        #that threshold to the threshold value:
        if mode == 'bottom':
            threshold*=-1.0
        fxcs[(fxcs < threshold)] = threshold

    if mode == 'bottom':
        fxcs*=-1.0
        fxm*=-1.0
    return wlcs,fxcs

def normalize_orders(list_of_orders,list_of_sigmas,deg=0,nsigma=4,sinusoid=False):
    """
    If deg is set to 1, this function will normalise based on the mean flux in each order.
    If set higher, it will remove the average spectrum in each order and fit a polynomial
    to the residual. This means that in the presence of spectral lines, the fluxes will be
    slightly lower than if def=1 is used. nsigma is only used if deg > 1, and is used to
    throw away outliers from the polynomial fit. The program also computes the total
    mean flux of each exposure in the time series - totalled over all orders. These
    are important to correctly weigh the cross-correlation functions later. The
    inter-order colour correction is assumed to be an insignificant modification to
    these weights.

    Parameters
    ----------
    list_of_orders : list
        The list of 2D orders that need to be normalised.

    list_of_sigmas : list
        The list of 2D error matrices corresponding to the 2D orders that need to be normalised.

    deg : int
        The polynomial degree to remove. If set to 1, only the average flux is removed. If higher,
        polynomial fits are made to the residuals after removal of the average spectrum.

    nsigma : int, float
        The number of sigmas beyond which outliers are rejected from the polynomial fit.
        Only used when deg > 1.

    Returns
    -------
    out_list_of_orders : list
        The normalised 2D orders.
    out_list_of_sigmas : list
        The corresponding errors.
    meanfluxes : np.array
        The mean flux of each exposure in the time series, averaged over all orders.
    """
    import numpy as np
    import tayph.functions as fun
    from tayph.vartests import dimtest,postest,typetest
    import tayph.util as ut
    import warnings
    import pdb
    typetest(list_of_orders,list,'list_of_orders in ops.normalize_orders()')
    typetest(list_of_sigmas,list,'list_of_sigmas in ops.normalize_orders()')

    dimtest(list_of_orders[0],[0,0])#Test that the first order is 2D.
    dimtest(list_of_sigmas[0],[0,0])#And that the corresponding sigma array is, as well.
    n_exp=np.shape(list_of_orders[0])[0]#Get the number of exposures.
    for i in range(len(list_of_orders)):#Should be the same for all orders.
        dimtest(list_of_orders[i],[n_exp,0])
        dimtest(list_of_sigmas[i],np.shape(list_of_orders[i]))
    typetest(deg,int,'degree in ops.normalize_orders()')
    typetest(nsigma,[int,float],'nsigma in ops.normalize_orders()')
    # postest(deg,'degree in ops.normalize_orders()')
    postest(nsigma,'degree in ops.normalize_orders()')


    N = len(list_of_orders)
    out_list_of_orders=[]
    out_list_of_sigmas=[]


    #First compute the exposure-to-exposure flux variations to be used as weights.
    meanfluxes = np.zeros(n_exp) #fun.findgen(n_exp)*0.0

    ### suggestions for improvement
    """
    m = [np.nanmedian(list_of_orders[i], axis=1) for i in range(N)]
    skipped = np.where(np.sum(np.isnan(m)) > 0)[0]
    meanfluxes = np.sum(m[np.sum(np.isnan(m)) <= 0]) / len(m[np.sum(np.isnan(m)) <= 0])
    """

    N_i = 0
    for i in range(N):
        m = np.nanmedian(list_of_orders[i],axis=1)#Median or mean?
        if np.sum(np.isnan(m)) > 0:
            print('---Warning in normalise_orders: Skipping order %s because many nans are present.'%i)
        else:
            N_i+=1
            meanfluxes+=m#These contain the exposure-to-exposure variability of the time-series.
    meanfluxes/=N_i#These are the weights.

    if deg == 0:

        ### suggestion for improvement: (no loop needed)!
        """
        #meanflux = m # we already did that above!
        meanblock = m / np.nanmean(meanflux)
        out_list_of_orders.append((list_of_orders[i].T/meanblock).T)
        """
        for i in range(N):
            #What I'm doing here is probably stupid and numpy division will probably work just fine without
            #IDL-relics.
            n_px=np.shape(list_of_orders[i])[1]
            meanflux=np.nanmedian(list_of_orders[i],axis=1) #Average flux in each order. Median or mean?
            meanblock=fun.rebinreform(meanflux/np.nanmean(meanflux),n_px).T#This is a slow operation. Row-by-row division is better done using a double-transpose...
            out_list_of_orders.append(list_of_orders[i]/meanblock)
            out_list_of_sigmas.append(list_of_sigmas[i]/meanblock)
    else:
        for i in range(N):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                meanspec=np.nanmean(list_of_orders[i],axis=0)#Average spectrum in each order.
            x=np.array(range(len(meanspec)))
            poly_block = list_of_orders[i]*0.0#Array that will host the polynomial fits.
            colour = list_of_orders[i]/meanspec#What if there are zeroes? I.e. padding around the edges of the order?
            for j,s in enumerate(list_of_orders[i]):
                idx = np.isfinite(colour[j])
                if np.sum(idx) > 0:
                    p = np.poly1d(np.polyfit(x[idx],colour[j][idx],deg))(x)#Polynomial fit to the colour variation.
                    res = colour[j]/p-1.0 #The residual, which is flat around zero if it's a good fit. This has all sorts of line residuals that we need to throw out.
                    #We do that using the weight keyword of polyfit, and just set all those weights to zero.
                    sigma = np.nanstd(res)
                    w = x*0.0+1.0#Start with a weight function that is 1.0 everywhere.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        w[np.abs(res)>nsigma*sigma] = 0.0
                    if sinusoid == False:
                        p2 = np.poly1d(np.polyfit(x[idx],colour[j][idx],deg,w=w[idx]))(x)#Second, weighted polynomial fit to the colour variation.
                    else:
                        p2 = fun.polysin(x,*fun.polysinfit(x[idx],colour[j][idx],deg,stepsize=1,polyprimer=True,lmfit=True,w=w[idx]))
                    poly_block[j] = p2

            out_list_of_orders.append(list_of_orders[i]/poly_block)
            out_list_of_sigmas.append(list_of_sigmas[i]/poly_block)
            ut.statusbar(i,N)
    return(out_list_of_orders,out_list_of_sigmas,meanfluxes)




def bin(x,y,n,err=[]):
    """
    A simple function to quickly bin a spectrum by a certain number of points.

    Parameters
    ----------
    x : list, np.ndarray
        The horizontal axis.

    y : list, np.ndarray
        The array corresponding to x.

    n : int, float
        The number of points by which to bin. Int is recommended, and it is converted
        to int if a float is provided, with the consequence of the rounding that int() does.

    err : list, np.array, optional
        If set, errors corresponding to the flux points.

    Returns
    -------
    x_bin : np.array
        The binned x-axis.
    y_bin : np.array
        The binned y-axis.
    e_bin : np.array
        Only if err is set to an array with non-zero length.
    """

    from tayph.vartests import typetest,dimtest,minlength
    import numpy as np

    typetest(x,[list,np.ndarray],'x in ops.bin()')
    typetest(y,[list,np.ndarray],'y in ops.bin()')
    typetest(n,[int,float],'n in ops.bin()')
    dimtest(x,[0],'x in ops.bin()')
    dimtest(y,[len(x)],'y in ops.bin()')
    minlength(x,0,'x in ops.bin()')
    minlength(x,100,'x in ops.bin()',warning_only=True)


    x_bin=[]
    y_bin=[]
    et=False
    if len(err) > 0:
        e_bin = []
        et = True
    for i in range(0,int(len(x)-n-1),n):
        x_bin.append(np.nanmean(x[i:i+n]))
        y_bin.append(np.nanmean(y[i:i+n]))
        if et:
            e_bin.append(np.sqrt(np.nansum(err[i:i+n]**2))/len(err[i:i+n]))
    if et:
        return(np.array(x_bin),np.array(y_bin),np.array(e_bin))
    else:
        return(np.array(x_bin),np.array(y_bin))


def bin_avg(wl,fx,wlm):
    """
    A simple function to bin a high-res model spectrum (wl,fx) to a lower resolution wavelength grid (wlm),
    by averaging the points fx inside the lower-resolution wavelength bins set by wlm.

    Parameters
    ----------
    wl : list, np.ndarray
        The horizontal axis.

    fx : list, np.ndarray
        The array corresponding to x.

    wlm : list, np.ndarray
        The lower resolution wlm array. wlm should have fewer points than wl and fx.

    Returns
    -------
    y_bin : np.array
        The binned flux points.
    """
    import numpy as np
    import sys
    from tayph.vartests import typetest,dimtest,minlength
    import matplotlib.pyplot as plt
    import tayph.util as ut
    typetest(wl,[list,np.ndarray],'wl in ops.bin_avg()')
    typetest(fx,[list,np.ndarray],'fx in ops.bin_avg()')
    typetest(wlm,[list,np.ndarray],'wlm in ops.bin_avg()')
    dimtest(wl,[0],'wl in ops.bin_avg()')
    dimtest(fx,[len(fx)],'fx in ops.bin_avg()')
    minlength(wl,0,'wl in ops.bin_avg()')
    minlength(wl,50,'wl in ops.bin_avg()',warning_only=True)
    minlength(wlm,0,'wlm in ops.bin_avg()')
    minlength(wlm,100,'wlm in ops.bin_avg()',warning_only=True)

    if len(wl) < len(wlm):
        raise ValueError(f"Error in bin_avg: The input grid should have (many) more points than the requested grid of interpolates ({len(wl)}, {len(wlm)}).")
    if max(wl) < min(wlm) or min(wl) > max(wlm):
        raise ValueError(f"Error in bin_avg: the supplied wl and wlm arrays have no overlap. Are the wavelength units the same? (mean(wl)={np.mean(wlm)}, mean(wlm)={np.mean(wlm)}).")

    dwl_start = wlm[1]-wlm[0]
    wlm_borders=[wlm[0]-dwl_start/2.0]
    for i in range(len(wlm)-1):
        dwl=wlm[i+1]-wlm[i]
        if i == 0:
            wlm_borders=[wlm[0]-dwl/2.0]
        wlm_borders.append(wlm[i]+dwl/2.0)
    wlm_borders.append(wlm[-1]+dwl/2.0)


    fxm = []

    for i in range(len(wlm_borders)-1):
        #This indexing is slow. But I always only need to compute it once!
        fx_sel = fx[(wl > wlm_borders[i])&(wl <= wlm_borders[i+1])]
        if len(fx_sel) == 0:
            raise RuntimeError(f"Error in bin_avg: No points were selected in step {i}. Wlm_borders has a smaller step size than fx?")
        else:
            fxbin=np.mean(fx[(wl > wlm_borders[i])&(wl <= wlm_borders[i+1])])
            fxm.append(fxbin)
    return(fxm)




def convolve(array,kernel,edge_degree=1,fit_width=2):
    """It's unbelievable, but I could not find the python equivalent of IDL's
    /edge_truncate keyword, which truncates the kernel at the edge of the convolution.
    Therefore, unfortunately, I need to code a convolution operation myself.
    Stand by to be slowed down by an order of magnitude #thankspython.

    Nope! Because I can just use np.convolve for most of the array, just not the edge...

    So the strategy is to extrapolate the edge of the array using a polynomial fit
    to the edge elements. By default, I fit over a range that is twice the length of the kernel; but
    this value can be modified using the fit_width parameter.

    Parameters
    ----------
    array : list, np.ndarray
        The horizontal axis.

    kernel : list, np.ndarray
        The convolution kernel. It is required to have a length that is less than 25% of the size of the array.

    edge_degree : int
        The polynomial degree by which the array is extrapolated in order to

    fit_width : int
        The length of the area at the edges of array used to fit the polynomial, in units of the length of the kernel.
        Increase this number for small kernels or noisy arrays.
    Returns
    -------
    array_convolved : np.array
        The input array convolved with the kernel

    Example
    -------
    >>> import numpy as np
    >>> a=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    >>> b=[-0.5,0,0.5]
    >>> c=convolve(a,b,edge_degree=1)
    """

    import numpy as np
    import tayph.functions as fun
    from tayph.vartests import typetest,postest,dimtest
    typetest(edge_degree,int,'edge_degree in ops.convolve()')
    typetest(fit_width,int,'edge_degree in ops.convolve()')
    typetest(array,[list,np.ndarray])
    typetest(kernel,[list,np.ndarray])
    dimtest(array,[0],'array in ops.convolve()')
    dimtest(kernel,[0],'array in ops.convolve()')
    postest(edge_degree,'edge_degree in ops.convolve()')
    postest(fit_width,'edge_degree in ops.convolve()')

    array = np.array(array)
    kernel= np.array(kernel)

    if len(kernel) >= len(array)/4.0:
        raise Exception(f"Error in ops.convolve(): Kernel length is larger than a quarter of the array ({len(kernel)}, {len(array)}). Can't extrapolate over that length. And you probably don't want to be doing a convolution like that, anyway.")

    if len(kernel) % 2 != 1:
        raise Exception('Error in ops.convolve(): Kernel needs to have an odd number of elements.')

    #Perform polynomial fits at the edges.
    x= np.arange(len(array), dtype=float) # fun.findgen(len(array))
    fit_left=np.polyfit(x[0:len(kernel)*2],array[0:len(kernel)*2],edge_degree)
    fit_right=np.polyfit(x[-2*len(kernel)-1:-1],array[-2*len(kernel)-1:-1],edge_degree)

    #Pad both the x-grid (onto which the polynomial is defined)
    #and the data array.
    pad= np.arange((len(kernel)-1)/2, dtype=float) #fun.findgen((len(kernel)-1)/2)
    left_pad=pad-(len(kernel)-1)/2
    right_pad=np.max(x)+pad+1
    left_array_pad=np.polyval(fit_left,left_pad)
    right_array_pad=np.polyval(fit_right,right_pad)

    #Perform the padding.
    x_padded = np.append(left_pad , x)
    x_padded = np.append(x_padded , right_pad) #Pad the array with the missing elements of the kernel at the edge.
    array_padded = np.append(left_array_pad,array)
    array_padded = np.append(array_padded,right_array_pad)

    #Reverse the kernel because np.convol does that automatically and I don't want that.
    #(Imagine doing a derivative with a kernel [-1,0,1] and it gets reversed...)
    kr = kernel[::-1]
    #The valid keyword effectively undoes the padding, leaving only those values for which the kernel was entirely in the padded array.
    #This thus again has length equal to len(array).
    return np.convolve(array_padded,kr,'valid')




def derivative(x):
    """
    This computes the simple numerical derivative of x by convolving with kernel [-1,0,1].

    Parameters
    ----------
    x : list, np.ndarray
        The array from which the derivative is required.

    Returns
    -------
    derivative : np.array
        The numerical derivative of x.

    Example
    -------
    >>> import numpy as np
    >>> x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    >>> dx=derivative(x)
    """
    import numpy as np
    from tayph.vartests import typetest,dimtest
    typetest(x,[list,np.ndarray],'x in derivative()')
    dimtest(x,[0],'x in derivative()')
    x=np.array(x)
    d_kernel=np.array([-1,0,1])/2.0
    return(convolve(x,d_kernel,fit_width=3))



def constant_velocity_wl_grid(wl,fx,oversampling=1.0):
    """This function will define a constant-velocity grid that is (optionally)
    sampled a number of times finer than the SMALLEST velocity difference that is
    currently in the grid.

    Example: wl_cv,fx_cv = constant_velocity_wl_grid(wl,fx,oversampling=1.5).

    This function is hardcoded to raise an exception if wl or fx contain NaNs,
    because interp1d does not handle NaNs.


    Parameters
    ----------
    wl : list, np.ndarray
        The wavelength array to be resampled.

    fx : list, np.ndarray
        The flux array to be resampled.

    oversampling : float
        The factor by which the wavelength array is *minimally* oversampled.


    Returns
    -------
    wl : np.array
        The new wavelength grid.

    fx : np.array
        The interpolated flux values.

    a : float
        The velocity step in km/s.


    """
    import astropy.constants as consts
    import numpy as np
    import tayph.functions as fun
    from tayph.vartests import typetest,nantest,dimtest,postest
    from scipy import interpolate
    import pdb
    import matplotlib.pyplot as plt
    typetest(oversampling,[int,float],'oversampling in constant_velocity_wl_grid()',)
    typetest(wl,[list,np.ndarray],'wl in constant_velocity_wl_grid()')
    typetest(fx,[list,np.ndarray],'fx in constant_velocity_wl_grid()')
    nantest(wl,'wl in in constant_velocity_wl_grid()')
    nantest(fx,'fx in constant_velocity_wl_grid()')
    dimtest(wl,[0],'wl in constant_velocity_wl_grid()')
    dimtest(fx,[len(wl)],'fx in constant_velocity_wl_grid()')
    postest(oversampling,'oversampling in constant_velocity_wl_grid()')

    oversampling=float(oversampling)
    wl=np.array(wl)
    fx=np.array(fx)

    c=consts.c.to('km/s').value

    dl=derivative(wl)
    dv=dl/wl*c
    a=np.min(dv)/oversampling

    wl_new=0.0
    #The following while loop will define the new pixel grid.
    #It starts trying 100,000 points, and if that's not enough to cover the entire
    #range from min(wl) to max(wl), it will add 100,000 more; until it's enough.
    n=len(wl)
    while np.max(wl_new) < np.max(wl):
        x=np.arange(n, dtype=float) #fun.findgen(n)
        wl_new=np.exp(a/c * x)*np.min(wl)
        n+=len(wl)
    wl_new[0]=np.min(wl)#Artificially set to zero to avoid making a small round
    #off error in that exponent.

    #Then at the end we crop the part that goes too far:
    wl_new_cropped=wl_new[(wl_new <= np.max(wl))]
    x_cropped=x[(wl_new <= np.max(wl))]
    i_fx = interpolate.interp1d(wl,fx)
    fx_new_cropped =i_fx(wl_new_cropped)
    return(wl_new_cropped,fx_new_cropped,a)



def blur_rotate(wl,order,dv,Rp,P,inclination,status=False,fast=False):
    """This function takes a spectrum and blurs it using a rotation x Gaussian
    kernel which has a FWHM width of dv km/s everywhere. Meaning that its width changes
    dynamically.
    Because the kernel needs to be recomputed on each element of the wavelength axis
    individually, this operation is much slower than convolution with a constant kernel,
    in which a simple shifting of the array, rather than a recomputation of the rotation
    profile is sufficient. By setting the fast keyword, the input array will first
    be oversampled onto a constant-velocity grid to enable the usage of a constant kernel,
    after which the result is interpolated back to the original grid.

    Input:
    The wavelength axis wl.
    The spectral axis order.
    The FHWM width of the resolution element in km/s.
    The Radius of the rigid body in Rj.
    The periodicity of the rigid body rotation in days.
    The inclination of the spin axis in degrees.

    Wavelength and order need to be numpy arrays and have the same number of elements.
    Rp, P and i need to be scalar floats.

    Output:
    The blurred spectral axis, with the same dimensions as wl and order.


    WARNING: THIS FUNCTION HANDLES NANS POORLY. I HAVE THEREFORE DECIDED CURRENTLY
    TO REQUIRE NON-NAN INPUT.




    This computes the simple numerical derivative of x by convolving with kernel [-1,0,1].

    Parameters
    ----------
    wl : list, np.ndarray
        The wavelength array.

    order : list, np.ndarray.
        The spectral axis.

    dv: float
        The FWHM of a resolution element in km/s.

    Rp: float
        The radius of the planet in jupiter radii.

    P: float
        The rotation period of the planet. For tidally locked planets, this is equal
        to the orbital period.

    inclination:
        The inclination of the spin axis in degrees. Presumed to be close to 90 degrees
        for transiting planets

    status: bool
        Output a statusbar, but only if fast == False.

    fast: bool
        Re-interpolate the input on a constant-v grid in order to speed up the computation
        of the convolution by eliminating the need to re-interpolate the kernel every step.



    Returns
    -------
    order_blurred : np.array
        The rotation-broadened spectrum on the same wavelength grid as the input.

    Example
    -------
    >>> import tayph.functions as fun
    >>> wl = fun.findgen(4000)*0.001+500.0
    >>> fx = wl*0.0
    >>> fx[2000] = 1.0
    >>> fx_blurred1 = blur_rotate(wl,fx,3.0,1.5,0.8,90.0,status=False,fast=False)
    >>> fx_blurred2 = blur_rotate(wl,fx,3.0,1.5,0.8,90.0,status=False,fast=True)
    """

    import numpy as np
    import tayph.util as ut
    import tayph.functions as fun
    from tayph.vartests import typetest,nantest,dimtest
    from matplotlib import pyplot as plt
    import astropy.constants as const
    import astropy.units as u
    import time
    import sys
    import pdb
    from scipy import interpolate
    typetest(dv,float,'dv in blur_rotate()')
    typetest(wl,[list,np.ndarray],'wl in blur_rotate()')
    typetest(order,[list,np.ndarray],'order in blur_rotate()')
    typetest(P,float,'P in blur_rotate()')
    typetest(Rp,float,'Rp in blur_rotate()')
    typetest(inclination,float,'inclination in blur_rotate()')
    typetest(status,bool,'status in blur_rotate()')
    typetest(fast,bool,'fast in blur_rotate()')
    nantest(wl,'dv in blur_rotate()')
    nantest(order,'order in blur_rotate()')
    dimtest(wl,[0],'wl in blur_rotate()')
    dimtest(order,[len(wl)],'order in blur_rotate()')#Test that wl and order are 1D, and that
    #they have the same length.

    if np.min(np.array([dv,P,Rp])) <= 0.0:
        raise Exception("ERROR in blur_rotate: dv, P and Rp should be strictly positive.")

    #ut.typetest_array('wl',wl,np.float64)
    #ut.typetest_array('order',order,np.float64)
    #This is not possible because order may be 2D...
    #And besides, you can have floats, np.float32 and np.float64... All of these would
    #need to pass. Need to fix typetest_array some day.


    order_blurred=order*0.0#init the output.
    truncsize=5.0#The gaussian is truncated at 5 sigma from the extremest points of the RV amplitude.
    sig_dv = dv / (2*np.sqrt(2.0*np.log(2))) #Transform FWHM to Gaussian sigma. In km/s.
    deriv = derivative(wl)
    if max(deriv) < 0:
        raise Exception("ERROR in ops.blur_rotate: WL derivative is smaller than 1.0. Sort wl in ascending order.")
    sig_wl=wl*sig_dv/(const.c.to('km/s').value)#in nm
    sig_px=sig_wl/deriv

    n=1000.0
    a=np.arange(n, dtype=float)/(n-1)*np.pi #fun.findgen(n)/(n-1)*np.pi
    rv=np.cos(a)*np.sin(np.radians(inclination))*(2.0*np.pi*Rp*const.R_jup/(P*u.day)).to('km/s').value #in km/s
    trunc_dist=np.round(sig_px*truncsize+np.max(rv)*wl/(const.c.to('km/s').value)/deriv).astype(int)
    # print('Maximum rotational rv: %s' % max(rv))
    # print('Sigma_px: %s' % np.nanmean(np.array(sig_px)))


    rvgrid_max=(np.max(trunc_dist)+1.0)*sig_dv+np.max(rv)
    rvgrid_n=rvgrid_max / dv * 100.0 #100 samples per lsf fwhm.
    rvgrid=np.arange(-rvgrid_n, rvgrid_n, dtype=float)/rvgrid_n*rvgrid_max #(fun.findgen(2*rvgrid_n+1)-rvgrid_n)/rvgrid_n*rvgrid_max#Need to make sure that this is wider than the truncation bin and more finely sampled than wl - everywhere.
    lsf=rvgrid*0.0
    #We loop through velocities in the velocity grid to build up the sum of Gaussians
    #that is the LSF.
    for v in rv:
        lsf+=fun.gaussian(rvgrid,1.0,v,sig_dv)#This defines the LSF on a velocity grid wih high fidelity.
    if fast:
        wlt,fxt,dv = constant_velocity_wl_grid(wl,order,4)
        dv_grid = rvgrid[1]-rvgrid[0]

        len_rv_grid_low = int(max(rvgrid)/dv*2-2)
        # print(len_rv_grid_low)
        # print(len(fun.findgen(len_rv_grid_low)))
        # print(len_rv_grid_low%2)
        if len_rv_grid_low%2 == 0:
            len_rv_grid_low -= 1
        rvgrid_low = np.arange(len_rv_grid_low)*dv #fun.findgen(len_rv_grid_low)*dv#Slightly smaller than the original grid.
        rvgrid_low -=0.5*np.max(rvgrid_low)
        lsf_low = interpolate.interp1d(rvgrid,lsf)(rvgrid_low)
        lsf_low /=np.sum(lsf_low)#This is now an LSF on a grid with the same spacing as the data has.
        #This means I can use it directly as a convolution kernel:
        fxt_blurred = convolve(fxt,lsf_low,edge_degree=1,fit_width=1)
        #And interpolate back to where it came from:
        order_blurred = interpolate.interp1d(wlt,fxt_blurred,bounds_error=False)(wl)
        #I can use interp1d because after blurring, we are now oversampled.
        # order_blurred2 = bin_avg(wlt,fxt_blurred,wl)
        return(order_blurred)



    #Now we loop through the wavelength grid to place this LSF at each wavelength position.
    for i in range(0,len(wl)):
        binstart=max([0,i-trunc_dist[i]])
        binend=i+trunc_dist[i]
        wlbin=wl[binstart:binend]

        wlgrid =   wl[i]*rvgrid/(const.c.to('km/s').value)+wl[i]#This converts the velocity grid to a d-wavelength grid centered on wk[i]
        #print([np.min(wlbin),np.min(wlgrid),np.max(wlbin),np.max(wlgrid)])

        i_wl = interpolate.interp1d(wlgrid,lsf,bounds_error=False,fill_value='extrapolate')#Extrapolate should not be necessary but sometimes there is a minute mismatch between the
        #start and end wavelengths of the constructed grid and the bin.
        try:
            lsf_wl=i_wl(wlbin)
        except:
            ut.tprint('Error in interpolating LSF onto wlbin. Pausing to debug.')
            pdb.set_trace()
        k_n=lsf_wl/np.sum(lsf_wl)#Normalize at each instance of the interpolation to make sure flux is conserved exactly.
        order_blurred[i]=np.sum(k_n*order[binstart:binend])
        if status == True:
            ut.statusbar(i,len(wl))
    return(order_blurred)



def airtovac(wlnm):
    """
    This converts air wavelengths to vaccuum wavelengths.

    Parameters
    ----------
    wlnm : float, np.ndarray
        The wavelength that is to be converted.

    Returns
    -------
    wlnm : float, np.array
        wavelengths in vaccuum.

    Example
    -------
    >>> import numpy as np
    >>> wlA=np.array([500.0,510.0,600.0,700.0])
    >>> wlV=airtovac(wlA)
    """
    import numpy as np
    from tayph.vartests import typetest
    typetest(wlnm,[float,np.ndarray],'wlmn in airtovac()')
    wlA=wlnm*10.0
    s = 1e4 / wlA
    n = 1 + (0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) +
    0.0001599740894897 / (38.92568793293 - s**2))
    return(wlA*n/10.0)

def vactoair(wlnm):
    """
    This converts vaccuum wavelengths to air wavelengths.

    Parameters
    ----------
    wlnm : float, np.ndarray
        The wavelength that is to be converted.

    Returns
    -------
    wlnm : float, np.array
        wavelengths in air.

    Example
    -------
    >>> import numpy as np
    >>> wlV=np.array([500.0,510.0,600.0,700.0])
    >>> wlA=vactoair(wlV)
    """
    import numpy as np
    from tayph.vartests import typetest
    typetest(wlnm,[float,np.ndarray],'wlmn in vactoair()')
    wlA = wlnm*10.0
    s = 1e4/wlA
    f = 1.0 + 5.792105e-2/(238.0185e0 - s**2) + 1.67917e-3/( 57.362e0 - s**2)
    return(wlA/f/10.0)



def clean_block(wl,block,deg=0,w=200,nsigma=5.0,verbose=False,renorm=True,parallel=False):
    """This quickly cleans a spectral block by performing trimming of zeroes at the edges,
    setting zeroes and negative values to NaN, normalising the flux along both axes, rejecting
    outlier values by using a running MAD, and optionally detrending using polynomials. This is
    intended for quick-look cross-correlation when reading in the data.

    The output is the trimmed wavelength axis, the outlier-rejected sequence of spectra (block),
    and the residuals after outlier rejection. If detrend set to True, polynomials will be fit, and
    the residuals after polyfitting are also returned. So setting detrend=True increases the number
    of output variables from 3 to 4.

    I've ran mad trying to make this faster but medians simply don't collapse into nice matrix
    operations. So what remains is to parallelise. I've set up the keyword structure to make this
    happen, as per Issue #64.

    Set the renorm keyword to maintain the average flux in the block. If set to false, this
    function will return spectra that have the same average as before cleaning.
    """
    import numpy as np
    import tayph.functions as fun
    import warnings
    import copy
    import tayph.util as ut
    if parallel: from joblib import Parallel, delayed
    block[np.abs(block)<1e-10*np.nanmedian(block)]=0.0#First set very small values to zero.
    sum=np.nansum(np.abs(block),axis=0)#Anything that is zero in this sum (i.e. the all-zero
    #columns) will be identified.
    npx=len(sum)
    leftsize=npx-len(np.trim_zeros(sum,'f'))#Size of zero-padding on the left.
    rightsize=npx-len(np.trim_zeros(sum,'b'))#Size of zero-padding on the right

    wl_trimmed=wl[leftsize:npx-rightsize-1]
    block = block[:,leftsize:npx-rightsize-1]
    block[block<=0]=np.nan            #We get rid of remaining zeroes and negative values:

    #Compute residuals of s1d spectra by double-normalisation:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        avg_flux=np.nanmean(block,axis=1)
        block = np.transpose(np.transpose(block)/avg_flux)
        block_avg=np.nanmean(block,axis=0)
        r=block/block_avg

        if renorm==False:
            avg_flux=avg_flux*0.0+1.0

        #Outlier rejection in the 2D residuals via a running MAD:
        MAD=fun.running_MAD_2D(r,w,verbose=verbose,parallel=parallel)#This can take long.
        warnings.simplefilter("ignore")
        r[np.abs(r-np.nanmean(r))/MAD>nsigma]=np.nan
        block[np.abs(r-np.nanmean(r))/MAD>nsigma]=np.nan
    if deg>0:
        r2=copy.deepcopy(r)
        for j,s in enumerate(r):
            nansel=np.isnan(s)
            p=np.polyfit(wl_trimmed[~nansel],s[~nansel],deg)
            block[j]/=np.polyval(p,wl_trimmed)
            r2[j]/=np.polyval(p,wl_trimmed)
            ut.statusbar(j,len(r))
        return(wl_trimmed,np.transpose(np.transpose(block)*avg_flux),r,r2)#Avg_flux is 1.0 unless
        #renorm=True.
    return(wl_trimmed,np.transpose(np.transpose(block)*avg_flux),r)

def fits_cleaner(uncleandirectory, cleandirectory, night, mode, cut_off = 0.1):
    """Runs through the exposures of a night and removes any bad ones which exist.
        args:
            uncleandirectory: the directory where the raw fits files are kept
            cleandirectory: the directroy where the clean files will go
            night: the name of the night you use (try to keep it consistent with your CC
            mode: visable or infrared modes (CARMENES-VIS, CARMENES-NIR)
            cut_off: the decimal cut off value for the exposures
        returns:
            No variables are returned but the well behaved files are copied to a new directroy
    """
    from os import listdir, mkdir, path
    import shutil
    import numpy as np
    from astropy.io import fits
    from astropy.table import Table, Column
    from astropy.time import Time
    from tayph.read import spec_stich_n_norm #import the stich algorithm from tayph

    #Each "mode" needs to be cleaned seperately but will go to the same directory
    if mode == "CARMENES-VIS":
        mode_ext = "vis_A"
    elif mode == "CARMENES-NIR":
        mode_ext = "nir_A"

    file_list = listdir(uncleandirectory + night) #obtains the file list from the night in question
    non_cached_file_list = []
    times = []
    #print(file_list)
    for e in file_list:
        if e[0] == "c":
            t = e[4:21]
            non_cached_file_list.append(e)
            t = t.replace("h",":")
            t = t.replace("m",":")
            t = t[:4] + "-" + t[4:6] + "-" + t[6:]
            times.append(Time(t, scale = "utc"))
    non_cached_file_list = Column(non_cached_file_list, "files")
    times = Column(times, "times")
    t = Table()
    t.add_column(times)
    t.add_column(non_cached_file_list)
    t.sort("times")

    file_list = list(t["files"])

    vis_files = []
    sum_vals = []
    print("Scanning through files and opening the relevent ones.")
    for i in range(0,len(file_list)):
        # Prints the current file being read out of the total made in the night
        print("File #" + str(i) + " of " + str(len(file_list)), end="\r")

        if file_list[i][-10:-5] == mode_ext: #Takes the explosures from the mode and night in question
            print(file_list[i])
            vis_files.append(file_list[i])
            hdul = fits.open(uncleandirectory + night + file_list[i])
            spec = hdul[1].data
            cont = hdul[2].data
            sig  = hdul[3].data
            wave = hdul[4].data
            wav_len, spec_1d = spec_stich_n_norm(spec, wave, cont, sig) #Stiches all the orders together to make 1d spec

            #All flux values get summed here, bad exposrues have lower values big several orders of magnitude
            if mode_ext == "vis_A":
                sum_val = np.nansum(spec_1d[wav_len<8000])

            if mode_ext == "nir_A":
                sum_val = np.nansum(spec_1d)

            sum_vals.append(sum_val)

    sum_vals = sum_vals / np.median(sum_vals)#Normalise the summed values
    mask = sum_vals > cut_off #Creates a mask based on a specified cutoff

    if path.isdir(cleandirectory + night) == False:
        #This checks if the directory for the night in the cleaned spectra has been made
        mkdir(cleandirectory + night)

    print(mask)

    #All good exposures are moved to this new directory
    for i in range(len(vis_files)):
        print(i)
        if mask[i] == True:
            shutil.copy(uncleandirectory + night + vis_files[i], cleandirectory + night + vis_files[i])
        elif mask[i] == False:
            print("Stopping Code")
            quit()
            #shutil.copy(uncleandirectory + night + vis_files[i], cleandirectory + night + vis_files[i])