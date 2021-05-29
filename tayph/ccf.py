__all__ = [
    "xcor",
    "mask_cor3D"
    "clean_ccf",
    "filter_ccf",
    "shift_ccf",
    "construct_KpVsys"
]

def xcor(list_of_wls,list_of_orders,list_of_wlm,list_of_fxm,drv,RVrange,list_of_errors=None,
parallel=False):
    """
    This routine takes a combined dataset (in the form of lists of wl spaces,
    spectral orders and possible a matching list of errors on those spectal orders),
    as well as a list of templates (wlm,fxm) to cross-correlate with, and the cross-correlation
    parameters (drv,RVrange). The code takes on the order of ~10 minutes for an entire
    HARPS dataset, which appears to be superior to my old IDL pipe.

    The CCF used is the Geneva-style weighted average; not the Pearson CCF. Therefore
    it measures true 'average' planet lines, with flux on the y-axis of the CCF.
    The template must therefore be (something close to) a binary mask, with values
    inside spectral lines (the CCF is scale-invariant so their overall scaling
    doesn't matter),

    It returnsthe RV axis and lists of the resulting CCF, uncertainties and template integrals in a
    tuple.

    Thanks to Brett Morris (bmorris3), this code now implements a clever numpy broadcasting trick to
    instantly apply and interpolate the wavelength shifts of the model template onto
    the data grid in 2 dimensions. The matrix multiplication operator (originally
    recommended to me by Matteo Brogi) allowed this 2D template matrix to be multiplied
    with a 2D spectral order. np.hstack() is used to concatenate all orders end to end,
    effectively making a giant single spectral order (with NaNs in between due to masking).

    All these steps have eliminated ALL the forloops from the equation, and effectuated a
    speed gain of a factor between 2,000 and 3,000. The time to do cross correlations is now
    typically measured in 100s of milliseconds rather than minutes.

    This way of calculation does impose some strict rules on NaNs, though. To keep things fast,
    NaNs are now used to set the interpolated template matrix to zero wherever there are NaNs in the
    data. These NaNs are found by looking at the first spectrum in the stack, with the assumption
    that every NaN is in an all-NaN column. In the standard cross-correlation work-flow, isolated
    NaNs are interpolated over (healed), after all.

    The places where there are NaN columns in the data are therefore set to 0 in the template
    matrix. The NaN values themselves are then set to to an arbitrary value, since they will never
    weigh into the average by construction.


    The templates are provided as lists of wavelength and weight arrays. These are passed as lists
    to enable parallel computation on multiple threads without having to re-instantiate the stacks
    of spectra each time a template has to be cross-correlated with. Like other places in Tayph
    where parellelisation is applied, this is much faster in theory. However, the interpolation of
    the templates onto the data grid signifies a multiplication of the size of each template
    variable equal to the number of velocity steps times the size of the wavelength axis of the
    data. Multiplied by the number of templates, the total instantaneous memory load can be
    enormous. For example in case of the demo data, the data's wavelength axis weights 2.6 MB.
    Times 600 velocity steps makes 1.6GB. Times 10 templates makes 16GB, which is (more than) the
    total memory of a typical laptop. And often, one may wish to compute dozens of templates
    at a time. Running out of memory may crash the system in the worst case, or severely slow down
    the computation, making it less efficient than a standard serial computation in a for-loop.

    Parallel computation is therefore made optional, set by the parallel keyword. For small numbers
    of templates or velocity steps; or on systems with large amounts of memory, parallelisation may
    entail very significant speed gains. Otherwise, the user can keep this parameter switched off.



    Parameters
    ----------
    list_of_wls : list
        List of wavelength axes of the data.

    list_of_orders : list
        List of corresponding 2D orders.

    list_of_errors : list
        Optional, list of corresponding 2D error matrices.

    list_of_wlm : tuple or list of nd.arrays
        Wavelength axis of the template.

    list_of_fxm : tuple or list of nd.arrays
        Weight-axis of the template.

    drv : int,float
        The velocity step onto which the CCF is computed. Typically ~1 km/s.

    RVrange : int,float
        The velocity range in the positive and negative direction over which to
        evaluate the CCF. Typically >100 km/s.

    parallel: bool
        Optional, enabling parallel computation (potentially highly demanding on memory).

    Returns
    -------
    RV : np.ndarray
        The radial velocity grid over which the CCF is evaluated.

    CCF : list of np.ndarrays
        The weighted average flux in the spectrum as a function of radial velocity.

    CCF_E : list of np.ndarrays
        Optional. The error on each CCF point propagated from the error on the spectral values.

    Tsums : list of np.ndarrays
        The sums of the template for each velocity step. Used for normalising the CCFs.
    """

    import tayph.functions as fun
    import astropy.constants as const
    import tayph.util as ut
    from tayph.vartests import typetest,dimtest,postest,nantest,lentest
    import numpy as np
    import scipy.interpolate
    import astropy.io.fits as fits
    import matplotlib.pyplot as plt
    import sys
    import pdb
    if parallel: from joblib import Parallel, delayed

#===FIRST ALL SORTS OF TESTS ON THE INPUT===
    if len(list_of_wls) != len(list_of_orders):
        raise ValueError(f'In xcor(): List of wls and list of orders have different length '
        f'({len(list_of_wls)} & {len(list_of_orders)}).')

    t_init = ut.start()
    NT = len(list_of_fxm)
    lentest(list_of_wlm,NT,'list_of_wlm in ccf.xcor()')
    typetest(drv,[int,float],'drv in ccf.xcor')
    typetest(RVrange,float,'RVrange in ccf.xcor()',)
    postest(RVrange,'RVrange in ccf.xcor()')
    postest(drv,'drv in ccf.xcor()')
    nantest(drv,'drv in ccf.xcor()')
    nantest(RVrange,'RVrange in ccf.xcor()')


    drv = float(drv)
    N=len(list_of_wls)#Number of orders.

    if np.ndim(list_of_orders[0]) == 1.0:
        n_exp=1
    else:
        n_exp=len(list_of_orders[0][:,0])#Number of exposures.

#===Then check that all orders indeed have n_exp exposures===
        for i in range(N):
            if len(list_of_orders[i][:,0]) != n_exp:
                raise ValueError(f'In xcor(): Not all orders have {n_exp} exposures.')

#===END OF TESTS. NOW DEFINE CONSTANTS===
    c=const.c.to('km/s').value#In km/s
    RV= np.arange(-RVrange, RVrange+drv, drv, dtype=float) #The velocity grid.
    beta=1.0-RV/c#The doppler factor with which each wavelength is to be shifted.
    n_rv = len(RV)


#===STACK THE ORDERS IN MASSIVE CONCATENATIONS===
    stack_of_orders = np.hstack(list_of_orders)
    stack_of_wls = np.concatenate(list_of_wls)
    if list_of_errors is not None:
        stack_of_errors2 = np.hstack(list_of_errors)**2#Stack them horizontally and square.
        #Check that the number of NaNs is the same in the orders as in the errors on the orders;
        #and that they are in the same place; meaning that if I add the errors to the orders, the
        #number of NaNs does not increase (NaN+value=NaN).
        if (np.sum(np.isnan(stack_of_orders)) != np.sum(np.isnan(stack_of_errors2+
        stack_of_orders))) and (np.sum(np.isnan(stack_of_orders)) !=
        np.sum(np.isnan(stack_of_errors2))):
            raise ValueError(f"in CCF: The number of NaNs in list_of_orders and list_of_errors is "
            f"not equal ({np.sum(np.isnan(list_of_orders))},{np.sum(np.isnan(list_of_errors2))})")

#===HERE IS THE JUICY BIT===
#===FIRST, FIND AND MARK NANS===
    #We check whether there are isolated NaNs:
    n_nans = np.sum(np.isnan(stack_of_orders),axis=0)#This is the total number of NaNs in each
    #column. Whenever the number of NaNs equals the length of a column, we ignore them:
    n_nans[n_nans==len(stack_of_orders)]=0
    if np.max(n_nans)>0:#If there are any columns which still have NaNs in them, we need to crash.
        raise ValueError(f"in CCF: Not all NaN values are purely in data columns. There are still "
        "isolated NaNs in the data. Remove those.")

    #So we checked that all NaNs were in whole columns. These columns have the following indices:
    nan_columns =  np.isnan(stack_of_orders[0])
    #We set all of them to arbitrarily high values, but set the template to zero in those locations
    #(see below). The reason this is OK is because at no time does the template see these values.
    #The fact that the template shifts doesn't matter: If a line appears or disappears by shifting
    #into a NaN-blocked region, T_sum changes as a function of RV, but not as a function of time.
    #So at for each time (column) of the CCF, T_sum is identical.

    #The only effect that could happen is that the planet signal appears from a NaN blocked area,
    #creating a tiny time-dependence of the planet lines included in the average.
    #(in fact this is implicitly always the case due to the edges of spectral orders and the
    #edges of the data)
    #Solving this would require that the template ignores all lines that pass over an edge or
    #to a NaN affected areay. To be implemented?

    stack_of_orders[np.isnan(stack_of_orders)] = 47e20#Set NaNs to arbitrarily high values.
    if list_of_errors is not None: stack_of_errors2[np.isnan(stack_of_errors2)] = 42e20#we have
    #already tested way before that the error array has NaNs are in the same place.



    shifted_wls = stack_of_wls * beta[:, np.newaxis]#2D broadcast of wl_data, each row shifted by
    #beta[i].
    def do_xcor(i):#Calling XCOR on template i.
        wlm = list_of_wlm[i]
        fxm = list_of_fxm[i]
        typetest(wlm,np.ndarray,'wlm in ccf.xcor')
        typetest(fxm,np.ndarray,'fxm in ccf.xcor')
        nantest(wlm,'fxm in ccf.xcor()')
        nantest(fxm,'fxm in ccf.xcor()')
        dimtest(wlm,[len(fxm)],'wlm in ccf.xcor()')

        #A wild 2D thing has appeared! It's super effective!
        T = scipy.interpolate.interp1d(wlm,fxm, bounds_error=False, fill_value=0)(shifted_wls)
        #... it's also massive in memory: A copy of the data's wavelength axis for EACH velocity
        #step. For the KELT-9 demo data, that's 2.7MB, times 600 velocity steps = 1.6 GB, times
        #NT templates. So if you're running 20 templates in a row, good luck!

        #How do we solve this?
        T[:,nan_columns] = 0.0#All NaNs are assumed to be in all-NaN columns. If that is not true,
        #the below nantest will fail.
        T_sums = np.sum(T,axis = 1)
        T = T.T/T_sums
        CCF = stack_of_orders @ T#Here it the entire cross-correlation. Over all orders and
        #velocity steps. No forloops.
        CCF_E = CCF*0.0
        #If the errors were provided, we do the same to those:
        if list_of_errors is not None: CCF_E = np.sqrt(stack_of_errors2 @ T**2)#This has been
        #mathematically proven: E^2 = e^2 * T^2


        #===THAT'S ALL. TEST INTEGRITY AND RETURN THE RESULT===
        nantest(CCF,'CCF in ccf.xcor()')#If anything went wrong with NaNs in the data, these tests
        #will fail because the matrix operation @ is non NaN-friendly.
        nantest(CCF_E,'CCF_E in ccf.xcor()')
        return(CCF,CCF_E,T_sums)

    if parallel:#This here takes a lot of memory.
        list_of_CCFs, list_of_CCF_Es, list_of_T_sums = zip(*Parallel(n_jobs=NT)(delayed(do_xcor)(i)
        for i in range(NT)))
    else:
        list_of_CCFs, list_of_CCF_Es, list_of_T_sums = zip(*[do_xcor(i) for i in range(NT)])

    if list_of_errors != None:
        return(RV,list_of_CCFs,list_of_CCF_Es,list_of_T_sums)
    return(RV,list_of_CCFs,list_of_T_sums)












#BINARY MASK VERSION
#This is for a list of orders. Because I do a search_sort, I cannot just stitch the orders together
#like before. But I figured out the version of this that indeces and matrix-multiplies the whole
#cross-correlation in one go for each order. That is nearly unreadable, so I have left the
#non-vectorised, for-loopy version hidden behind the fast=False keyword, so that you can verify that
#the vectorised and the loopy version yield the same answer.
def mask_cor3D(list_of_wls,list_of_orders,wlT,T,drv=1.0,RVrange=200,fast=True):
    import numpy as np
    import astropy.constants as const
    import pdb
    #How to deal with the edges?
    #If no_edges is True, any lines that are located beyond the edge of the wavelength range at
    #any RV shift are entirely excluded from the cross-correlation. The purpose is to make sure that
    #the cross-correlation measures the same spectral lines at all RV shifts. This means that
    #if RVrange is large compared to the wavelength range, setting this to True will eat away many
    #of your spectral lines (and also make execution faster).


    c=const.c.to('km/s').value#In km/s
    RV= np.arange(-RVrange, RVrange+drv, drv, dtype=float) #fun.findgen(2.0*RVrange/drv+1)*drv-RVrange#..... CONTINUE TO DEFINE THE VELOCITY GRID
    beta=1.0+RV/c#The doppler factor with which each wavelength is to be shifted.
    n_rv = len(RV)
    CCF = np.zeros((len(list_of_orders[0]),len(RV)))
    #I treat binary masks as line lists. Instead of evaluating a template spectrum on the data, we
    #just search for the nearest neighbour datapoints, distributing the weight in the line over them
    #and then integrating - for each line and for each radial velocity shift.

    T_sum = 0
    for o in range(len(list_of_wls)):#Loop ove orders.
        wl = list_of_wls[o]
        order = list_of_orders[o]
        sel_lines = (wlT>np.min(wl)*(1+np.max(RV)/c))&(wlT<np.max(wl)/(1+np.max(RV)/c))
        if len(sel_lines) > 0:
            wlT_order = wlT[sel_lines]#Select only the lines that actually fall into the wavelength array for all velocity shifts.
            T_order   = T[sel_lines]
            T_sum+=np.sum(T_order)
            shifted_wlT = wlT_order * beta[:, np.newaxis]
            indices = np.searchsorted(wl,shifted_wlT)#The indices to the right of each target line.
            #Every row in indices is a list of N spectral lines.
            #For large numbers of lines, this vectorised search is not faster than serial, but we use
            #the 2D output to cause total vectorised mayhem next.


            if fast:#Witness the power of this fully armed and operational battle station!
                w = (wl[indices]-shifted_wlT)/(wl[indices] - wl[indices-1])
                CCF += (order[:,indices]*(1-w)+order[:,indices-1]*w)@T_order



            else:
                for j in range(len(beta)):#==len(indices, which is len(RV)*(N+1)).
                    #W = wl*0.0 #Activate this to plot the "template" at this value of beta..
                    for n,i in enumerate(indices[j]):
                        bin = wl[i-1:i+1]#This selects 2 elements: wl[i-1] and wl[i]. So the : means "until".
                        w = (bin[1]-shifted_wlT[j,n])/(bin[1]-bin[0]) #Weights are constructed onto the data line by line.
                        #This number is small if bin[1]=wlT[n], meaning that it measures the weight on the point
                        #left of the target line position. So wl[i] gets weight w-1, and wl[i-1] gets w.
                        #W[i]+=1-w*T_order[n]#Activate this to plot the "template" at this value of beta.
                        #W[i-1]+=w*T_order[n]
                        CCF[:,j] += (order[:,i]*(1-w) + order[:,i-1]*w)*T_order[n]
    return(RV,CCF/T_sum)











def clean_ccf(rv,ccf,ccf_e,dp):
    """
    This routine normalizes the CCF fluxes and subtracts the average out of
    transit CCF, using the transit lightcurve as a mask.


    Parameters
    ----------

    rv : np.ndarray
        The radial velocity axis

    ccf : np.ndarray
        The CCF with second dimension matching the length of rv.

    ccf_e : np.ndarray
        The error on ccf.

    dp : str or path-like
        The datapath of the present dataset, to establish which exposures in ccf
        are in or out of transit.

    Returns
    -------

    ccf_n : np.ndarray
        The transit-lightcurve normalised CCF.

    ccf_ne : np.ndarray
        The error on ccf_n

    ccf_nn : np.ndarray
        The CCF relative to the out-of-transit time-averaged, if sufficient (>25%
        of the time-series) out of transit exposures were available. Otherwise, the
        average over the entire time-series is used.

    ccf_ne : np.array
        The error on ccf_nn.


    """

    import numpy as np
    import tayph.functions as fun
    import tayph.util as ut
    from matplotlib import pyplot as plt
    import pdb
    import math
    import tayph.system_parameters as sp
    import tayph.operations as ops
    import astropy.io.fits as fits
    import sys
    import copy
    from tayph.vartests import typetest,dimtest,nantest

    typetest(rv,np.ndarray,'rv in clean_ccf()')
    typetest(ccf,np.ndarray,'ccf in clean_ccf')
    typetest(ccf_e,np.ndarray,'ccf_e in clean_ccf')
    dp=ut.check_path(dp)
    dimtest(ccf,[0,len(rv)])
    dimtest(ccf_e,[0,len(rv)])
    nantest(rv,'rv in clean_ccf()')
    nantest(ccf,'ccf in clean_ccf()')
    nantest(ccf_e,'ccf_e in clean_ccf()')
    #ADD PARAMGET DV HERE.

    transit=sp.transit(dp)
    # transitblock = fun.rebinreform(transit,len(rv))

    Nrv = int(math.floor(len(rv)))

    baseline_ccf  = np.hstack((ccf[:,0:int(0.25*Nrv)],ccf[:,int(0.75*Nrv):]))
    baseline_ccf_e= np.hstack((ccf_e[:,0:int(0.25*Nrv)],ccf_e[:,int(0.75*Nrv):]))
    baseline_rv   = np.hstack((rv[0:int(0.25*Nrv)],rv[int(0.75*Nrv):]))
    meanflux=np.median(baseline_ccf,axis=1)#Normalize the baseline flux, but away from the signal of the planet.
    meanflux_e=1.0/len(baseline_rv)*np.sqrt(np.nansum(baseline_ccf_e**2.0,axis=1))#1/N times sum of squares.
    #I validated that this is approximately equal to ccf_e/sqrt(N).
    meanblock=fun.rebinreform(meanflux,len(rv))
    meanblock_e=fun.rebinreform(meanflux_e,len(rv))

    ccf_n = ccf/meanblock.T
    ccf_ne = np.abs(ccf_n) * np.sqrt((ccf_e/ccf)**2.0 + (meanblock_e.T/meanblock.T)**2.0)#R=X/Z -> dR = R*sqrt( (dX/X)^2+(dZ/Z)^2 )
    #I validated that this is essentially equal to ccf_e/meanblock.T; as expected because the error on the mean spectrum is small compared to ccf_e.


    if np.sum(transit==1) == 0:
        print('------WARNING in Cleaning: The data contains only in-transit exposures.')
        print('------The mean ccf is taken over the entire time-series.')
        meanccf=np.nanmean(ccf_n,axis=0)
        meanccf_e=1.0/len(transit)*np.sqrt(np.nansum(ccf_ne**2.0,axis=0))#I validated that this is approximately equal
        #to sqrt(N)*ccf_ne, where N is the number of out-of-transit exposures.
    elif np.sum(transit==1) <= 0.25*len(transit):
        print('------WARNING in Cleaning: The data contains very few (<25%) out of transit exposures.')
        print('------The mean ccf is taken over the entire time-series.')
        meanccf=np.nanmean(ccf_n,axis=0)
        meanccf_e=1.0/len(transit)*np.sqrt(np.nansum(ccf_ne**2.0,axis=0))#I validated that this is approximately equal
        #to sqrt(N)*ccf_ne, where N is the number of out-of-transit exposures.
    if np.min(transit) == 1.0:
        print('------WARNING in Cleaning: The data is not predicted to contain in-transit exposures.')
        print(f'------If you expect to be dealing with transit-data, please check the ephemeris '
        f'at {dp}')
        print('------The mean ccf is taken over the entire time-series.')
        meanccf=np.nanmean(ccf_n,axis=0)
        meanccf_e=1.0/len(transit)*np.sqrt(np.nansum(ccf_ne**2.0,axis=0))#I validated that this is approximately equal
        #to sqrt(N)*ccf_ne, where N is the number of out-of-transit exposures.
    else:
        meanccf=np.nanmean(ccf_n[transit == 1.0,:],axis=0)
        meanccf_e=1.0/np.sum(transit==1)*np.sqrt(np.nansum(ccf_ne[transit == 1.0,:]**2.0,axis=0))#I validated that this is approximately equal
        #to sqrt(N)*ccf_ne, where N is the number of out-of-transit exposures.




    meanblock2=fun.rebinreform(meanccf,len(meanflux))
    meanblock2_e=fun.rebinreform(meanccf_e,len(meanflux))

    ccf_nn = ccf_n/meanblock2#MAY NEED TO DO SUBTRACTION INSTEAD TOGETHER W. NORMALIZATION OF LIGHTCURVE. SEE ABOVE.
    ccf_nne = np.abs(ccf_n/meanblock2)*np.sqrt((ccf_ne/ccf_n)**2.0 + (meanblock2_e/meanblock2)**2.0)
    #I validated that this error is almost equal to ccf_ne/meanccf


    #ONLY WORKS IF LIGHTCURVE MODEL IS ACCURATE, i.e. if Euler observations are available.
    # print("---> WARNING IN CLEANING.CLEAN_CCF(): NEED TO ADD A FUNCTION THAT YOU CAN NORMALIZE BY THE LIGHTCURVE AND SUBTRACT INSTEAD OF DIVISION!")
    return(ccf_n,ccf_ne,ccf_nn-1.0,ccf_nne)


def filter_ccf(rv,ccf,v_width):
    """
    Performs a high-pass filter on a CCF.
    """
    import copy
    import tayph.operations as ops
    ccf_f = copy.deepcopy(ccf)
    wiggles = copy.deepcopy(ccf)*0.0
    dv = rv[1]-rv[0]#Assumes that the RV axis is constant.
    w = v_width / dv
    for i,ccf_row in enumerate(ccf):
        wiggle = ops.smooth(ccf_row,w,mode='gaussian')
        wiggles[i] = wiggle
        ccf_f[i] = ccf_row-wiggle
    return(ccf_f,wiggles)

def shift_ccf(RV,CCF,drv):
    """
    This shifts the rows of a CCF based on velocities provided in drv.
    Improve those tests. I got functions for that.
    """
    import tayph.functions as fun
    import tayph.util as ut
    import numpy as np
    #import matplotlib.pyplot as plt
    import scipy.interpolate
    import pdb
    import astropy.io.fits as fits
    import matplotlib.pyplot as plt
    import sys
    import scipy.ndimage.interpolation as scint

    if np.ndim(CCF) == 1.0:
        print("ERROR in shift_ccf: CCF should be a 2D block.")
        sys.exit()
    else:
        n_exp=len(CCF[:,0])#Number of exposures.
        n_rv=len(CCF[0,:])
    if len(RV) != n_rv:
        print('ERROR in shift_ccf: RV does not have the same length as the base size of the CCF block.')
        sys.exit()
    if len(drv) != n_exp:
        print('ERROR in shift_ccf: drv does not have the same height as the CCF block.')
        sys.exit()
    dv = RV[1]-RV[0]
    CCF_new=CCF*0.0
    for i in range(n_exp):
        #C_i=scipy.interpolate.interp1d(RV,CCF[i],fill_value=(0.0,0.0))
        #CCF_new[i,:] = C_i(RV-drv[i]*2.0)
        CCF_new[i,:] = scint.shift(CCF[i],drv[i]/dv,mode='nearest',order=1)
    return(CCF_new)


def construct_KpVsys(rv,ccf,ccf_e,dp,kprange=[0,300],dkp=1.0):
    """The name says it all. Do good tests."""
    import tayph.functions as fun
    import tayph.operations as ops
    import numpy as np
    import tayph.system_parameters as sp
    import matplotlib.pyplot as plt
    import astropy.io.fits as fits
    import tayph.util as ut
    import sys
    import pdb
    from joblib import Parallel, delayed

    Kp = np.arange(kprange[0], kprange[1]+dkp, dkp, dtype=float) #fun.findgen((kprange[1]-kprange[0])/dkp+1)*dkp+kprange[0]
    n_exp = np.shape(ccf)[0]
    KpVsys = np.zeros((len(Kp),len(rv)))
    KpVsys_e = np.zeros((len(Kp),len(rv)))
    transit = sp.transit(dp)-1.0
    transit /= np.nansum(transit)
    transitblock = fun.rebinreform(transit,len(rv)).T

    def Kp_parallel(i):
        dRV = sp.RV(dp,vorb=i)*(-1.0)
        ccf_shifted = shift_ccf(rv,ccf,dRV)
        ccf_e_shifted = shift_ccf(rv,ccf_e,dRV)
        return (np.nansum(transitblock * ccf_shifted,axis=0), (np.nansum((transitblock*ccf_e_shifted)**2.0,axis=0))**0.5)

    KpVsys, KpVsys_e = zip(*Parallel(n_jobs=-1)(delayed(Kp_parallel)(i) for i in Kp))


    return(Kp,KpVsys,KpVsys_e)

    # CCF_total = np.zeros((n_exp,n_rv))
    #
    # Tsums = fun.findgen(n_rv)*0.0*float('NaN')
    # T_i = scipy.interpolate.interp1d(wlm,fxm, bounds_error=False, fill_value=0)
    # t1=ut.start()
    #
    # for i,order in enumerate(list_of_orders):
    #     CCF = np.zeros((n_exp,n_rv))*float('NaN')
    #     shifted_wavelengths = list_of_wls[i] * beta[:, np.newaxis]#2D broadcast of wl_data, each row shifted by beta[i].
    #     T=T_i(shifted_wavelengths)
    #     masked_order = np.ma.masked_array(order,np.isnan(order))
    #
    #     for j,spectrum in enumerate(masked_order):
    #         x = np.repeat(spectrum[:, np.newaxis],n_rv, axis=1).T
    #         CCF[j] = np.ma.average(x,weights=T, axis=1)
    #
    #     CCF_total+=CCF
    #     ut.statusbar(i,len(list_of_orders))
    #
    # ut.end(t1)
    # ut.save_stack('test_ccf_compared.fits',[fits.getdata('test_ccf.fits'),CCF])
    # pdb.set_trace()



#
# #===Define the output CCF array.
#     CCF = np.zeros((n_exp,len(shift)))#*float('NaN')
#     CCF_E = CCF*0.0
#     Tsums = fun.findgen(len(shift))*0.0*float('NaN')
# #===Then comes the big forloop.
#     #The outer loop is the shifts. For each, we loop through the orders.
#
#
#     counter = 0
#     for i in range(len(shift)):
#         T_sum = 0.0
#         wlms=wlm*shift[i]
#         for j in range(N):
#             wl=list_of_wls[j]
#             order=list_of_orders[j]
#
#             T_i=scipy.interpolate.interp1d(wlms[(wlms >= np.min(wl)-10.0) & (wlms <= np.max(wl)+10.0)],fxm[(wlms >= np.min(wl)-10.0) & (wlms <= np.max(wl)+10.0)],bounds_error=False,fill_value=0.0)
#             T = T_i(wl)
#             T_matrix=fun.rebinreform(T,n_exp)
#             CCF[:,i]+=np.nansum(T_matrix*order,1)
#             if list_of_errors != None:
#                 sigma=list_of_errors[j]
#                 CCF_E[:,i]+=np.nansum((T_matrix*sigma)**2.0,1)#CHANGE THIS WITH PYTHON @ OPERATOR AS PER HOW MATTEO CODES THIS. Matrix multiplication is 20x faster than normal multiplication + summing.
#             T_sum+=np.sum(np.abs(T))
#
#
#         CCF[:,i] /= T_sum
#         CCF_E[:,i] /= T_sum**2.0
#         Tsums[i] = T_sum
#         T_sum = 0.0
#         counter += 1
#         ut.statusbar(i,shift)
#     nantest(CCF,'CCF in ccf.xcor()')
#     nantest(CCF_E,'CCF_E in ccf.xcor()')
#
#
#     if plot == True:
#         fig, (a0,a1,a2) = plt.subplots(3,1,gridspec_kw = {'height_ratios':[1,1,1]},figsize=(10,7))
#         a02 = a0.twinx()
#         for i in range(N):
#             meanspec=np.nanmean(list_of_orders[i],axis=0)
#             meanwl=list_of_wls[i]
#             T_i=scipy.interpolate.interp1d(wlm[(wlm >= np.min(meanwl)-0.0) & (wlm <= np.max(meanwl)+0.0)],fxm[(wlm >= np.min(meanwl)-0.0) & (wlm <= np.max(meanwl)+0.0)],fill_value='extrapolate')
#             T = T_i(meanwl)
#         # meanspec-=min(meanspec)
#         # meanspec/=np.max(meanspec)
#         # T_plot-=np.min(T)
#         # T_plot/=np.median(T_plot)
#             a02.plot(meanwl,T,color='orange',alpha=0.3)
#             a0.plot(meanwl,meanspec,alpha=0.5)
#         a1.plot(RV,Tsums,'.')
#         if list_of_errors != None:
#             a2.errorbar(RV,np.mean(CCF,axis=0),fmt='.',yerr=np.mean(np.sqrt(CCF_E),axis=0)/np.sqrt(n_exp),zorder=3)
#         else:
#             a2.plot(RV,np.mean(CCF,axis=0),'.')
#         a0.set_title('t-averaged data and template')
#         a1.set_title('Sum(T)')
#         a2.set_title('t-averaged CCF')
#         a0.tick_params(axis='both',labelsize='5')
#         a02.tick_params(axis='both',labelsize='5')
#         a1.tick_params(axis='both',labelsize='5')
#         a2.tick_params(axis='both',labelsize='5')
#         fig.tight_layout()
#         plt.show()
#                     # a0.set_xlim((674.5,675.5))
#                     # a1.set_xlim((674.5,675.5))
#     if list_of_errors != None:
#         return(RV,CCF,np.sqrt(CCF_E),Tsums)
#     return(RV,CCF,Tsums)
