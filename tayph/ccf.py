def xcor(list_of_wls,list_of_orders,wlm,fxm,drv,RVrange,plot=False,list_of_errors=None):
    """
    This routine takes a combined dataset (in the form of lists of wl spaces,
    spectral orders and possible a matching list of errors on those spectal orders),
    as well as a template (wlm,fxm) to cross-correlate with, and the cross-correlation
    parameters (drv,RVrange). The code takes on the order of ~10 minutes for an entire
    HARPS dataset, which appears to be superior to my old IDL pipe.

    The CCF used is the Geneva-style weighted average; not the Pearson CCF. Therefore
    it measures true 'average' planet lines, with flux on the y-axis of the CCF.
    The template must therefore be (something close to) a binary mask, with values
    inside spectral lines (the CCF is scale-invariant so their overall scaling
    doesn't matter),

    It returns the RV axis and the resulting CCF in a tuple.

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
    NaNs are now used to set the interpolated template matrix to zero wherever there are NaNs in the data.
    These NaNs are found by looking at the first spectrum in the stack, with the assumption that
    every NaN is in an all-NaN column. In the standard cross-correlation work-flow, isolated
    NaNs are interpolated over (healed), after all.

    The places where there are NaN columns in the data are therefore set to 0 in the template matrix.
    The NaN values themselves are then set to to an arbitrary value, since they will never
    weigh into the average by construction.


    Parameters
    ----------
    list_of_wls : list
        List of wavelength axes of the data.

    list_of_orders : list
        List of corresponding 2D orders.

    list_of_errors : list
        Optional, list of corresponding 2D error matrices.

    wlm : np.ndarray
        Wavelength axis of the template.

    fxm : np.ndarray
        Weight-axis of the template.

    drv : int,float
        The velocity step onto which the CCF is computed. Typically ~1 km/s.

    RVrange : int,float
        The velocity range in the positive and negative direction over which to
        evaluate the CCF. Typically >100 km/s.

    plot : bool
        Set to True for diagnostic plotting.

    Returns
    -------
    RV : np.ndarray
        The radial velocity grid over which the CCF is evaluated.

    CCF : np.ndarray
        The weighted average flux in the spectrum as a function of radial velocity.

    CCF_E : np.ndarray
        Optional. The error on each CCF point propagated from the error on the spectral values.

    Tsums : np.ndarray
        The sum of the template for each velocity step. Used for normalising the CCFs.
    """

    import tayph.functions as fun
    import astropy.constants as const
    import tayph.util as ut
    from tayph.vartests import typetest,dimtest,postest,nantest
    import numpy as np
    import scipy.interpolate
    import astropy.io.fits as fits
    import matplotlib.pyplot as plt
    import sys
    import pdb

#===FIRST ALL SORTS OF TESTS ON THE INPUT===
    if len(list_of_wls) != len(list_of_orders):
        raise ValueError(f'In xcor(): List of wls and list of orders have different length ({len(list_of_wls)} & {len(list_of_orders)}).')

    dimtest(wlm,[len(fxm)],'wlm in ccf.xcor()')
    typetest(wlm,np.ndarray,'wlm in ccf.xcor')
    typetest(fxm,np.ndarray,'fxm in ccf.xcor')
    typetest(drv,[int,float],'drv in ccf.xcor')
    typetest(RVrange,float,'RVrange in ccf.xcor()',)
    postest(RVrange,'RVrange in ccf.xcor()')
    postest(drv,'drv in ccf.xcor()')
    nantest(wlm,'fxm in ccf.xcor()')
    nantest(fxm,'fxm in ccf.xcor()')
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
    RV=fun.findgen(2.0*RVrange/drv+1)*drv-RVrange#..... CONTINUE TO DEFINE THE VELOCITY GRID
    beta=1.0-RV/c#The doppler factor with which each wavelength is to be shifted.
    n_rv = len(RV)


#===STACK THE ORDERS IN MASSIVE CONCATENATIONS===
    stack_of_orders = np.hstack(list_of_orders)
    stack_of_wls = np.concatenate(list_of_wls)
    if list_of_errors is not None:
        stack_of_errors = np.hstack(list_of_errors)#Stack them horizontally

        #Check that the number of NaNs is the same in the orders as in the errors on the orders;
        #and that they are in the same place; meaning that if I add the errors to the orders, the number of
        #NaNs does not increase (NaN+value=NaN).
        if (np.sum(isnan(stack_of_orders)) != np.sum(np.isnan(stack_of_errors+stack_of_orders))) and (np.sum(isnan(stack_of_orders)) != np.sum(np.isnan(stack_of_errors))):
            raise ValueError(f"in CCF: The number of NaNs in list_of_orders and list_of_errors is not equal ({np.sum(np.isnan(list_of_orders))},{np.sum(np.isnan(list_of_errors))})")

#===HERE IS THE JUICY BIT===
    shifted_wavelengths = stack_of_wls * beta[:, np.newaxis]#2D broadcast of wl_data, each row shifted by beta[i].
    T = scipy.interpolate.interp1d(wlm,fxm, bounds_error=False, fill_value=0)(shifted_wavelengths)#...making this a 2D thing.
    T[:,np.isnan(stack_of_orders[0])] = 0.0#All NaNs are assumed to be in all-NaN columns. If that is not true, the below nantest will fail.
    T_sums = np.sum(T,axis = 1)
    stack_of_orders[np.isnan(stack_of_orders)] = 47e20#Set NaNs to arbitrarily high values.
    CCF = stack_of_orders @ T.T/T_sums#Here it the entire cross-correlation. Over all orders and velocity steps. No forloops.
    CCF_E = CCF*0.0

    #If the errors were provided, we do the same to those:
    if list_of_errors is not None:
        stack_of_errors[np.isnan(stack_of_errors)] = 42e20#we have already tested that these NaNs are in the same place.
        CCF_E = stack_of_errors**2 @ (T.T/T_sums)**2#This has been mathematically proven.


#===THAT'S ALL. TEST INTEGRITY AND RETURN THE RESULT===
    nantest(CCF,'CCF in ccf.xcor()')#If anything went wrong with NaNs in the data, these tests will fail because the matrix operation @ is non NaN-friendly.
    nantest(CCF_E,'CCF_E in ccf.xcor()')


    if list_of_errors != None:
        return(RV,CCF,np.sqrt(CCF_E),T_sums)
    return(RV,CCF,T_sums)




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
