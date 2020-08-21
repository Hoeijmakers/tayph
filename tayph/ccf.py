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
    shift=1.0+RV/c#Array along which the template will be shifted during xcor.



#===Define the output CCF array.
    CCF = np.zeros((n_exp,len(shift)))#*float('NaN')
    CCF_E = CCF*0.0
    Tsums = fun.findgen(len(shift))*0.0*float('NaN')
#===Then comes the big forloop.
    #The outer loop is the shifts. For each, we loop through the orders.


    counter = 0
    for i in range(len(shift)):
        T_sum = 0.0
        wlms=wlm*shift[i]
        for j in range(N):
            wl=list_of_wls[j]
            order=list_of_orders[j]

            T_i=scipy.interpolate.interp1d(wlms[(wlms >= np.min(wl)-10.0) & (wlms <= np.max(wl)+10.0)],fxm[(wlms >= np.min(wl)-10.0) & (wlms <= np.max(wl)+10.0)],bounds_error=False,fill_value=0.0)
            T = T_i(wl)
            T_matrix=fun.rebinreform(T,n_exp)
            CCF[:,i]+=np.nansum(T_matrix*order,1)
            if list_of_errors != None:
                sigma=list_of_errors[j]
                CCF_E[:,i]+=np.nansum((T_matrix*sigma)**2.0,1)#CHANGE THIS WITH PYTHON @ OPERATOR AS PER HOW MATTEO CODES THIS. Matrix multiplication is 20x faster than normal multiplication + summing.
            T_sum+=np.sum(np.abs(T))


        CCF[:,i] /= T_sum
        CCF_E[:,i] /= T_sum**2.0
        Tsums[i] = T_sum
        T_sum = 0.0
        counter += 1
        ut.statusbar(i,shift)
    nantest(CCF,'CCF in ccf.xcor()')
    nantest(CCF_E,'CCF_E in ccf.xcor()')


    if plot == True:
        fig, (a0,a1,a2) = plt.subplots(3,1,gridspec_kw = {'height_ratios':[1,1,1]},figsize=(10,7))
        a02 = a0.twinx()
        for i in range(N):
            meanspec=np.nanmean(list_of_orders[i],axis=0)
            meanwl=list_of_wls[i]
            T_i=scipy.interpolate.interp1d(wlm[(wlm >= np.min(meanwl)-0.0) & (wlm <= np.max(meanwl)+0.0)],fxm[(wlm >= np.min(meanwl)-0.0) & (wlm <= np.max(meanwl)+0.0)],fill_value='extrapolate')
            T = T_i(meanwl)
        # meanspec-=min(meanspec)
        # meanspec/=np.max(meanspec)
        # T_plot-=np.min(T)
        # T_plot/=np.median(T_plot)
            a02.plot(meanwl,T,color='orange',alpha=0.3)
            a0.plot(meanwl,meanspec,alpha=0.5)
        a1.plot(RV,Tsums,'.')
        if list_of_errors != None:
            a2.errorbar(RV,np.mean(CCF,axis=0),fmt='.',yerr=np.mean(np.sqrt(CCF_E),axis=0)/np.sqrt(n_exp),zorder=3)
        else:
            a2.plot(RV,np.mean(CCF,axis=0),'.')
        a0.set_title('t-averaged data and template')
        a1.set_title('Sum(T)')
        a2.set_title('t-averaged CCF')
        a0.tick_params(axis='both',labelsize='5')
        a02.tick_params(axis='both',labelsize='5')
        a1.tick_params(axis='both',labelsize='5')
        a2.tick_params(axis='both',labelsize='5')
        fig.tight_layout()
        plt.show()
                    # a0.set_xlim((674.5,675.5))
                    # a1.set_xlim((674.5,675.5))
    if list_of_errors != None:
        return(RV,CCF,np.sqrt(CCF_E),Tsums)
    return(RV,CCF,Tsums)
