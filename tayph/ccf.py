__all__ = [
    "xcor",
    "mask_cor",
    "clean_ccf",
    "filter_ccf",
    "construct_KpVsys"
]

def xcor(list_of_wls,list_of_orders,list_of_wlm,list_of_fxm,drv,RVrange,list_of_errors=None,
parallel=False):
    """
    This routine takes a combined dataset (in the form of lists of wl spaces,
    spectral orders and possible a matching list of errors on those spectal orders),
    as well as a list of spectral templates (wlm,fxm) to cross-correlate with, and the
    cross-correlation parameters (drv,RVrange). The code can be run in parallel if sufficient
    RAM is available.

    The CCF used is the Geneva-style weighted average (not the Pearson CCF) with a spectral
    template. Therefore it measures true 'average' planet lines, with relative flux on the y-axis
    of the CCF. The template should be a weighting function with narrow lines, something akin to a
    binary mask but defined on a spectral grid. These lines are typically computed via some form of
    radiative transfer model. The CCF will be scale-invariant so their overall scaling does not
    matter.

    It returnsthe RV axis and lists of the resulting CCF, uncertainties and template integrals in a
    tuple.

    Thanks to Brett Morris (bmorris3), this code implements a clever numpy broadcasting trick to
    instantly apply and interpolate the wavelength shifts of the model template onto
    the data grid in 2 dimensions. The matrix multiplication operator (originally
    recommended to me by Matteo Brogi) allows this 2D template matrix to be multiplied
    with a 2D spectral order. np.hstack() is used to concatenate all orders end to end,
    effectively making a giant single spectral order (perhaps with NaNs in between due to masking).

    All these steps have eliminated ALL the forloops from the equation, and effectuated a
    speed gain of a factor between 2,000 and 3,000. The time to do cross correlations is now
    typically measured in 100s of milliseconds rather than minutes.

    This way of calculation does impose some strict rules on NaNs. To keep things fast,
    NaNs are now used to set the interpolated template matrix to zero wherever there are NaNs in the
    data. These NaNs are found by looking at the first spectrum in the stack, with the assumption
    that every NaN is in an all-NaN column. In the standard cross-correlation work-flow, isolated
    NaNs are interpolated over (healed), after all.

    The places where there are NaN columns in the data are therefore set to 0 in the template
    matrix. The NaN values themselves are then set to to an arbitrary value, since they will never
    weigh into the average by construction.


    Multiple templates are provided as lists of wavelength and weight arrays. These are passed as
    lists to enable parallel computation on multiple threads without having to re-instantiate the
    stacks of spectra each time a template has to be cross-correlated with. Like other places in
    Tayph where parellelisation is applied, this is much faster in theory. However, the
    interpolation of the templates onto the data grid signifies a multiplication of the size of each
     template variable equal to the number of velocity steps times the size of the wavelength axis
    of the data. Multiplied by the number of templates, the total instantaneous memory load can be
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
    NT = len(list_of_fxm)

    if parallel == True:
        NC = NT*1
    if int(parallel) > 1:
        NC = int(parallel)
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
    ut.tprint('------Stacking orders, wavelengths and errors')
    stack_of_orders = np.hstack(list_of_orders)#This is very memory intensive, essentially copying
    stack_of_wls = np.concatenate(list_of_wls)#the data in memory again... Can be 3 to 4 GB in
    #case of ESPRESSO. Issue 93 deals with this.

    if list_of_errors is not None:
        stack_of_errors2 = np.hstack(list_of_errors)**2#Stack them horizontally and square.
        #Check that the number of NaNs is the same in the orders as in the errors on the orders;
        #and that they are in the same place; meaning that if I add the errors to the orders, the
        #number of NaNs does not increase (NaN+value=NaN).
        if (np.sum(np.isnan(stack_of_orders)) != np.sum(np.isnan(stack_of_errors2+
        stack_of_orders))) and (np.sum(np.isnan(stack_of_orders)) !=
        np.sum(np.isnan(stack_of_errors2))):
            raise ValueError(f"in CCF: The number of NaNs in list_of_orders and list_of_errors is "
            f"not equal ({np.sum(np.isnan(list_of_orders))},{np.sum(np.isnan(list_of_errors))})")

#===HERE IS THE JUICY BIT===
#===FIRST, FIND AND MARK NANS===
    #We check whether there are isolated NaNs:
    ut.tprint('------Selecting all columns that are NaN')
    n_nans = np.sum(np.isnan(stack_of_orders),axis=0)#This is the total number of NaNs in each
    #column. Whenever the number of NaNs equals the length of a column, we ignore them:
    n_nans[n_nans==len(stack_of_orders)]=0
    if np.max(n_nans)>0:#If there are any columns which still have NaNs in them, we need to crash.
        ut.tprint(f"ERROR in CCF: Not all NaN values are purely in data columns. There are still "
        "isolated NaNs in the data. This could be due to the template having NaNs or poorly "
        "covering the data, or if entire exposures in the time-series are NaN (perhaps due to "
        "masking of outliers where the SNR is really variable and low). You can identify this "
        "in the masking GUI. PDB-ing you out of here so that you can debug.")
        pdb.set_trace()

    #So we checked that all NaNs were in whole columns. These columns have the following indices:
    nan_columns =  np.isnan(stack_of_orders[0])
    ut.tprint(f'------{np.sum(nan_columns)} columns identified to be ignored during CCF.')

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
        #step. For the KELT-9 demo data, that's 2.7MB, times 600 velocity steps = 1.6 GB. For
        #ESPRESSO data and larger velocity excursions, this could be a whole other story.
        #The orders of an ESPRESSO dataset with 40 exposures measure 420MB; i.e. 10.5 MB per
        #exposure. Very often, I put large RV excursions to accomdate large filter kernels, e.g.
        #800 or 1000km/s with 1km/s steps. That's 2000 steps.
        #2000 * 10.5 MB = 21 GB. My laptop does not have that kind of memory.
        #I could add a conserve-memory switch that carries out this computation in chunks of
        #small ranges of RV; computing a series of vertical swaths of the final CCF in RV intervals,
        #then h-stacking those. The number of RVs should then be limited to e.g. the number of
        #exposures, so that this casting only inflates memory by once the size of the data (which
        #is small compared to the amount of memory already being used).

        T[:,nan_columns] = 0.0#All NaNs are assumed to be in all-NaN columns. If that is not true,
        #the below nantest will fail.
        T_sums = np.sum(T,axis = 1)
        CCF = stack_of_orders @ T.T/T_sums#Here it the entire cross-correlation. Over all orders and
        #velocity steps. No forloops.
        CCF_E = CCF*0.0
        #If the errors were provided, we do the same to those:
        if list_of_errors is not None: CCF_E = np.sqrt(stack_of_errors2 @ (T.T/T_sums)**2)#This has been
        del(T)
        #mathematically proven: E^2 = e^2 * T^2


        #===THAT'S ALL. TEST INTEGRITY AND RETURN THE RESULT===
        try:
            nantest(CCF,'CCF in ccf.xcor()')#If anything went wrong with NaNs in the data, these
            #tests will fail because the matrix operation @ is non NaN-friendly.
            nantest(CCF_E,'CCF_E in ccf.xcor()')
        except:
            ut.tprint('ERROR in XCOR(): NaNs were detected in the CCF. This does not happen '
            'because the data has NaNs, because xcor() checks for this. Instead, this can happen '
            'if the template and the data do not overlap, maybe because of a wavelength-unit'
            'mismatch? When using Tayph under normal circumstances, this error should never be'
            'triggered, as abundant checks are performed when XCOR() is run in the normal '
            'Tayph workflow. Are you doing development, or using xcor() in a stand-alone '
            'setting? PDB-ing you out of here so that you can plot the input of xcor() and debug.')
            pdb.set_trace()
        return(CCF,CCF_E,T_sums)

    if parallel:#This here takes a lot of memory.
        ut.tprint(f'------Starting parallel CCF with {NT} templates on {NC} cores')
        list_of_CCFs, list_of_CCF_Es, list_of_T_sums = zip(*Parallel(n_jobs=NC)(delayed(do_xcor)(i)
        for i in range(NT)))
    else:
        ut.tprint(f'------Starting CCF on {NT} templates in sequence')
        list_of_CCFs, list_of_CCF_Es, list_of_T_sums = zip(*[do_xcor(i) for i in range(NT)])

    if list_of_errors != None:
        return(RV,list(list_of_CCFs),list(list_of_CCF_Es),list(list_of_T_sums))
    return(RV,list(list_of_CCFs),list(list_of_T_sums))












#BINARY MASK VERSION
#This is for a list of orders. Because I do a search_sort, I cannot just stitch the orders together
#like before. But I figured out the version of this that indeces and matrix-multiplies the whole
#cross-correlation in one go for each order. That is nearly unreadable, so I have left the
#non-vectorised, for-loopy version hidden behind the fast=False keyword, so that you can verify that
#the vectorised and the loopy version yield the same answer.

def mask_cor(list_of_wls,list_of_orders,list_of_wlm,list_of_fxm,drv,RVrange,list_of_errors=None,
parallel=False,fast=True,strict_edges=True,return_templates=False,zero_point = 0):
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
    import copy
    if parallel: from joblib import Parallel, delayed
    import matplotlib.pyplot as plt
    """
    This routine takes a combined dataset (in the form of lists of wl spaces,
    spectral orders and possible a matching list of errors on those spectal orders),
    as well as a list of spectral masks (wlm,fxm) to cross-correlate with, and the
    cross-correlation parameters (drv,RVrange). The code can be run in parallel if sufficient
    RAM is available.

    The CCF used is the Geneva-style weighted average (not the Pearson CCF) with a weighted line
    list (sometimes called a binary mask, and similar to the implementation of the
    cross-correlation performed by the HARPS and ESPRESSO pipelines). Therefore it measures true
    'average' planet lines, with relative flux on the y-axis of the CCF. The template should be a
    list of line positions and their weights, resembling scaled delta functions in discrete
    wavelength space. These lines are typically derived from some form of radiative transfer model.
    The CCF will be scale-invariant so their overall scaling does not matter.

    Instead of evaluating a template spectrum on the data, this implementation just searches for
    the nearest neighbour datapoints to each line included in the list, distributing the weight in
    the line over them depending on the sub-pixel distance between the points, and then integrating
    over all lines, radial velocity shifts and spectral orders.


    It returns the RV axis and lists of the resulting CCF, uncertainties and template integrals in
    a tuple, and optionally the spectral templates corresponding to the effective weighting
    functions and the lines actually included/counted in each order, if requested.

    Thanks to Brett Morris (bmorris3), this code implements a clever numpy broadcasting trick to
    instantly apply and interpolate the wavelength shifts of the model template onto
    the data grid in 2 dimensions. The matrix multiplication operator (originally
    recommended to me by Matteo Brogi) allows this 2D template matrix to be multiplied
    with a 2D spectral order. Combined with the fact that the calculation includes only spectral
    pixels surrounding the spectral lines, this can be done relatively much faster than the
    cross-correlation above.

    The main purpose of this way of computing cross-correlations is to provide realistic
    measurements of the average spectral lines in the planet spectrum (not measurements that are
    blurred by the width of the template); and to provide full control over what lines are included
    and how they are weighted. The purpose being accurate average line measurements has
    consequences for how data edges and masked regions (NaNs) are treated. First of all, the
    assumption is that every NaN is in an all-NaN column. In the standard cross-correlation
    work-flow, isolated NaNs are interpolated over (healed), after all.

    To avoid any edge effects, set the strict_edges keyword to True. In this case, lines that
    would cross an edge at any RV excursion will be ignored. The philosophy of ignoring edge-lines
    would generalise to masked regions as well: Any lines that cross NaN regions would have to be
    ignored. However for simplicity, and for avoiding chaotic cutting away of lines whenever a user
    defines a couple of columns in the data as NaN, we have currently imposed a hard crash when
    a mask is used where NaNs are anywhere *but at the edges of the order*. So NaN regions can be
    defined, as long as they are continuous to the order edge. What is included in each order is
    the spectral pixel from the first non-NaN column to the last non-NaN column. Any remaining NaNs
    in between will cause the code to abort.

    Setting strict_edges to False will cause all lines that enter into the data wavelength regime
    at any time to be included, introducing a slight dependence of the average measurement with
    time because the radial velocity excursion of the planet makes some of its lines cross the
    order edges, and these lines would be included at some radial velocities but not others,
    changing the meaning of the average. The places where there are NaN columns in the data are
    set to 0 in the template matrix. The NaN values themselves are then set to to an arbitrary
    value, since they will never weigh into the average by construction.


    Multiple templates are provided as lists of wavelength and weight arrays. These are passed as
    lists to enable parallel computation on multiple threads without having to re-instantiate the
    stacks of spectra each time a template has to be cross-correlated with. Like other places in
    Tayph where parellelisation is applied, this speeds things up greatly, in theory. However, the
    interpolation of the templates onto the data grid signifies a multiplication of the size of each
     template variable equal to the number of velocity steps times the size of the wavelength axis
    of the data. Multiplied by the number of templates, the total instantaneous memory load can be
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

    parallel : bool
        Optional, enabling parallel computation (potentially highly demanding on memory).

    strict_edges : bool
        Optional, including only lines that are inside the spectral range of the data at all
        requested radial velocity shifts. True by default.

    return_templates : bool
        Optional, to return the spectra weighting function that is effectively applied to each
        spectral column of each order, as well as the list of line positions included in the
        computation. Off by default.

    zero_point : float
        Optional, to indicate at what radial velocity shift the template and the line positions are
        returned if return_templates is on. Zero by default.

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

    if len(list_of_wls) != len(list_of_orders):
        raise ValueError(f'In ccf.mask_cor(): List of wls and list of orders have different length '
        f'({len(list_of_wls)} & {len(list_of_orders)}).')
    NT = len(list_of_fxm)
    lentest(list_of_wlm,NT,'list_of_wlm in ccf.mask_cor()')
    typetest(drv,[int,float],'drv in ccf.mask_cor')
    typetest(RVrange,float,'RVrange in ccf.mask_cor()',)
    postest(RVrange,'RVrange in ccf.mask_cor()')
    postest(drv,'drv in ccf.mask_cor()')
    nantest(drv,'drv in ccf.mask_cor()')
    nantest(RVrange,'RVrange in ccf.mask_cor()')
    drv = float(drv)
    N=len(list_of_wls)#Number of orders.

    if np.ndim(list_of_orders[0]) == 1.0:
        n_exp=1
    else:
        n_exp=len(list_of_orders[0][:,0])#Number of exposures.

    #===Then check that all orders indeed have n_exp exposures===
        for i in range(N):
            if len(list_of_orders[i][:,0]) != n_exp:
                raise ValueError(f'In mask_cor(): Not all orders have {n_exp} exposures.')




    #Definition of the velocity axis:
    c=const.c.to('km/s').value#In km/s
    RV= np.arange(-RVrange, RVrange+drv, drv, dtype=float)
    beta=1.0+RV/c#The doppler factor with which each wavelength is to be shifted.
    n_rv = len(RV)#Number of RV points, is used quite a bit later.

    #Allow the user to return weight functions and line lists at RV points other than RV=0:
    if zero_point == 0:
        zero_point = int((n_rv-1)/2)#Index of RV zero-point.
    else:
        zero_point = np.argmin(np.abs(RV-zero_point))



    #Copy the input to prevent backward propagation of modifications:
    list_of_orders = copy.deepcopy(list_of_orders)
    list_of_wls = copy.deepcopy(list_of_wls)
    if list_of_errors is not None:
        list_of_errors = copy.deepcopy(list_of_errors)







    #===========================================
    #NOW WE DEAL WITH MASKED COLUMNS, I.E. NANS:
    #===========================================
    # list_of_wls_clipped = []
    # list_of_orders_clipped = []
    # if list_of_errors is not None:
    #     list_of_errors_clipped = []
    to_pop = []
    for o in range(len(list_of_wls)):#Loop over orders.
        wl = list_of_wls[o]
        order = list_of_orders[o]
        if list_of_errors is not None:
            error = list_of_errors[o]
            #Check that the number of NaNs is the same in the orders as in the errors on the orders;
            #and that they are in the same place; meaning that if I add the errors to the orders,
            #the number of NaNs does not increase (NaN+value=NaN).
            if (np.sum(np.isnan(order)) != np.sum(np.isnan(error+order))) and (np.sum(np.isnan(
            order)) != np.sum(np.isnan(error))):
                raise ValueError(f"in CCF: The number of NaNs in order {o} and in its uncertainty "
                f"array are not equal ({np.sum(np.isnan(order))},{np.sum(np.isnan(error))}).")

        #We check whether there are isolated NaNs and crash if there are:
        n_nans = np.sum(np.isnan(order),axis=0)#This is the total number of NaNs in each column.
        n_nans[n_nans==len(order)]=0#Whenever the number of NaNs equals the length of a column, we
        #dont count them. The remainder should be zero NaNs.
        if np.max(n_nans)>0:#If there are any columns which still have NaNs, we crash.
            raise ValueError(f"in mask_cor: Not all NaN values in order {o} are in full-NaN "
            "data columns. There are still isolated NaNs in the data. Remove those.")

        #That means that all remaining nans are in columns. We clip the order edges
        nans=~np.isnan(np.sum(order,axis=0))#Everywhere where there is no NaN equals 1.0
        if np.max(nans)==1:#Only proceed if there are at least some non_nan columns.
            #Otherwise, the order will not be contributing to the CCF at all and all the rest
            #can be ignored.
            if np.min(nans)==0:#If there are indeed masked edge columns, we clip from the first to
            #the last non-zero is-this-not-a-NaN index. Else, we leave the order arrays as they are.
                list_of_orders[o]=order[:,np.nonzero(nans)[0][0]:np.nonzero(nans)[0][-1]+1]
                list_of_wls[o] = wl[np.nonzero(nans)[0][0]:np.nonzero(nans)[0][-1]+1]
                if list_of_errors is not None:
                    list_of_errors[o] = error[:,np.nonzero(nans)[0][0]:np.nonzero(nans)[0][-1]+1]

            #If clipping the order edges has not removed all the nans, there are still NaN columns
            #left in the middle of the orders. If strict_edges is set, this raises an error:
            if strict_edges and np.min(nans[np.nonzero(nans)[0][0]:np.nonzero(nans)[0][-1]+1])!=1:
                # ut.writefits('order_wl_with_nan_column.fits',wl)
                ut.writefits('order_with_nan_column.fits',order)
                raise ValueError(f"in mask_cor: Strict_edges is True but there are NaN columns in "
                f"order {o} that are not continuously connected to the order edge. This is not "
                f" allowed. Remove those. The order has been written to FITS.")
        else:#If there are no non_nan columns at all, we simply delete that order from the list.
            to_pop.append(o)

    to_pop.sort(reverse=True)
    for i in to_pop:
        list_of_orders.pop(i)
        list_of_wls.pop(i)
        if list_of_errors is not None:
            list_of_errors.pop(i)

    #Now we should have ended up with lists of orders, wavelengths and optionally errors, that are
    #clipped to exclude NaN-masked edges, that dont have isolated NaNs in them, and that have no
    #all-NaN orders in them. And if strict_edges was set, there are no more NaNs in columns in the
    #middle of the orders either.
    #Test that the latter statement is true:
    if strict_edges:
        for i in range(len(list_of_orders)):
            nantest(list_of_orders[i],f'list_of_orders in ccf.mask_cor() after rejecting NaNs.')
            lentest(list_of_wls,len(list_of_orders),'list_of_wls in ccf.mask_cor() after '
            'rejecting NaNs.')
            if list_of_errors is not None:
                nantest(list_of_errors[i],f'list_of_orders in ccf.mask_cor() after rejecting NaNs.')
                lentest(list_of_errors,len(list_of_orders),'list_of_errors in ccf.mask_cor() '
                'after rejecting NaNs.')


    #Proceed with the calculation.
    def do_template(i):
        """
        From here you will witness the final destruction of the Alliance and the end of your
        insigificant rebellion.
        """
        wlT = list_of_wlm[i]
        T = list_of_fxm[i]
        T_sum = np.array([0.0]*n_rv)#An array of floats.
        T_sum_error = np.array([0.0]*n_rv)#Store these separately because the w's need to be factored in.
        CCF = np.zeros((len(list_of_orders[0]),len(RV)))
        CCF_E = copy.deepcopy(CCF)#Remains all zero if list_of_errors == None.
        W = []#List of weight functions of each order.
        L = []#List of lines in each order.
        for o in range(len(list_of_wls)):#Loop over orders.
            wl = list_of_wls[o]
            order = list_of_orders[o]
            if list_of_errors is not None:
                error = list_of_errors[o]

            if return_templates: Wo = wl*0.0#Weight function in this order, to be filled in.
            if strict_edges == True:#Select only lines that are in the desired wavelength range
                #for all velocity shifts.
                sel_lines = (wlT>np.min(wl)/(1+np.min(RV)/c))&(wlT<np.max(wl)/(1+np.max(RV)/c))
            else:#Select all lines that are in the desired wavelength range at any RV shift:
                sel_lines = (wlT>np.min(wl)/(1+np.max(RV)/c))&(wlT<np.max(wl)/(1+np.min(RV)/c))

            if len(sel_lines) > 0:#And, only proceed if there are lines in this wl range.
                wlT_order = wlT[sel_lines]#The template lines contained in this order. Selected with
                #strict edges on or off.
                T_order   = T[sel_lines]# 1 x N (1 row of N lines)
                shifted_wlT = wlT_order * beta[:, np.newaxis]#Template line positions shifted to
                #each velocity beta. Measures nRV x N.
                indices = np.searchsorted(wl,shifted_wlT)#The indices to the right of each
                #target line. Every row in indices is a list of N spectral lines.
                #For large numbers of lines, this vectorised search is not faster than serial,
                #but we use the 2D output to cause total vectorised mayhem next.
                #To make the most meaningful measurement of the "average line depth" in the planet
                #spectrum, spectral lines should not be allowed to appear or disappear as a function
                #of radial velocity shift. That means: Lines should not be allowed to travel over
                #an edge; and lines should not be allowed to travel into NaN regions.
                #To implement this I have introduced the strict_edges keyword that discards all
                #lines that travel beyond an edge, given the RV excursion requested. In this case,
                #the array T_order (containing the line weights) is the same for each RV shift.
                #That means that the cross-correlation over all velocities can be expressed as a
                #matrix multiplication. Which is especially fast.
                #Alternatively, you might want to still do a cross-correlation, but allowing lines
                #to travel through NaN regions and edges. That will make the measured average
                #planetary spectral line (slightly) time-dependent, because whenever the planet
                #spectrum intersects with an edge or a NaN region, that line cannot be counted
                #towards the weighted average, and therefore the sample over which the average is
                #taken, changes. However, there will be more lines available in this approach,
                #meaning that detection SNR's could be higher.
                #To deal with lines that go over the edge.
                #The template, i.e. w, and T need to be zero there. There are two cases:
                #Lines that go over the left edge result in indices=0. Lines that go over the right
                #edge result in indices=len(wl). Both of these give indexing errors below when
                #wl[indices] and wl[indices-1] are requested.
                #I therefore say indices[indices==0] = -1 and indices[indices==len(wl)] = -1.
                #That places all lines on the edge in the last column of order.
                #Then I need to set the w of those lines (and only those lines) to 0, as:
                #w[indices==-1] = 0.0
                #then make a new variable for (w-1) and also set that to zero, and then adjust
                #T_sum by using a 2D version of T_order, corresponding to indices and w.

                if fast:#Now witness the power of this fully armed and operational battle station!
                    if not strict_edges:
                        indices[indices == len(wl)] = -1
                        indices[indices == 0] = -1
                        w = (wl[indices]-shifted_wlT)/(wl[indices] - wl[indices-1])
                        w_inv = 1-w
                        w[indices==-1]=0.0
                        w_inv[indices==-1]=0.0
                        T_order_matrix = np.tile(T_order,(n_rv,1))
                        T_order_matrix[indices==-1]=0.0
                        T_sum += np.sum(T_order_matrix,axis=1)
                        CCF += np.sum((order[:,indices]*w_inv+order[:,indices-1]*w)*T_order_matrix,axis=2)
                        if list_of_errors is not None:
                            CCF_E += np.sum((error[:,indices]**2 *w_inv**2 + error[:,indices-1]**2 * w**2)*T_order_matrix**2,axis=2)
                            # T_sum_error+=np.sum(T_order_matrix)**2
                        if return_templates:
                            Wo[indices[zero_point]]+=(1-w[zero_point])*T_order_matrix[zero_point]
                            Wo[indices[zero_point]-1]+=w[zero_point]*T_order_matrix[zero_point]
                            W.append(Wo)#The template at RV=0 of each order is appended.
                            L.append(shifted_wlT[zero_point])
                    else:#Fire at will, commander!
                        w = (wl[indices]-shifted_wlT)/(wl[indices] - wl[indices-1])
                        CCF += (order[:,indices]*(1-w)+order[:,indices-1]*w)@T_order
                        if list_of_errors is not None:
                            CCF_E += (error[:,indices]**2 *(1-w)**2 +
                            error[:,indices-1]**2 *w**2)@T_order**2
                            # T_sum_error+=np.sum(((1-w)**2+w**2) @ T_order**2)
                        T_sum+=np.sum(T_order)
                        if return_templates:
                            Wo[indices[zero_point]]+=(1-w[zero_point])*T_order
                            Wo[indices[zero_point]-1]+=w[zero_point]*T_order
                            W.append(Wo)#The template at RV=0 of each order is appended.
                            L.append(shifted_wlT[zero_point])

                else:#Single reactor ignition:
                    for j in range(len(beta)):#==len(indices), because
                        #indices is a matrix of len(RV)*N. So for each value of beta we calculate
                        #the cross correlation function.
                        for n,i in enumerate(indices[j]):#And we loop over all lines in the template
                            #in this order to build up the value of the CCF at this velocity shift.
                            if i > 0 and i < len(wl):#Select only lines that still lie within the
                                #wavelength range. Some lines may be shifted over the edge because
                                #of beta.

                                #We selects 2 elements: wl[i-1] and wl[i].
                                #Because indices[j] measures the indices of wl to the right of each
                                #target line, we also select wl[i-1] to obtain the indices to the
                                #left and to the right of each line. We then weigh each of these
                                #based on whether the position is closer to the left or right:
                                w = (wl[i]-shifted_wlT[j,n])/(wl[i] - wl[i-1])#This number is small
                                #if wl[i]=wlT[n], meaning that w measures how close the line is to
                                #the index to the left. So wl[i] gets weight w-1, and wl[i-1] gets
                                #weight w.

                                #Now we start filling in the CCF. Upon the first pass, CCF was
                                #simply defined as an empty matrix. We now fill it in column
                                #by column.
                                #Because each row in the CCF corresponds to a row in the order,
                                #column, by column filling is possible: w and T are the same for
                                #each row in the order, yielding a column of values for each w and
                                #T.
                                CCF[:,j] += (order[:,i]*(1-w) + order[:,i-1]*w)*T_order[n]
                                if list_of_errors is not None:
                                    CCF_E[:,j] += (error[:,i]**2 *(1-w)**2 + error[:,i-1]**2 * w**2 )*T_order[n]**2
                                    T_sum_error[j] += T_order[n]
                                T_sum[j] += T_order[n]
                                if return_templates and j == zero_point:
                                    Wo[i]+=(1-w)*T_order[n]#Activate this to plot the "template" at this value of beta.
                                    Wo[i-1]+=w*T_order[n]
                        if return_templates and j == zero_point:
                            W.append(Wo)
                            L.append(wlT_order)
        return(CCF/T_sum,np.sqrt(CCF_E)/T_sum,T_sum,W,L)

    if parallel:#This here takes a lot of memory.
        list_of_CCFs, list_of_CCF_Es, list_of_T_sums,list_of_weights,list_of_lines = zip(*Parallel(
        n_jobs=NT)(delayed(do_template)(i) for i in range(NT)))
    else:
        list_of_CCFs, list_of_CCF_Es, list_of_T_sums,list_of_weights,list_of_lines = zip(
        *[do_template(i) for i in range(NT)])



    #Return errors on CCF if errors on spectrum are given; and return the weight function (at RV=0)
    #for each spectral order and each template, if requested.
    if list_of_errors != None:
        if return_templates:
            return(RV,list(list_of_CCFs),list(list_of_CCF_Es),list(list_of_T_sums),list(list_of_weights),list(list_of_lines))
        else:
            return(RV,list(list_of_CCFs),list(list_of_CCF_Es),list(list_of_T_sums))
    else:
        if return_templates:
            return(RV,list(list_of_CCFs),list(list_of_T_sums),list(list_of_weights),list(list_of_lines))
        else:
            return(RV,list(list_of_CCFs),list(list_of_T_sums))




def clean_ccf(rv,ccf,ccf_e,dp,intransit):
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
        ut.tprint('------WARNING in Cleaning: The data contains only in-transit exposures.')
        ut.tprint('------The mean ccf is taken over the entire time-series.')
        meanccf=np.nanmean(ccf_n,axis=0)
        meanccf_e=1.0/len(transit)*np.sqrt(np.nansum(ccf_ne**2.0,axis=0))#I validated that this is approximately equal
        #to sqrt(N)*ccf_ne, where N is the number of out-of-transit exposures.
    elif np.sum(transit==1) <= 0.25*len(transit) and intransit==True:
        ut.tprint('------WARNING in Cleaning: The data contains very few (<25%) out of transit exposures.')
        ut.tprint('------The mean ccf is taken over the entire time-series.')
        meanccf=np.nanmean(ccf_n,axis=0)
        meanccf_e=1.0/len(transit)*np.sqrt(np.nansum(ccf_ne**2.0,axis=0))#I validated that this is approximately equal
        #to sqrt(N)*ccf_ne, where N is the number of out-of-transit exposures.
    elif np.min(transit) == 1.0 and intransit==True:
        ut.tprint('------WARNING in Cleaning: The data is not predicted to contain in-transit exposures.')
        ut.tprint(f'------If you expect to be dealing with transit-data, please check the ephemeris '
        f'at {dp}. If you are dealing with non-transit data, set the transit keyword in the run-'
        'file to False.')
        ut.tprint('------The mean ccf is taken over the entire time-series.')
        meanccf=np.nanmean(ccf_n,axis=0)
        meanccf_e=1.0/len(transit)*np.sqrt(np.nansum(ccf_ne**2.0,axis=0))#I validated that this is approximately equal
        #to sqrt(N)*ccf_ne, where N is the number of out-of-transit exposures.
    elif intransit==False:
        ut.tprint('------The mean ccf is taken over the entire time-series.')
        meanccf=np.nanmean(ccf_n,axis=0)
        meanccf_e=1.0/len(transit)*np.sqrt(np.nansum(ccf_ne**2.0,axis=0))
    else:
        meanccf=np.nanmean(ccf_n[transit == 1.0,:],axis=0)
        meanccf_e=1.0/np.sum(transit==1)*np.sqrt(np.nansum(ccf_ne[transit == 1.0,:]**2.0,axis=0))#I validated that this is approximately equal
        ut.tprint('------The mean ccf is taken over the in-transit exposures of the time-series.')
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

def construct_KpVsys(rv,ccf,ccf_e,dp,kprange=[0,300],dkp=1.0,parallel=True,transit=True):
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
    import copy
    from joblib import Parallel, delayed
    import scipy.ndimage.interpolation as scint

    Kp = np.arange(kprange[0], kprange[1]+dkp, dkp, dtype=float) #fun.findgen((kprange[1]-kprange[0])/dkp+1)*dkp+kprange[0]
    n_exp = np.shape(ccf)[0]
    KpVsys = np.zeros((len(Kp),len(rv)))
    KpVsys_e = np.zeros((len(Kp),len(rv)))
    LC = sp.transit(dp)-1.0
    dv = rv[1]-rv[0]
    n_exp=len(ccf[:,0])#Number of exposures.
    if transit:
        LC /= np.nansum(LC)
        transitblock = fun.rebinreform(LC,len(rv)).T
    else:
        transitblock = fun.rebinreform(LC,len(rv)).T * 0.0 + 1.0/len(LC)



    # pdb.set_trace()
    ccf_copy = copy.deepcopy(ccf)
    ccf_e_copy = copy.deepcopy(ccf_e)

    def Kp_parallel(i):
        dRV = sp.RV(dp,vorb=i)*(-1.0)
        ccf_shifted = ccf_copy*0.0
        ccf_e_shifted = ccf_e_copy*0.0
        for i in range(n_exp):
            ccf_shifted[i,:] = scint.shift(ccf_copy[i],dRV[i]/dv,mode='nearest',order=1)
            ccf_e_shifted[i,:] = scint.shift(ccf_e_copy[i],dRV[i]/dv,mode='nearest',order=1)
        # ccf_shifted = copy.deepcopy(ccf)*0.0#shift_ccf(rv,ccf,dRV)
        # ccf_e_shifted = copy.deepcopy(ccf_e)*0.0#shift_ccf(rv,ccf_e,dRV)
        return (np.nansum(transitblock * ccf_shifted,axis=0), (np.nansum((transitblock*ccf_e_shifted)**2.0,axis=0))**0.5)

    if parallel:
        KpVsys, KpVsys_e = zip(*Parallel(n_jobs=-1)(delayed(Kp_parallel)(i) for i in Kp))
    else:
        KpVsys, KpVsys_e = zip(*[Kp_parallel(i) for i in Kp])

    return(np.array(Kp),np.array(KpVsys),np.array(KpVsys_e))

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
