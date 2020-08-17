#The following are not strictly molecfit-specific
def write_telluric_transmission_to_file(wls,T,outpath):
    """This saves a list of wl arrays and a corresponding list of transmission-spectra
    to a pickle file, to be read by the function below."""
    import pickle
    import tayph.util as ut
    ut.check_path(outpath)
    print(f'------Saving teluric transmission to {outpath}')
    with open(outpath, 'wb') as f: pickle.dump((wls,T),f)

def read_telluric_transmission_from_file(inpath):
    import pickle
    import tayph.util as ut
    print(f'------Reading teluric transmission from {inpath}')
    ut.check_path(inpath,exists=True)
    pickle_in = open(inpath,"rb")
    return(pickle.load(pickle_in))#This is a tuple that can be unpacked into 2 elements.


def apply_telluric_correction(inpath,list_of_wls,list_of_orders,list_of_sigmas):
    """
    This applies a set of telluric spectra (computed by molecfit) for each exposure
    in our time series that were written to a pickle file by write_telluric_transmission_to_file.

    List of errors are provided to propagate telluric correction into the error array as well.

    Parameters
    ----------
    inpath : str, path like
        The path to the pickled transmission spectra.

    list_of_wls : list
        List of wavelength axes.

    list_of_orders :
        List of 2D spectral orders, matching to the wavelength axes in dimensions and in number.

    list_of_isgmas :
        List of 2D error matrices, matching dimensions and number of list_of_orders.

    Returns
    -------
    list_of_orders_corrected : list
        List of 2D spectral orders, telluric corrected.

    list_of_sigmas_corrected : list
        List of 2D error matrices, telluric corrected.

    """
    import scipy.interpolate as interp
    import numpy as np
    import tayph.util as ut
    import tayph.functions as fun
    from tayph.vartests import dimtest,postest,typetest,nantest
    wlT,fxT = read_telluric_transmission_from_file(inpath)
    typetest(list_of_wls,list,'list_of_wls in apply_telluric_correction()')
    typetest(list_of_orders,list,'list_of_orders in apply_telluric_correction()')
    typetest(list_of_sigmas,list,'list_of_sigmas in apply_telluric_correction()')
    typetest(wlT,list,'list of telluric wave-axes in apply_telluric_correction()')
    typetest(fxT,list,'list of telluric transmission spectra in apply_telluric_correction()')


    No = len(list_of_wls)
    x = fun.findgen(No)

    if No != len(list_of_orders):
        raise Exception('Runtime error in telluric correction: List of data wls and List of orders do not have the same length.')

    Nexp = len(wlT)

    if Nexp != len(fxT):
        raise Exception('Runtime error in telluric correction: List of telluric wls and telluric spectra read from file do not have the same length.')

    if Nexp !=len(list_of_orders[0]):
        raise Exception(f'Runtime error in telluric correction: List of telluric spectra and data spectra read from file do not have the same length ({Nexp} vs {len(list_of_orders[0])}).')
    list_of_orders_cor = []
    list_of_sigmas_cor = []
    # ut.save_stack('test.fits',list_of_orders)
    # pdb.set_trace()

    for i in range(No):#Do the correction order by order.
        order = list_of_orders[i]
        order_cor = order*0.0
        error  = list_of_sigmas[i]
        error_cor = error*0.0
        wl = list_of_wls[i]
        dimtest(order,[0,len(wl)],f'order {i}/{No} in apply_telluric_correction()')
        dimtest(error,np.shape(order),f'errors {i}/{No} in apply_telluric_correction()')

        for j in range(Nexp):
            T_i = interp.interp1d(wlT[j],fxT[j],fill_value="extrapolate")(wl)
            postest(T_i,f'T-spec of exposure {j} in apply_telluric_correction()')
            nantest(T_i,f'T-spec of exposure {j} in apply_telluric_correction()')
            order_cor[j]=order[j]/T_i
            error_cor[j]=error[j]/T_i#I checked that this works because the SNR before and after telluric correction is identical.
        list_of_orders_cor.append(order_cor)
        list_of_sigmas_cor.append(error_cor)
        ut.statusbar(i,x)
    return(list_of_orders_cor,list_of_sigmas_cor)
