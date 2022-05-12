__all__ = [
    "get_telluric",
    "build_template",
    "get_model",
    "inject_model",
    "inject_model_into_order"
]



def get_telluric():
    import tayph.util as ut
    from astropy.utils.data import download_file
    from pathlib import Path
    from astropy.io import fits
    #Load the telluric spectrum from my Google Drive:
    telluric_link = ('https://drive.google.com/uc?export=download&id=1yAtoLwI3h9nvZK0IpuhLvIp'
        'Nc_1kjxha')

    telpath = Path(download_file(telluric_link,cache=True))
    if telpath.exists() == False:
        ut.tprint(f'------Download to {telpath} failed. Is the cache damaged? Rerunning with '
            'cache="update".')
        telpath = Path(download_file(telluric_link,cache='update'))#If the cache is damaged.
        if telpath.exists() == False:#If it still doesn't exist...
            raise Exception(f'{telpath} does not exist after attempted repair of the astropy '
        'cache. To continue, run read_e2ds() with measure_RV=False, and troubleshoot your '
        'astropy cache.')
    else:
        ut.tprint(f'------Download to {telpath} was successful. Proceeding.')
    ttt=fits.getdata(telpath)
    return(ttt[0],ttt[1])





def build_template(templatename,binsize=1.0,maxfrac=0.01,mode='top',resolution=0.0,c_subtract=True,
twopass=False,template_library='models/library',verbose=False):
    """This routine reads a specified model from the library and turns it into a
    cross-correlation template by subtracting the top-envelope (or bottom envelope),
    if c_subtract is set to True. Also blurs the template to the resolution of the data, if it is
    not a binary line mask. Returns the wavelength axis and flux axis of the template, and whether
    the template is a binary mask (True) or a spectrum (False)."""

    import tayph.util as ut
    from tayph.vartests import typetest,postest,notnegativetest
    import numpy as np
    import tayph.operations as ops
    import astropy.constants as const
    from astropy.io import fits
    from matplotlib import pyplot as plt
    from scipy import interpolate
    from pathlib import Path
    typetest(templatename,str,'templatename mod.build_template()')
    typetest(binsize,[int,float],'binsize mod.build_template()')
    typetest(maxfrac,[int,float],'maxfrac mod.build_template()')
    typetest(mode,str,'mode mod.build_template()',)
    typetest(resolution,[int,float],'resolution in mod.build_template()')
    typetest(twopass,bool,'twopass in mod.build_template()')




    binsize=float(binsize)
    maxfrac=float(maxfrac)
    resolution=float(resolution)
    postest(binsize,'binsize mod.build_template()')
    postest(maxfrac,'maxfrac mod.build_template()')
    notnegativetest(resolution,'resolution in mod.build_template()')
    template_library=ut.check_path(template_library,exists=True)

    c=const.c.to('km/s').value

    if mode not in ['top','bottom']:
        raise Exception(f'RuntimeError in build_template: Mode should be set to "top" or "bottom" ({mode}).')
    wlt,fxt=get_model(templatename,library=template_library)

    if wlt[-1] <= wlt[0]:#Reverse the wl axis if its sorted the wrong way.
        wlt=np.flipud(wlt)
        fxt=np.flipud(fxt)

    if get_model(templatename,library=template_library,is_binary=True):#Bypass all template-specific operations.
        return(wlt,fxt,True)

    if c_subtract == True:
        wle,fxe=ops.envelope(wlt,fxt-np.median(fxt),binsize,selfrac=maxfrac,mode=mode)#These are binpoints of the top-envelope.
        #The median of fxm is first removed to decrease numerical errors, because the spectrum may
        #have values that are large (~1.0) while the variations are small (~1e-5).
        e_i = interpolate.interp1d(wle,fxe,fill_value='extrapolate')
        envelope=e_i(wlt)
        T = fxt-np.median(fxt)-envelope
        absT = np.abs(T)
        T[(absT < 1e-4 * np.max(absT))] = 0.0 #This is now continuum-subtracted and binary-like.
        #Any values that are small are taken out.
        #This therefore assumes that the model has lines that are deep compared to the numerical
        #error of envelope subtraction (!).
    else:
        T = fxt*1.0

    if resolution !=0.0:
        dRV = c/resolution
        if verbose:
            ut.tprint(f'------Blurring template to resolution of data ({round(resolution,0)}, {round(dRV,2)} km/s)')
        wlt_cv,T_cv,vstep=ops.constant_velocity_wl_grid(wlt,T,oversampling=2.0)
        if verbose:
            ut.tprint(f'---------v_step is {np.round(vstep,3)} km/s')
            ut.tprint(f'---------So the resolution blurkernel has an avg width of {np.round(dRV/vstep,3)} px.')
        T_b=ops.smooth(T_cv,dRV/vstep,mode='gaussian')
        wlt = wlt_cv*1.0
        T = T_b*1.0
    return(wlt,T,False)


def get_model(name,library='models/library',root='models',is_binary=False):
    """This program queries a model from a library file, with predefined models
    for use in model injection, cross-correlation and plotting. These models have
    a standard format. They are 2 rows by N points, where N corresponds to the
    number of wavelength samples. The first row contains the wavelengths, the
    second the associated flux values. The library file has two columns that are
    formatted as follows:

    modelname  modelpath
    modelname  modelpath
    modelname  modelpath

    modelpath starts in the models/ subdirectory.
    Set the root variable if a location is required other than the ./models directory.

    Example call:
    wlm,fxm = get_model('WASP-121_TiO',library='models/testlib')

    Set the is_binary keyword to ask whether the file is a binary .dat file or not, and return a
    single boolean. Used to switch between cross-correlation and template construction modes.
    """

    from tayph.vartests import typetest,dimtest
    from tayph.util import check_path
    from astropy.io import fits
    from pathlib import Path
    import errno
    import os
    import numpy as np
    import pdb
    typetest(name,str,'name in mod.get_model()')
    library=check_path(library,exists=True)



    #First open the library file.
    f = open(library, 'r')
    x = f.read().splitlines()#Read everything into a big string array.
    f.close()
    n_lines=len(x)#Number of models in the library.
    models={}#This will contain the model names.

    for i in range(0,n_lines):
        line=x[i].split()
        if len(line)>1:
            value=(line[1])
            models[line[0]] = value

    try:
        modelpath=Path(models[name])
    except KeyError:
        raise KeyError(f'Model {name} is not present in library at {str(library)}') from None
    if str(modelpath)[0]!='/':#Test if this is an absolute path.
        root=check_path(root,exists=True)
        modelpath=root/modelpath
    if modelpath.exists():
        if modelpath.suffix == '.fits':
            if is_binary: return(False)
            modelarray=fits.getdata(modelpath)#Two-dimensional array with wl on the first row and
            #spectral flux on the second row.
        elif modelpath.suffix == '.dat':
            if is_binary: return(True)
            modelarray = np.loadtxt(modelpath).T#Two-dimensional array with wavelength positions of
            #spectral lines on the first column and weights on the second column (transposed).
        else:
            raise RunTimeError(f'Model file {modelpath} from library {str(library)} is required to '
            'have extension .fits or .dat.')
    else:
        raise FileNotFoundError(f'Model file {modelpath} from library {str(library)} does not '
        'exist.')
    dimtest(modelarray,[2,0],varname=f'model {str(modelpath)}')
    return(modelarray[0,:],modelarray[1,:])


def inject_model(list_of_wls,list_of_orders,dp,modelname,model_library='library/models',
intransit=True):
    """This function takes a list of spectral orders and injects a model with library
    identifier modelname, and system parameters as defined in dp. The model is blurred taking into
    account spectral resolution and rotation broadening (with an LSF as per Brogi et al.) and
    finite-exposure broadening (with a box LSF).

    It returns a copy of the list of orders with the model injected."""

    import tayph.util as ut
    import tayph.system_parameters as sp
    import tayph.models
    import astropy.constants as const
    import numpy as np
    import scipy
    import tayph.operations as ops
    from tayph.vartests import typetest, dimtest
    import pdb
    import tayph.functions as fun
    import copy
    import matplotlib.pyplot as plt

    # dimtest(order,[0,len(wld)])
    dp=ut.check_path(dp)
    typetest(modelname,str,'modelname in models.inject_model()')
    typetest(model_library,str,'model_library in models.inject_model()')

    c=const.c.to('km/s').value#In km/s
    Rd=sp.paramget('resolution',dp)
    planet_radius=sp.paramget('Rp',dp)
    inclination=sp.paramget('inclination',dp)
    P=sp.paramget('P',dp)
    transit=sp.transit(dp)
    n_exp=len(transit)
    vsys=sp.paramget('vsys',dp)
    rv=sp.RV(dp)+vsys
    dRV=sp.dRV(dp)
    phi=sp.phase(dp)
    dimtest(transit,[n_exp])
    dimtest(rv,[n_exp])
    dimtest(phi,[n_exp])
    dimtest(dRV,[n_exp])


    if intransit:
        mask=(transit-1.0)/(np.min(transit-1.0))
    else:#Emission
        mask=transit*0.0+1.0




    wlm,fxm=get_model(modelname,library=model_library)
    if wlm[-1] <= wlm[0]:#Reverse the wl axis if its sorted the wrong way.
        wlm=np.flipud(wlm)
        fxm=np.flipud(fxm)

    #With the model and the revelant parameters in hand, now only select that
    #part of the model that covers the wavelengths of the order provided.
    #A larger wavelength range would take much extra time because the convolution
    #is a slow operation.


    N_orders = len(list_of_wls)
    if N_orders != len(list_of_orders):
        raise RuntimeError(f'in models.inject_model: List_of_wls and list_of_orders are not of the '
        f'same length ({N_orders} vs {len(list_of_orders)})')



    wl_mins = []
    wl_maxs = []
    for wlo in list_of_wls:
        wl_mins.append(np.min(wlo))
        wl_maxs.append(np.max(wlo))

    if np.min(wlm) > np.min(wl_mins) or np.max(wlm) < np.max(wl_maxs):
        raise RuntimeError('in model injection: Data grid falls (partly) outside of model range. '
        'Use a model with a wavelength range that encapsulates the data fully.')

    modelsel=[(wlm >= np.min(wl_mins)-1.0) & (wlm <= np.max(wl_maxs)+1.0)]

    wlm=wlm[tuple(modelsel)]
    fxm=fxm[tuple(modelsel)]

    if np.median(fxm) < 0.5:
        print('WARNING in model injection: The median value of the injection model over the '
        'wavelength range of the data is less than 0.5. This model is probably not suitable '
        'for injection. This could have happened if your injection model is accidentally '
        'continuum subtracted (e.g. around zero). Please double-check your injection model and '
        'make sure that it has a continuum.')

    blurtrigger = 0
    if np.median(fxm[fun.selmax(fxm,0.05)]) > 1.01:
        ut.tprint('WARNING in model injection: The median of the highest 5% of values in the model '
        'that is to be injected is higher than 1.01; meaning that this model would add over 1% of '
        'the stellar flux at many wavelengths. In normal use-cases this is unphysical. This could '
        'have happened if you incorrectly continuum-normalised your injection model. Please double '
        f'check the model file (model name "{modelname}" in libraryfile {model_library}).')
        blurtrigger = 1
    if np.median(fxm[fun.selmax(fxm,0.99,s=0.99)]) < 0.8:
        ut.tprint('WARNING in model injection: The median of the lowest 1% of values in the model '
        'that is to be injected is less than 0.8; meaning that this model would absorb over 20% of '
        'the stellar flux at some wavelengths. In normal use-cases this is unphysical. This could '
        'have happened if you incorrectly continuum-normalised your injection model. Please double '
        f'check the model file (model name "{modelname}" in libraryfile {model_library}).')
        blurtrigger = 1


    fxm_b=ops.blur_rotate(wlm,fxm,c/Rd,planet_radius,P,inclination)#Only do this once per dataset.

    if blurtrigger == 0 and (np.median(fxm[fun.selmax(fxm,0.05)]) > 1.01 or
    np.median(fxm[fun.selmax(fxm,0.99,s=0.99)]) < 0.8):#acts as fun.selmin (if it existed).
        ut.tprint('ERROR in model injection: The injection model after blurring has attained '
        'unphysical values, even though everything was fine before blurring. This implies that '
        'something really bad has happened during rotation-blurring. Have you provided correct '
        'values for the spectral resolution, the planet radius, the period, the inclination in '
        'the config file? Has the speed of light c changed? Has ops.blur_rotate been tampered '
        'with? ')

    oversampling = 2.5
    wlm_cv,fxm_bcv,vstep=ops.constant_velocity_wl_grid(wlm,fxm_b,oversampling=oversampling)


    if np.min(dRV) < c/Rd/10.0:

        dRV_min = c/Rd/10.0#If the minimum dRV is less than 10% of the spectral
        #resolution, we introduce a lower limit to when we are going to blur, because the effect
        #becomes insignificant.
    else:
        dRV_min = np.min(dRV)

    if dRV_min/vstep < 3: #Meaning, if the smoothing is going to be undersampled by this choice
        #in v_step, it means that the oversampling parameter in ops.constant_velocity_wl_grid was
        #not high enough. Then we adjust it. I choose a value of 3 here to be safe, even though
        #ops.smooth below accepts values as low as 2.
        oversampling_new = 3.0/(dRV_min/vstep)*oversampling#scale up the oversampling factor.
        wlm_cv,fxm_bcv,vstep=ops.constant_velocity_wl_grid(wlm,fxm_b,oversampling=oversampling_new)






    list_of_orders_injected=copy.deepcopy(list_of_orders)

    for i in range(n_exp):
        if dRV[i] >= c/Rd/10.0:
            fxm_b2=ops.smooth(fxm_bcv,dRV[i]/vstep,mode='box')
        else:
            fxm_b2=copy.deepcopy(fxm_bcv)
        shift=1.0+rv[i]/c
        fxm_i=scipy.interpolate.interp1d(wlm_cv*shift,fxm_b2,fill_value=1.0,bounds_error=False) #This is a class that can be called.
        #Fill_value = 1 because if the model does not fully cover the order, it will be padded with 1.0s,
        #assuming that we are dealing with a model that is in transmission.
        for j in range(len(list_of_orders)):
            list_of_orders_injected[j][i]*=(1.0+mask[i]*(fxm_i(list_of_wls[j])-1.0))#This assumes
            #that the model is in transit radii. This can definitely be vectorised!

        #These are for checking that the broadening worked as expected:
        # injection_total[i,:]= scipy.interpolate.interp1d(wlm_cv*shift,fxm_b2)(wld)
        # injection_rot_only[i,:]=scipy.interpolate.interp1d(wlm*shift,fxm_b)(wld)
        # injection_pure[i,:]=scipy.interpolate.interp1d(wlm*shift,fxm)(wld)

    # ut.save_stack('test.fits',[injection_pure,injection_rot_only,injection_total])
    # pdb.set_trace()
    # ut.writefits('test.fits',injection)
    return(list_of_orders_injected)
