__all__ = [
    "build_template",
    "get_model"
]





def build_template(templatename,binsize=1.0,maxfrac=0.01,mode='top',resolution=0.0,twopass=False,template_library='models/library'):
    """This routine reads a specified model from the library and turns it into a
    cross-correlation template by subtracting the top-envelope (or bottom envelope)"""

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
    ut.check_path(template_library,exists=True)
    template_library=Path(template_library)

    c=const.c.to('km/s').value

    if mode not in ['top','bottom']:
        raise Exception(f'RuntimeError in build_template: Mode should be set to "top" or "bottom" ({mode}).')
    wlt,fxt=get_model(templatename,library=template_library)

    if wlt[-1] <= wlt[0]:#Reverse the wl axis if its sorted the wrong way.
        wlt=np.flipud(wlt)
        fxt=np.flipud(fxt)

    wle,fxe=ops.envelope(wlt,fxt-np.median(fxt),binsize,selfrac=maxfrac,mode=mode)#These are binpoints of the top-envelope.
    #The median of fxm is first removed to decrease numerical errors, because the spectrum may
    #have values that are large (~1.0) while the variations are small (~1e-5).
    e_i = interpolate.interp1d(wle,fxe,fill_value='extrapolate')
    envelope=e_i(wlt)
    # plt.plot(wlt,fxt-np.median(fxt))
    # plt.plot(wlt,envelope,'.')
    # plt.show()
    T = fxt-np.median(fxt)-envelope
    absT = np.abs(T)
    T[(absT < 1e-4 * np.max(absT))] = 0.0 #This is now continuum-subtracted and binary-like.
    #Any values that are small are taken out.
    #This therefore assumes that the model has lines that are deep compared to the numerical
    #error of envelope subtraction (!).
    # plt.plot(wlt,T)
    # plt.show()
    if resolution !=0.0:
        dRV = c/resolution
        print('------Blurring template to resolution fo data (%s, %s km/s)' % (round(resolution,0),round(dRV,2)))
        wlt_cv,T_cv,vstep=ops.constant_velocity_wl_grid(wlt,T,oversampling=2.0)
        print('---------v_step is %s km/s' % vstep)
        print('---------So the resolution blurkernel has an avg width of %s px.' % (dRV/vstep))
        T_b=ops.smooth(T_cv,dRV/vstep,mode='gaussian')
        wlt = wlt_cv*1.0
        T = T_b*1.0
    return(wlt,T)


def get_model(name,library='models/library',root='models'):
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
    """

    from tayph.vartests import typetest,dimtest
    from tayph.util import check_path
    from astropy.io import fits
    from pathlib import Path
    import errno
    import os
    typetest(name,str,'name in mod.get_model()')
    check_path(library,exists=True)

    check_path(root,exists=True)
    root = Path(root)
    library = Path(library)
    #First open the library file.

    f = open(library, 'r')

    x = f.read().splitlines()#Read everything into a big string array.
    f.close()
    n_lines=len(x)#Number of models in the library.
    models={}#This will contain the model names.

    for i in range(0,n_lines):
        line=x[i].split()
        value=(line[1])
        models[line[0]] = value

    try:
        modelpath=root/models[name]
    except KeyError:
        raise KeyError(f'Model {name} is not present in library at {str(library)}') from None

    try:
        modelarray=fits.getdata(modelpath)#Two-dimensional array with wl on the first row and flux on the second row.
    except FileNotFoundError:
        raise FileNotFoundError(f'Model file {modelpath} from library {str(library)} does not exist.')
    dimtest(modelarray,[2,0])
    return(modelarray[0,:],modelarray[1,:])
