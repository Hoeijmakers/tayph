#This package contains high-level wrappers for running the entire sequence.

__all__ = [
    'make_project_folder',
    'start_run',
    'run_instance',
    'read_e2ds',
    'molecfit',
    'check_molecfit'
]


def make_project_folder(pwd='.'):
    from pathlib import Path
    import os
    import os.path
    import sys

    pwd = Path(pwd)
    if not os.listdir(pwd):
        root = pwd
    else:
        print("---pwd is not empty. Placing project in subfolder called cross_correlation.")
        root = Path('cross_correlation')
        try:
            root.mkdir(exist_ok=False)
        except FileExistsError:
            print('cross_correlation subfolder already exists. Please provide an empty project '
            'folder to start.')
            sys.exit()


    dirs = ['data','models','output']
    for d in dirs:
        (root/d).mkdir(exist_ok=True)
    (root/'data'/'KELT-9'/'night1').mkdir(parents=True,exist_ok=True)




def start_run(configfile,parallel=True,xcor_parallel=False,dp='',debug=False):
    """
    This is the main command-line initializer of the cross-correlation routine provided by Tayph.
    It parses a configuration file located at configfile which should contain predefined keywords.
    These keywords are passed on to the run_instance routine, which executes the analysis cascade.
    A call to this function is called a "run" of Tayph. A run has this configuration file as input,
    as well as a dataset, a (list of) cross-correlation template(s) and a (list of) models for
    injection purposes.

    Various routines in the main analysis cascade have been parallelised as of May 2021,
    allowing for a great speed up on systems that support multi-threading. In case simple
    parallelisation via the joblib package is not possible on your system, parallel computation
    can be switched off by setting parallel=False.

    Parallelisation of the cross-correlation function is handled separately, because it is highly
    demanding on the available memory, with each template doppler-shifted to all radial velocity
    samples having to be loaded into memory at once. Memory usage scales with the number of
    templates, the number of spectral pixels per template, and the number of cross-correlation steps
    ( = RVrange / dv).

    As an example, the templates provided along with the demo data weigh 2.8 MB each when being
    interpolated onto the wavelength grid of the data. That's 1.4GB for 500 RV points (e.g. 250 km/s
    on either side for 1 km/s velocity steps, close to the bare minimum you would need). Doing more
    than 5 such templates in parallel will overflow the memory of a standard laptop (8GB RAM).
    Realisticly, you might be computing 8 or 16 templates in parallel (depending on the number of
    threads you have), with 1000 steps in velocity, meaning that you'd
    need 20 to 40 GB of RAM. That's in the realm of servers.

    Memory overflow won't necessarily make your system crash, as long as it has allocated sufficient
    swap memory. However, that does make your calculation very slow, potentially slower than doing
    the computation in sequence. Therefore, parallelisation of the cross-correlation operation is
    provided via the xcor_parallel keyword, and is switched off by default. In case you are running
    Tayph on a server with many cores and plenty of RAM, switching this on may effect speed gains
    of factors of 5 to 10 in cross-correlation.

    Set the dp keyword to an alternative datapath to override the datapath in the runfile. This is
    to execute the same runfile on multiple datasets (e.g. nights) in a straightforward way (i.e.)
    a loop).

    Set the debug to run the cascade in a stepwise manner, calling pdb.set_trace() after each step
    to allow on the fly inspection of variables and errors.
    """
    import tayph.system_parameters as sp
    import tayph.util as ut
    import numpy as np
    import sys
    cf = configfile



    ut.tprint('')
    ut.tprint('')
    ut.tprint('')
    ut.tprint('')
    ut.tprint('')
    ut.tprint('')
    ut.tprint('')
    ut.tprint('')
    ut.tprint('')
    ut.tprint('')
    ut.tprint('')
    ut.tprint(' = = = = = = = = = = = = = = = = =')
    ut.tprint(' = = = = WELCOME TO TAYPH! = = = =')
    ut.tprint(' = = = = = = = = = = = = = = = = =')
    ut.tprint('')
    ut.tprint(f'    Running {cf}')
    ut.tprint('')
    ut.tprint(' = = = = = = = = = = = = = = = = =')
    ut.tprint('')
    ut.tprint('')
    ut.tprint('')

    if not debug:
        ut.tprint('---Start')
    else:
        ut.tprint('[STARTING TAYPH IN DEBUG MODE]')
        ut.tprint('The code will be halted between major steps using pdb, allowing you to '
        'inspect the values of parameters to enable advanced debugging.')

    ut.tprint(f'---Load parameters from {cf}')
    modellist = sp.paramget('model',cf,full_path=True).split(',')
    templatelist = sp.paramget('template',cf,full_path=True).split(',')

    params={'dp':sp.paramget('datapath',cf,full_path=True),
            'shadowname':sp.paramget('shadowname',cf,full_path=True),
            'maskname':sp.paramget('maskname',cf,full_path=True),
            'RVrange':sp.paramget('RVrange',cf,full_path=True),
            'drv':sp.paramget('drv',cf,full_path=True),
            'f_w':sp.paramget('f_w',cf,full_path=True),
            'do_colour_correction':sp.paramget('do_colour_correction',cf,full_path=True),
            'do_telluric_correction':sp.paramget('do_telluric_correction',cf,full_path=True),
            'do_xcor':sp.paramget('do_xcor',cf,full_path=True),
            'inject_model':sp.paramget('inject_model',cf,full_path=True),
            'plot_xcor':sp.paramget('plot_xcor',cf,full_path=True),
            'make_mask':sp.paramget('make_mask',cf,full_path=True),
            'apply_mask':sp.paramget('apply_mask',cf,full_path=True),
            'do_berv_correction':sp.paramget('do_berv_correction',cf,full_path=True),
            'do_keplerian_correction':sp.paramget('do_keplerian_correction',cf,full_path=True),
            'make_doppler_model':sp.paramget('make_doppler_model',cf,full_path=True),
            'skip_doppler_model':sp.paramget('skip_doppler_model',cf,full_path=True),
            'modellist':modellist,
            'templatelist':templatelist,
            'c_subtract':sp.paramget('c_subtract',cf,full_path=True),
            # 'invert_template':sp.paramget('invert_template',cf,full_path=True),
            'template_library':sp.paramget('template_library',cf,full_path=True),
            'model_library':sp.paramget('model_library',cf,full_path=True),
            'debug':debug
    }
    #Optional keywords:
    try:
        transitkeyword = sp.paramget('transit',cf,full_path=True)
    except:
        transitkeyword = None
        ut.tprint("------Optional keyword 'transit' (True/False) not set. This will be defaulted "
        "to True in tayph.run_instance(). Set to False if this is not in-transit data (warnings "
        "or errors will come up later if this is the case).")

    try:
        sinusoidkeyword = sp.paramget('sinusoid',cf,full_path=True)
    except:
        sinusoidkeyword = False
        ut.tprint("------Optional keyword 'sinusoid' (True/False) not set. This will be defaulted "
        "to False in tayph.run_instance(). Set to True if this is not in-transit data (warnings "
        "or errors will come up later if this is the case).")

    if transitkeyword == True or transitkeyword == False:
        params['transit'] = transitkeyword
    if sinusoidkeyword == True or sinusoidkeyword==False:
        params['sinusoid'] = sinusoidkeyword

    order_skip_num = []
    try:
        order_skiplist = sp.paramget('skip_orders',cf,full_path=True,force_string = True).split(',')
    except:
        ut.tprint("------Optional keyword 'skip_orders' not set. If you want to skip ranges of "
        "orders, you can do so by setting this keyword as e.g. 0,1,2,5,10,15-20,22,23 etc.")
        order_skiplist=None
    if order_skiplist:
        for O in order_skiplist:
            try:
                order_num = int(O)
                order_skip_num.append(order_num)
            except:#This is a dash-separated range:
                order_span = O.split('-')
                if len(order_span) == 2:
                    try:
                        first_order = int(order_span[0])
                        final_order = int(order_span[1])
                    except:
                        raise Exception('ERROR in parsing order skip list. Should be formatted '
                        f'as 0,1,2,5,10,15-20,22,23. Encountered {O}')
                    if first_order >=final_order or first_order <0:
                        raise Exception('ERROR in parsing order skip list. Should be formatted '
                        f'as 0,1,2,5,10,15-20,22,23. Encountered {O}')
                    order_range = np.arange(first_order,final_order+1)
                    for OO in order_range:
                        order_skip_num.append(OO)
                else:
                    raise Exception('ERROR in parsing order skip list. Should be formatted as '
                    f'0,1,2,5,10,15-20,22,23. Encountered {O}')
        params['skip_orders'] = [*set(order_skip_num)]
        ut.tprint("------The following orders are set for skipping during this run "
        f"{','.join(order_skiplist)}")
    if len(dp) > 0:
        params['dp'] = dp
    run_instance(params,parallel=parallel,xcor_parallel=xcor_parallel)


def run_instance(p,parallel=True,xcor_parallel=False):
    """This runs the entire cross correlation analysis cascade."""
    import numpy as np
    from astropy.io import fits
    import astropy.constants as const
    import astropy.units as u
    from matplotlib import pyplot as plt
    import os.path
    import scipy.interpolate as interp
    import pylab
    import pdb
    import os.path
    import os
    import sys
    import glob
    import math
    import distutils.util
    import pickle
    import copy
    from pathlib import Path
    if parallel or xcor_parallel: from joblib import Parallel, delayed
    import tayph.util as ut
    import tayph.operations as ops
    import tayph.functions as fun
    import tayph.system_parameters as sp
    import tayph.tellurics as telcor
    import tayph.masking as masking
    import tayph.models as models
    from tayph.ccf import xcor,clean_ccf,filter_ccf,construct_KpVsys,mask_cor
    from tayph.vartests import typetest,notnegativetest,nantest,postest,typetest_array,dimtest
    from tayph.vartests import lentest
    import tayph.shadow as shadow

    #First parse the parameter dictionary into required variables and test the datapath.
    typetest(p,dict,'params in run_instance()')
    dp = ut.check_path(p['dp'],exists=True)

    #Read all the parameters.
    modellist = p['modellist']
    templatelist = p['templatelist']

    model_library = p['model_library']
    template_library = p['template_library']
    shadowname = p['shadowname']
    maskname = p['maskname']
    RVrange = p['RVrange']
    drv = p['drv']
    try:#Lazy way of giving this a default and creating backward compatibility with old-style config files.
        intransit = p['transit']
        sinusoid = p['sinusoid']
    except:
        intransit=True
        sinusoid=False
    try:
        skip_orders = p['skip_orders']
    except:
        skip_orders = []
    f_w = p['f_w']
    do_colour_correction=p['do_colour_correction']
    do_telluric_correction=p['do_telluric_correction']
    do_xcor=p['do_xcor']
    inject_model=p['inject_model']
    plot_xcor=p['plot_xcor']
    make_mask=p['make_mask']
    apply_mask=p['apply_mask']
    c_subtract=p['c_subtract']
    # invert_template=p['invert_template']
    do_berv_correction=p['do_berv_correction']
    do_keplerian_correction=p['do_keplerian_correction']
    make_doppler_model=p['make_doppler_model']
    skip_doppler_model=p['skip_doppler_model']
    debug = p['debug']
    resolution = sp.paramget('resolution',dp)
    air = sp.paramget('air',dp)#Are wavelengths in air or not?


    #All the checks
    typetest(modellist,[str,list],'modellist in run_instance()')
    typetest(templatelist,[str,list],'templatelist in run_instance()')
    typetest(model_library,str,'model_library in run_instance()')
    typetest(template_library,str,'template_library in run_instance()')
    typetest(skip_orders,list,'skip_orders in run_instance()')
    ut.check_path(model_library,exists=True)
    ut.check_path(template_library,exists=True)
    if type(modellist) == str:
        modellist = [modellist]#Force this to be a list
    if type(templatelist) == str:
        templatelist = [templatelist]#Force this to be a list
    typetest_array(modellist,str,'modellist in run_instance()')
    typetest_array(templatelist,str,'modellist in run_instance()')
    typetest(shadowname,str,'shadowname in run_instance()')
    typetest(maskname,str,'shadowname in run_instance()')
    typetest(RVrange,[int,float],'RVrange in run_instance()')
    typetest(drv,[int,float],'drv in run_instance()')
    typetest(f_w,[int,float],'f_w in run_instance()')
    typetest(resolution,[int,float],'resolution in run_instance()')
    nantest(RVrange,'RVrange in run_instance()')
    nantest(drv,'drv in run_instance()')
    nantest(f_w,'f_w in run_instance()')
    nantest(resolution,'resolution in run_instance()')
    postest(RVrange,'RVrange in run_instance()')
    postest(drv,'drv in run_instance()')
    postest(resolution,'resolution in run_instance()')
    notnegativetest(f_w,'f_w in run_instance()')
    typetest(do_colour_correction,bool, 'do_colour_correction in run_instance()')
    typetest(do_telluric_correction,bool,'do_telluric_correction in run_instance()')
    typetest(do_xcor,bool,              'do_xcor in run_instance()')
    typetest(inject_model,bool,         'inject_model in run_instance()')
    typetest(plot_xcor,bool,            'plot_xcor in run_instance()')
    typetest(make_mask,bool,            'make_mask in run_instance()')
    typetest(apply_mask,bool,           'apply_mask in run_instance()')
    typetest(c_subtract,bool,           'c_subtract in run_instance()')
    # typetest(invert_template,bool,      'invert_template in run_instance()')
    typetest(do_berv_correction,bool,   'do_berv_correction in run_instance()')
    typetest(do_keplerian_correction,bool,'do_keplerian_correction in run_instance()')
    typetest(make_doppler_model,bool,   'make_doppler_model in run_instance()')
    typetest(skip_doppler_model,bool,   'skip_doppler_model in run_instance()')
    typetest(air,bool,'air in run_instance()')
    typetest(intransit,bool,'intransit in run_instance()')
    typetest(debug,bool,'debug in run_instance()')
    typetest(sinusoid,bool,'sinusoid in run_instance()')



    #We start by defining constants and preparing for generating output.
    c=const.c.value/1000.0#in km/s
    colourdeg = 3 #A fitting degree for the colour correction. #should be set in the config file?
    mw = 20.0
    c_thresh = 5.0

    ut.tprint('---Passed parameter input tests.')


    #Override application of doppler shadow correction if Transit==False.
    if intransit==False and make_doppler_model==True:
        ut.tprint('WARNING: Transit=False but make_doppler_model=True. This is unphysical. '
        'Disabling making of doppler model and proceeding.')
        make_doppler_model = False
    if intransit==False and skip_doppler_model==False:
        ut.tprint('WARNING: Transit=False but skip_doppler_model is also False. This is unphysical.'
        ' Disabling application of doppler model and proceeding.')
        skip_doppler_model = True


    ut.tprint(f'---Initiating output folder tree in {Path("output")/dp}.')
    libraryname=str(template_library).split('/')[-1]
    if str(dp).split('/')[0] == 'data':
        dataname=str(dp).replace('data/','')
        ut.tprint('------Data is located in data/ folder. Assuming output name for this dataset as '
        f'{dataname}')
    else:
        dataname=dp
        ut.tprint('------Data is NOT located in data/ folder. Assuming output name for this '
        f'dataset as {dataname}')



    if debug:
        prfx = '[DEBUG] '
        ut.tprint(prfx+'Testing of input variables and output data structure complete. The '
        f'next step is reading the data into lists from {dp}.')
        pdb.set_trace()

    list_of_wls=[]#This will store all the data.
    list_of_orders=[]#All of it needs to be loaded into your memory, more than once.
    list_of_sigmas=[]#Hope that's ok...

    trigger2 = 0#These triggers are used to limit the generation of output in the forloop.
    trigger3 = 0
    n_negative_total = 0#This will hold the total number of pixels that were set to NaN because
    #they were zero when reading in the data.


    #Read in the file names from the datafolder.
    filelist_orders= [str(i) for i in Path(dp).glob('order_*.fits')]
    if len(filelist_orders) == 0:#If no order FITS files are found:
        raise Exception(f'Runtime error: No orders_*.fits files were found in {dp}.')
    try:
        order_numbers = [int(i.split('order_')[1].split('.')[0]) for i in filelist_orders]
    except:
        raise Exception('Runtime error: Failed at casting fits filename numerals to ints. Are the '
        'filenames of all of the spectral orders correctly formatted (e.g. order_5.fits)?')
    order_numbers.sort()#This is the ordered list of numerical order IDs.
    n_orders = len(order_numbers)
    if n_orders == 0:
        raise Exception(f'Runtime error: n_orders may never have ended up as zero? ({n_orders})')


    #Loading the data according to the file names identified.
    if do_xcor == True or plot_xcor == True or make_mask == True:
        ut.tprint(f'---Loading orders from {dp}.')


        for i in order_numbers:
            wavepath = dp/f'wave_{i}.fits'
            orderpath= dp/f'order_{i}.fits'
            sigmapath= dp/f'sigma_{i}.fits'
            ut.check_path(wavepath,exists=True)
            ut.check_path(orderpath,exists=True)
            ut.check_path(sigmapath,exists=False)
            wave_order = ut.readfits(wavepath)#2D or 1D?
            order_i = ut.readfits(orderpath)

            #Check dimensionality of wave axis and order. Either 2D or 1D.
            if wave_order.ndim == 2:
                if i == np.min(order_numbers):
                    ut.tprint('------Wavelength axes provided are 2-dimensional.')
                n_px = np.shape(wave_order)[1]#Pixel width of the spectral order.
                dimtest(wave_order,np.shape(order_i),'wave_order in tayph.run_instance()')
                #Need to have same shape.
            elif wave_order.ndim == 1:
                if i == np.min(order_numbers):
                    ut.tprint('------Wavelength axes provided are 1-dimensional.')
                n_px = len(wave_order)
                dimtest(order_i,[0,n_px],f'order {i} in run_instance()')
            else:
                raise Exception(f'Wavelength axis of order {i} is neither 1D nor 2D.')

            if i == np.min(order_numbers):
                n_exp = np.shape(order_i)[0]#For the first order, we fix n_exp.
                #All other orders should have the same n_exp.
                ut.tprint(f'------{n_exp} exposures recognised.')
            else:
                dimtest(order_i,[n_exp,n_px],f'order {i} in run_instance()')




            #Deal with air or vaccuum wavelengths:
            if air == False:
                if i == np.min(order_numbers):
                    ut.tprint("------Assuming wavelengths are in vaccuum.")
                list_of_wls.append(copy.deepcopy(wave_order))
            else:
                if i == np.min(order_numbers):
                    ut.tprint("------Applying airtovac correction.")
                list_of_wls.append(ops.airtovac(wave_order))


            #Now test for negatives, set them to NaN and track them.
            n_negative = len(order_i[order_i <= 0])
            if trigger3 == 0 and n_negative > 0:
                print("------Setting negative values to NaN.")
                trigger3 = -1
            n_negative_total+=n_negative
            order_i[order_i <= 0] = np.nan #This is very important for later when we are computing
            #average spectra and the like, to avoid divide-by-zero cases.
            postest(order_i,f'order {i} in run_instance().')#make sure whatever comes out here is
            #strictly positive.
            list_of_orders.append(order_i)




            #Try to get a sigma file. If it doesn't exist, we raise a warning. If it does, we test
            #its dimensions and append it.
            try:
                sigma_i = ut.readfits(sigmapath)
                dimtest(sigma_i,[n_exp,n_px],f'order {i} in run_instance().')
                list_of_sigmas.append(sigma_i)
            except FileNotFoundError:
                if trigger2 == 0:
                    ut.tprint('------WARNING: Sigma (flux error) files not provided. '
                    'Assuming sigma = sqrt(flux). This is standard practise for HARPS data, but '
                    'e.g. ESPRESSO has a pipeline that computes standard errors on each pixel more '
                    'accurately. The sqrt approximation is known to be inappropriate for data '
                    'of CARMENES or GIANO, because these pipeline renormalise the spectra. Make '
                    'sure that the errors are treated properly, otherwise the error-bars on the '
                    'output CCFs will be invalid.')
                    trigger2=-1
                list_of_sigmas.append(np.sqrt(order_i))



        #This ends the loop in which the data are read in. Print the final number of pixels that
        #was vetted because they were negative.
        ut.tprint(f'------{n_negative_total} negative values set to NaN in total '
        f'({np.round(100.0*n_negative_total/n_exp/n_px/len(order_numbers),2)}% of total spectral '
        'pixels in dataset).')


    #Test integrity just to be sure.
    if len(list_of_orders) != n_orders:
        raise Exception('Runtime error: n_orders is not equal to the length of list_of_orders. '
        'Something went wrong when reading them in?')

    ut.tprint('---Finished loading dataset to memory.')




    #Track the minimum and maximum wavelengths of the data. This is done here and not when
    #reading the data because at the end of each loop it is determined whether or not to do
    #airtovac, and it is appended immediately, so I split the following off:
    min_wl = np.inf
    max_wl = -1*np.inf
    for wli in list_of_wls:
        min_wl = np.min([min_wl,np.nanmin(wli)])
        max_wl = np.max([max_wl,np.nanmax(wli)])

    ut.tprint(f'---The spectra cover a range from {np.round(min_wl,2)}--{np.round(max_wl,2)} nm.')
    ut.tprint('---Testing existence and coverage of template(s).')
    testlist_of_templates = []
    for A in templatelist:
        wltt,void=models.get_model(A,template_library)
        if np.min(wltt) > max_wl or np.max(wltt) < min_wl:
            raise Exception('ERROR when testing the integrity of the cross-correlation templates.'
            f'The wavelength range of template {A} does not overlap at all with the data'
            f'({np.round(np.min(wltt),2)}--{np.round(np.max(wltt),2)} '
            f'vs {np.round(min_wl,2)}--{np.round(max_wl,2)} respectively). Are the wavelengths of '
            'template and data both in nm?')
        if np.min(wltt) > min_wl or np.max(wltt) < max_wl:
            ut.tprint(f'WARNING: The wavelength range of template {A} does not fully cover the '
            f'range of the data ({np.round(np.min(wltt),2)}--{np.round(np.max(wltt),2)} '
            f'vs {np.round(min_wl,2)}--{np.round(max_wl,2)} respectively). The template will be '
            'padded with zeroes in the missing region.')

    if inject_model:
        ut.tprint('---Testing existence and coverage of model(s).')
        for A in modellist:
            wltt,void=models.get_model(A,model_library)
            if np.min(wltt) > max_wl or np.max(wltt) < min_wl:
                raise Exception('ERROR when testing the integrity of the injection models.'
                f'The wavelength range of model {A} does not overlap at all with the data'
                f'({np.round(np.min(wltt),2)}--{np.round(np.max(wltt),2)} '
                f'vs {np.round(min_wl,2)}--{np.round(max_wl,2)} respectively). Are the wavelengths '
                'of model and data both in nm?')
            if np.min(wltt) > min_wl or np.max(wltt) < max_wl:
                ut.tprint(f'WARNING: The wavelength range of model {A} does not fully cover the '
                f'range of the data ({np.round(np.min(wltt),2)}--{np.round(np.max(wltt),2)} '
                f'vs {np.round(min_wl,2)}--{np.round(max_wl,2)} respectively). The model will be '
                'padded with ones in the missing region upon injection.')

    if debug:
        prfx = '[DEBUG] '
        ut.tprint(prfx+'The data is read and these tests have passed. This means that the data'
        f'files have the right shape, and that the models and templates cover the right '
        'wavelengths. The next step is reading the telluric model files and dividing them '
        'out of the data.')
        pdb.set_trace()
#Apply telluric correction file or not.
    # plt.plot(list_of_wls[60],list_of_orders[60][10],color='red')
    # plt.plot(list_of_wls[60],list_of_orders[60][10]+list_of_sigmas[60][10],color='red',alpha=0.5)
    ##plot corrected spectra
    # plt.plot(list_of_wls[60],list_of_orders[60][10]/list_of_sigmas[60][10],color='red',alpha=0.5)
    ##plot SNR
    if do_telluric_correction == True and n_orders > 0:
        ut.tprint('---Applying telluric correction')
        telpath = dp/'telluric_transmission_spectra.pkl'
        list_of_orders,list_of_sigmas,list_of_Ts = telcor.apply_telluric_correction(telpath,
        list_of_wls,list_of_orders,list_of_sigmas,parallel=parallel) # this is in parallel now
    else:
        list_of_Ts = None

    # plt.plot(list_of_wls[60],list_of_orders[60][10],color='blue')
    # plt.plot(list_of_wls[60],list_of_orders[60][10]+list_of_sigmas[60][10],color='blue',alpha=0.5)
    ##plot corrected spectra

    # plt.plot(list_of_wls[60],list_of_orders[60][10]/list_of_sigmas[60][10],color='blue',alpha=0.5)
    ##plot SNR
    # plt.show()
    # pdb.set_trace()


    if debug:
        ut.tprint(prfx+'Telluric correction has passed. The next step is to do velocity'
        'corrections.')
        pdb.set_trace()

#Do velocity correction of wl-solution. Explicitly after telluric correction
#but before masking. Because the cross-correlation relies on columns being masked.
#Then if you start to move the spectra around before removing the time-average,
#each masked column becomes slanted. Bad deal.
    rv_cor = 0#This initialises as an int. If any of the following is true, it becomes a float.
    if do_berv_correction:
        rv_cor += sp.berv(dp)
    if do_keplerian_correction:
        rv_cor-=sp.RV_star(dp)*(1.0)

    gamma = 1.0+(rv_cor*u.km/u.s/const.c)#Doppler factor.

    if type(rv_cor) != int:#If rv_cor is populated:
        if do_berv_correction == True and do_keplerian_correction == False:
            ut.tprint('---Reinterpolating data to correct BERV velocities')
        elif do_berv_correction == False and do_keplerian_correction == True:
            ut.tprint('---Reinterpolating data to correct Keplerian velocities')
        else:
            ut.tprint('---Reinterpolating data to correct BERV and Keplerian velocities')
        ut.tprint(f'------From {np.round(rv_cor[0],3)} to {np.round(rv_cor[-1],3)} km/s')
    elif list_of_wls[0].ndim==2:#Else, if wavelength grid is 2D:
        ut.tprint('---Reinterpolating 2D data onto a common wavelength axis')

    list_of_orders_cor = []
    list_of_sigmas_cor = []
    list_of_wls_cor = []
    list_of_Ts_cor = []
    list_of_Ts_1D = []
    for i in range(len(list_of_wls)):
        order = list_of_orders[i]
        sigma = list_of_sigmas[i]
        order_cor = order*0.0
        sigma_cor = sigma*0.0
        if do_telluric_correction:
            T_order = list_of_Ts[i]
            T_cor = order * 0.0
        if list_of_wls[i].ndim==2:
            wl_cor = list_of_wls[i][0]#Interpolate onto the 1st wavelength axis of the series if 2D.
        elif list_of_wls[i].ndim==1:
            wl_cor = list_of_wls[i]
        else:
            raise Exception(f'Wavelength axis of order {i} is neither 1D nor 2D.')

        for j in range(len(list_of_orders[0])):
            if list_of_wls[i].ndim==2:
                if type(rv_cor) != int:#If wl 2D and rv_cor is non-zero:
                    order_cor[j] = interp.interp1d(list_of_wls[i][j]*gamma[j],order[j],
                    bounds_error=False)(wl_cor)
                    sigma_cor[j] = interp.interp1d(list_of_wls[i][j]*gamma[j],sigma[j],
                    bounds_error=False)(wl_cor)#I checked that this works because it doesn't affect
                    #the SNR, apart from wavelength-shifting it.
                else:#If wl is 2D and rv_cor is not populated, there is no multiplication with gamma
                    order_cor[j] = interp.interp1d(list_of_wls[i][j],order[j],
                    bounds_error=False)(wl_cor)
                    sigma_cor[j] = interp.interp1d(list_of_wls[i][j],sigma[j],
                    bounds_error=False)(wl_cor)
            else:
                if type(rv_cor) != int:#If wl 1D and rv_cor is non-zero:
                    order_cor[j] = interp.interp1d(list_of_wls[i]*gamma[j],order[j],
                    bounds_error=False)(wl_cor)
                    sigma_cor[j] = interp.interp1d(list_of_wls[i]*gamma[j],sigma[j],
                    bounds_error=False)(wl_cor)
                else:
                    #No interpolation at all:
                    order_cor[j]=order[j]
                    sigma_cor[j]=sigma[j]
            if do_telluric_correction and make_mask == True:
                # lentest(list_of_wls[i]*gamma[j].value,len(T_order[j]),varname='list_of_wls[i]')
                # T_cor[j] = interp.interp1d(list_of_wls[i]*gamma[j].value,T_order[j],
                # bounds_error=False,fill_value=1)(wl_cor)
                if list_of_wls[i].ndim==2:
                    lentest(list_of_wls[i][j],len(T_order[j]),varname='list_of_wls[i]')
                    if type(rv_cor) != int:
                        T_cor[j] = interp.interp1d(list_of_wls[i][j]*gamma[j].value,T_order[j],
                        bounds_error=False,fill_value=1)(wl_cor)
                    else:
                        T_cor[j] = interp.interp1d(list_of_wls[i][j],T_order[j],
                        bounds_error=False,fill_value=1)(wl_cor)
                else:
                    lentest(list_of_wls[i],len(T_order[j]),varname='list_of_wls[i]')
                    if type(rv_cor) != int:
                        T_cor[j] = interp.interp1d(list_of_wls[i]*gamma[j].value,T_order[j],
                        bounds_error=False,fill_value=1)(wl_cor)
                    else:
                        T_cor[j] = interp.interp1d(list_of_wls[i],T_order[j],
                        bounds_error=False,fill_value=1)(wl_cor)

        list_of_orders_cor.append(order_cor)
        list_of_sigmas_cor.append(sigma_cor)
        list_of_wls_cor.append(wl_cor)
        if do_telluric_correction and make_mask == True:
            list_of_Ts_cor.append(T_cor)
        ut.statusbar(i,np.arange(len(list_of_wls)))
    # plt.plot(list_of_wls[60][3],list_of_orders[60][3]/list_of_sigmas[60][3],color='blue')
    # plt.plot(list_of_wls_cor[60],list_of_orders_cor[60][3]/list_of_sigmas_cor[60][3],color='red')
    # plt.show()
    list_of_orders = list_of_orders_cor
    list_of_sigmas = list_of_sigmas_cor
    list_of_wls = list_of_wls_cor
    if do_telluric_correction and make_mask == True:
        for i in range(len(list_of_Ts_cor)):
            list_of_Ts_1D.append(np.nanmean(list_of_Ts_cor[i],axis=0))
    #Now the spectra are telluric corrected and velocity corrected, and the wavelength axes have
    #been collapsed from 2D to 1D (if they were 2D in the first place, e.g. for ESPRESSO).


    if len(list_of_orders) != n_orders:
        raise RuntimeError('n_orders is no longer equal to the length of list_of_orders, though it '
        'was before. Something went wrong during telluric correction or velocity correction.')



    if debug:
        ut.tprint(prfx+'The velocities have been corrected, taking into account the shape of '
        'wavelength grid (1D or 2D).')






    #Compute / create a mask and save it to file (or not)
    if make_mask == True and len(list_of_orders) > 0:
        if debug:
            ut.tprint('Telluric spectra have been traced to be passed to the mask making routine,'
            'routing. Mask-making is the next step, with colour correction if set '
            f'({do_colour_correction}).')
            pdb.set_trace()
        if do_colour_correction == True:
            list_of_orders_mask = ops.normalize_orders(list_of_orders,list_of_orders,colourdeg,sinusoid=sinusoid)[0]
            #I dont want outliers to affect the colour correction later on, so colour correction
            #can't be done before masking. At the same time, masking shouldn't suffer from colour
            #variations either. So this needs to be done twice.
            #The second variable is a dummy to replace the expected list_of_sigmas input.
            ut.tprint('---Constructing mask with intra-order colour correction applied')
            masking.mask_orders(list_of_wls,list_of_orders_mask,dp,maskname,mw,c_thresh,
            manual=True,list_of_Ts=list_of_Ts_1D)
            #There are some hardcoded values here: mw and c_thresh
            #List of Ts is set to None if telluric correction is not applied.
            del list_of_orders_mask

        else:
            ut.tprint('---Constructing mask WITHOUT intra-order colour correction applied.')
            ut.tprint('---Switch on colour correction if you see colour variations in the 2D '
                'spectra.')
            masking.mask_orders(list_of_wls,list_of_orders,dp,maskname,mw,c_thresh,
            manual=True,list_of_Ts=list_of_Ts_1D)
        if apply_mask == False:
            ut.tprint('---WARNING in run_instance: Mask was made but is not applied to data '
                '(apply_mask == False)')


#Apply the mask that was previously created and saved to file.
    if apply_mask == True:
        if debug:
            ut.tprint(prfx+'The next step is application of the mask.')
            pdb.set_trace()
        ut.tprint('---Applying mask to spectra and uncertainty arrays')
        list_of_orders = masking.apply_mask_from_file(dp,maskname,list_of_orders)
        list_of_sigmas = masking.apply_mask_from_file(dp,maskname,list_of_sigmas)


#Skipping orders:
    if len(skip_orders) > 0:
        if debug:
            ut.tprint(prfx+'The next step is removal of orders set my the skip_orders keyword.')
            pdb.set_trace()
        ut.tprint(f'---Removing orders {skip_orders}')
        list_of_wls_not_skipped = []
        list_of_orders_not_skipped = []
        list_of_sigmas_not_skipped = []
        for i in range(len(list_of_orders)):
            if i not in skip_orders:
                list_of_wls_not_skipped.append(copy.deepcopy(list_of_wls[i]))
                list_of_orders_not_skipped.append(copy.deepcopy(list_of_orders[i]))
                list_of_sigmas_not_skipped.append(copy.deepcopy(list_of_sigmas[i]))
        n_orders -= len(skip_orders)
        del list_of_orders
        del list_of_sigmas
        del list_of_wls
        list_of_wls = copy.deepcopy(list_of_wls_not_skipped)
        list_of_orders = copy.deepcopy(list_of_orders_not_skipped)
        list_of_sigmas = copy.deepcopy(list_of_sigmas_not_skipped)
        del list_of_wls_not_skipped
        del list_of_orders_not_skipped
        del list_of_sigmas_not_skipped



#Interpolate over all isolated NaNs and set bad columns to NaN (so that they are ignored in the CCF)
    if do_xcor == True:
        if debug:
            ut.tprint(prfx+'The next step is healing of isolated NaN values by interpolation.')
            pdb.set_trace()
        ut.tprint('---Healing NaNs')
        list_of_orders = masking.interpolate_over_NaNs(list_of_orders,parallel=parallel)
        list_of_sigmas = masking.interpolate_over_NaNs(list_of_sigmas,quiet=True,parallel=parallel)








    #This is the point from which model injection will also start.
    #List_of_orders and list_of_wls are taken to inject the models into below,
    #after first the data is correlated.


    #Normalize the orders to their average flux in order to effectively apply a broad-band colour
    #correction (colour is typically a function of airmass and seeing).
    if do_colour_correction == True:
        if debug:
            ut.tprint(prfx+'The next step is colour correction. The variables list_of_orders and '
            'list_of_sigmas will be deleted after this step..')
            pdb.set_trace()
        ut.tprint('---Normalizing orders to common flux level')
        # plt.plot(list_of_wls[60],list_of_orders[60][10]/list_of_sigmas[60][10],color='blue',
        #alpha=0.4)
        list_of_orders_normalised,list_of_sigmas_normalised,meanfluxes = (
        ops.normalize_orders(list_of_orders,list_of_sigmas,colourdeg,sinusoid=sinusoid))#I tested that this works
        #because it doesn't alter the SNR.
        meanfluxes_norm = meanfluxes/np.nanmean(meanfluxes)
        if inject_model == False:#Conserve memory, as these won't be used anymore.
            del list_of_orders
            del list_of_sigmas
    else:
        meanfluxes_norm = np.ones(len(list_of_orders[0]))
        list_of_orders_normalised = list_of_orders
        list_of_sigmas_normalised = list_of_sigmas
        if debug:
            ut.tprint(prfx+'Colour correction was skipped. The variables list_of_orders and '
            'list_of_sigmas have been renamed to ..._normalised.')
        #fun.findgen(len(list_of_orders[0]))*0.0+1.0#All unity.
        # plt.plot(list_of_wls[60],list_of_orders_normalised[60][10]/list_of_sigmas[60][10],
        # color='red',alpha=0.4)
        # plt.show()
        # sys.exit()



    if len(list_of_orders_normalised) != n_orders:
        raise RuntimeError('n_orders is no longer equal to the length of list_of_orders, though it '
            'was before. Something went wrong during masking or colour correction.')


    #SOMEWHERE AT THE START, TEST THAT EACH OF THE REQUESTED TEMPLATES IS ACTUALLY BINARY OR
    #SPECTRAL. DONT ALLOW MIXING TEMPLATES, MAKES THE XCOR CODE TOO COMPLEX, WHEN SWITCHING IN
    #CCF.PY SOMEWHERE.
    #is_binary = models.get_model(templatename,template_libary,is_binary=True)

        #Construct the cross-correlation templates in case we will be computing or plotting the CCF.
        #These will be saved in lists so that they can be used twice if necessary: once for the
        #data and once for the injected models.
    if do_xcor == True or plot_xcor == True:
        if debug:
            ut.tprint(prfx+'The next step is construction of templates, and recognition of '
            'templates and binary masks.')
            pdb.set_trace()
        def construct_template(templatename,verbose=False):#For use in parallelised iterator.
            wlt,T,is_binary=models.build_template(templatename,binsize=0.5,maxfrac=0.01,resolution=resolution,
                template_library=template_library,c_subtract=c_subtract,verbose=verbose)
                #Top-envelope subtraction and blurring.
            T*=(-1.0)#Absorption lines become emission lines. Absorption is going to be a positive
            #signal
            outpath=Path('output')/Path(dataname)/Path(libraryname)/Path(templatename)

            if not os.path.exists(outpath):
                ut.tprint(f"------The output location ({outpath}) didn't exist, I made it now.")
                os.makedirs(outpath)

            return (wlt, T, is_binary, outpath)


        if parallel:
            list_of_wlts, list_of_templates, binary_flags, outpaths = zip(*Parallel(
            n_jobs=len(templatelist))(delayed(construct_template)(A) for A in templatelist))
        else:
            list_of_wlts, list_of_templates, binary_flags, outpaths = zip(
            *[construct_template(A,verbose=True) for A in templatelist])


    #Divide templates up into spectral and mask templates:
    wlTs_S,T_S,outpaths_S,names_S,wlTs_M,T_M,outpaths_M,names_M = [],[],[],[],[],[],[],[]
    for i in range(len(binary_flags)):
        if binary_flags[i]:#Mask templates
            wlTs_M.append(list_of_wlts[i])
            T_M.append(list_of_templates[i])
            outpaths_M.append(outpaths[i])
            names_M.append(templatelist[i])#The names of the templates that are masks.
        else:#Spectral templates
            wlTs_S.append(list_of_wlts[i])
            T_S.append(list_of_templates[i])
            outpaths_S.append(outpaths[i])
            names_S.append(templatelist[i])#The names of the templates that are spectral.


    if do_xcor:#Perform the cross-correlation on the entire list of orders and the entire list of
    #templates, in parallel or sequentially.
        if debug:
            ut.tprint(prfx+'The next step is cross-correlation. The variables '
            'list_of_orders_normalised and list_of_sigmas_normalised will be deleted after this '
            'step.')
            pdb.set_trace()
        if xcor_parallel:
            ut.tprint(f'---Performing cross-correlation with {len(list_of_templates)} '
            'templates in parallel.')
        else:
            ut.tprint(f'---Performing cross-correlation with {len(list_of_templates)} '
            'templates in sequence.')

        #First we do all the spectral templates (case 1):
        if len(wlTs_S) > 0:
            t1 = ut.start()
            RV,CCFs1,CCF_Es1,T_sums1 = xcor(list_of_wls,list_of_orders_normalised,wlTs_S,
            T_S,drv,RVrange,list_of_errors=list_of_sigmas_normalised,parallel=xcor_parallel)
            txcor  = ut.end(t1,silent=True)
            print(f'------Spectral correlation completed. Time spent: {np.round(txcor,1)}s '
            f'({np.round(txcor/len(T_S),1)} per template).')

        #Then all the mask templates (case 2):
        if len(wlTs_M) > 0:
            t1 = ut.start()
            RV,CCFs2,CCF_Es2,T_sums2 = mask_cor(list_of_wls,list_of_orders_normalised,wlTs_M,
            T_M,drv,RVrange,list_of_errors=list_of_sigmas_normalised,parallel=parallel)
            tmcor  = ut.end(t1,silent=True)
            print(f'------Line-list correlation completed. Time spent: {np.round(tmcor,1)}s '
            f'({np.round(tmcor/len(T_M),1)} per template).')

        del list_of_orders_normalised
        del list_of_sigmas_normalised

        #Now merge the outcome in single arrays:
        if len(wlTs_S) > 0 and len(wlTs_M) > 0:#and make 100% sure that these are lists and not
            #e.g. arrays.
            CCFs = list(CCFs1)+list(CCFs2)
            CCF_Es = list(CCF_Es1)+list(CCF_Es2)
            T_sums = list(T_sums1)+list(T_sums2)
            outpaths = list(outpaths_S)+list(outpaths_M)
            T_names = list(names_S)+list(names_M)
        elif len(wlTs_S) > 0 and len(wlTs_M) == 0:
            CCFs = CCFs1
            CCF_Es = CCF_Es1
            T_sums = T_sums1
            outpaths = outpaths_S
            T_names = names_S
        elif len(wlTs_S) == 0 and len(wlTs_M) > 0:
            CCFs = CCFs2
            CCF_Es = CCF_Es2
            T_sums = T_sums2
            outpaths = outpaths_M
            T_names = names_M
        else:
            raise(RuntimeError(f"Error in length of wlTs_S ({len(wlTs_S)}) and/or the length of "
            f" wlTs_M ({len(wlTs_M)}). One or both of these should be greater than 0, and up to "
            "one of these may be zero."))

        #The following is to compare time spent in parallel computation.
        # txxx2 = ut.start()
        # RV,list_of_CCFs,list_of_CCF_Es,list_of_T_sums = xcor(list_of_wls,list_of_orders_normalised,
        # list_of_wlts,list_of_templates,drv,RVrange,list_of_errors=list_of_sigmas_normalised,
        # parallel=True)
        # txcor_p  = ut.end(txxx2,silent=True)



    #Save CCFs to disk, read them back in and perform cleaning steps.
    if debug:
        ut.tprint(prfx+'Cross-correlation complete. The next step is writing output and cleaning.')
        pdb.set_trace()
    ut.tprint('---Writing CCFs to file and peforming cleaning steps.')
    for i in range(len(T_names)):
        outpath = outpaths[i]
        if do_xcor:
            ut.tprint(f'------Writing CCF of {T_names[i]} to {str(outpath)}')
            ut.writefits(outpath/'ccf.fits',CCFs[i])
            ut.writefits(outpath/'ccf_e.fits',CCF_Es[i])
            ut.writefits(outpath/'RV.fits',RV)
            ut.writefits(outpath/'Tsum.fits',T_sums[i])
        else:
            ut.tprint(f'---Reading CCF of template {T_names[i]} from {str(outpath)}.')
            if os.path.isfile(outpath/'ccf.fits') == False:
                raise FileNotFoundError(f'CCF output not located at {outpath}. Rerun with '
                'do_xcor=True to create these files.')

        #Read regardless of whether XCOR was performed or not. Less code to duplicate...
        rv=ut.readfits(outpath/'RV.fits')
        ccf = ut.readfits(outpath/'ccf.fits')
        ccf_e = ut.readfits(outpath/'ccf_e.fits')
        Tsums = ut.readfits(outpath/'Tsum.fits')

        ut.tprint('---Cleaning CCFs')
        ccf_n,ccf_ne,ccf_nn,ccf_nne= clean_ccf(rv,ccf,ccf_e,dp,intransit)

        if make_doppler_model == True:
            if debug:
                ut.tprint(prfx+'The next step is construction of the doppler shadow.')
                pdb.set_trace()
            shadow.construct_doppler_model(rv,ccf_nn,dp,shadowname,xrange=[-200,200],Nxticks=20.0,
            Nyticks=10.0)
            make_doppler_model = False # This sets it to False after it's been run once, i.e. for
            #the first template.
        if skip_doppler_model == False:
            if debug:
                ut.tprint(prfx+'The next step is application of the doppler shadow.')
                pdb.set_trace()
            ut.tprint(f'---Reading doppler shadow model from {shadowname}')
            doppler_model,dsmask = shadow.read_shadow(dp,shadowname,rv,ccf)#This returns both the
            #model evaluated on the rv,ccf grid, as well as the mask that blocks the planet trace.
            ccf_clean,matched_ds_model = shadow.match_shadow(rv,ccf_nn,dsmask,dp,doppler_model)

            #THIS IS AN ADDITIVE CORRECTION, SO CCF_NNE DOES NOT NEED TO BE ALTERED AND IS STILL
            #VALID VOOR CCF_CLEAN
        else:
            ut.tprint('---Not performing shadow correction')
            ccf_clean = ccf_nn*1.0
            matched_ds_model = ccf_clean*0.0


        #High-pass filtering
        if f_w > 0.0:
            if debug:
                ut.tprint(prfx+'The next step is high-pass filtering.')
                pdb.set_trace()
            ut.tprint(f'---Performing high-pass filter on the CCF with width {f_w}')
            ccf_clean_filtered,wiggles = filter_ccf(rv,ccf_clean,v_width = f_w)#THIS IS ALSO AN
            #ADDITIVE CORRECTION, SO CCF_NNE IS STILL VALID.
        else:
            ut.tprint('---Skipping high-pass filter')
            ccf_clean_filtered = ccf_clean*1.0
            wiggles = ccf_clean*0.0#This filtering is additive so setting to zero is accurate.

        ut.tprint('---Weighing CCF rows by mean fluxes that were normalised out')
        ccf_clean_weighted = np.transpose(np.transpose(ccf_clean_filtered)*meanfluxes_norm)
        #MULTIPLYING THE AVERAGE FLUXES BACK IN! NEED TO CHECK THAT THIS ALSO GOES PROPERLY WITH
        #THE ERRORS!
        ccf_nne = np.transpose(np.transpose(ccf_nne)*meanfluxes_norm)

        ut.save_stack(outpath/'cleaning_steps.fits',[ccf,ccf_nn,ccf_clean,matched_ds_model,
        ccf_clean_filtered,wiggles,ccf_clean_weighted])
        ut.writefits(outpath/'ccf_cleaned.fits',ccf_clean_weighted)
        ut.writefits(outpath/'ccf_cleaned_error.fits',ccf_nne)

        # pdb.set_trace()
        #Turn off KpVsys for now.
        ut.tprint('---Constructing KpVsys')
        Kp,KpVsys,KpVsys_e = construct_KpVsys(rv,ccf_clean_weighted,ccf_nne,dp,parallel=parallel,transit=intransit)
        ut.writefits(outpath/'KpVsys.fits',KpVsys)
        ut.writefits(outpath/'KpVsys_e.fits',KpVsys_e)
        ut.writefits(outpath/'Kp.fits',Kp)

    if debug==True and inject_model==False:
        ut.tprint('Halting at the end to give access to final variables.')
        pdb.set_trace()

    # print('DONE')
    # print(f'Time spent in serial xcor: {txcor}s')
    # print(f'{txcor/len(list_of_wlts)}s per template.')
    # print('')
    # print(f'Time spent in parallel xcor: {txcor_p}s')
    # print(f'{txcor_p/len(list_of_wlts)}s per template.')


    #Now repeat it all for the model injection.
    if inject_model == True and do_xcor == True:
        if debug:
            ut.tprint(prfx+'The next step is model injection. The whole cascade is repeated for.'
            'each model. If you have set multipe models, you will be passed to pdb for each of '
            'them')
            pdb.set_trace()
        for modelname in modellist:
            if do_xcor == True:
                print('---Injecting model '+modelname)
                list_of_orders_injected=models.inject_model(list_of_wls,list_of_orders,dp,modelname,
                model_library=model_library,intransit=intransit)#Start with the unnormalised orders
                #from before. Normalize the orders to their average flux in order to effectively
                #apply a broad-band colour correction (colour is a function of airmass and seeing).
                if debug:
                    ut.tprint(prfx+'Model injection complete.')
                if do_colour_correction == True:
                    if debug:
                        ut.tprint(prfx+'The next step is colour correction. The variables '
                        'list_of_orders and list_of_sigmas will be deleted after this step..')
                        pdb.set_trace()
                    print('------Normalizing injected orders to common flux level')
                    list_of_orders_injected,list_of_sigmas_injected,meanfluxes_injected = (
                    ops.normalize_orders(list_of_orders_injected,list_of_sigmas,colourdeg,sinusoid=sinusoid))
                    meanfluxes_norm_injected = meanfluxes_injected/np.mean(meanfluxes_injected)
                else:
                    meanfluxes_norm_injected = np.ones(len(list_of_orders_injected[0])) #fun.findgen(len(list_of_orders_injected[0]))*0.0+1.0

                if xcor_parallel:
                    ut.tprint(f'---Performing cross-correlation with {len(list_of_templates)} '
                    'templates in parallel.')
                else:
                    ut.tprint(f'---Performing cross-correlation with {len(list_of_templates)} '
                    'templates in sequence.')
                if debug:
                    ut.tprint(prfx+'The next step is cross-correlation. The variables '
                    'list_of_orders_normalised and list_of_sigmas_normalised will be deleted after '
                    'this step.')
                    pdb.set_trace()
                #First we do all the spectral templates (case 1):
                if len(wlTs_S) > 0:
                    t1 = ut.start()
                    RV_i,CCFs1_i,CCF_Es1_i,T_sums1_i = xcor(list_of_wls,list_of_orders_injected,
                    wlTs_S,T_S,drv,RVrange,list_of_errors=list_of_sigmas_injected,
                    parallel=xcor_parallel)
                    txcor  = ut.end(t1,silent=True)
                    print(f'------Spectral correlation completed. Time spent: '
                    f'{np.round(txcor,1)}s ({np.round(txcor/len(T_S),1)} per template).')

                #Then all the mask templates (case 2):
                if len(wlTs_M) > 0:
                    t1 = ut.start()
                    RV_i,CCFs2_i,CCF_Es2_i,T_sums2_i = mask_cor(list_of_wls,
                    list_of_orders_injected,wlTs_M,T_M,drv,RVrange,
                    list_of_errors=list_of_sigmas_injected,parallel=parallel)
                    tmcor  = ut.end(t1,silent=True)
                    print(f'------Line-list correlation completed. Time spent: '
                    f'{np.round(tmcor,1)}s ({np.round(tmcor/len(T_M),1)} per template).')

                #Now merge the outcome in single arrays:
                if len(wlTs_S) > 0 and len(wlTs_M) > 0:#and make 100% sure that these are lists
                    #and not e.g. arrays.
                    CCFs_i = list(CCFs1_i)+list(CCFs2_i)
                    CCF_Es_i = list(CCF_Es1_i)+list(CCF_Es2_i)
                    T_sums_i = list(T_sums1_i)+list(T_sums2_i)
                    outpaths = list(outpaths_S)+list(outpaths_M)
                    T_names = list(names_S)+list(names_M)
                elif len(wlTs_S) > 0 and len(wlTs_M) == 0:
                    CCFs_i = CCFs1_i
                    CCF_Es_i = CCF_Es1_i
                    T_sums_i = T_sums1_i
                    outpaths = outpaths_S
                    T_names = names_S
                elif len(wlTs_S) == 0 and len(wlTs_M) > 0:
                    CCFs_i = CCFs2_i
                    CCF_Es_i = CCF_Es2_i
                    T_sums_i = T_sums2_i
                    outpaths = outpaths_M
                    T_names = names_M
                else:
                    raise(RuntimeError(f"Error in length of wlTs_S ({len(wlTs_S)}) and/or the "
                    f"length of wlTs_M ({len(wlTs_M)}). One or both of these should be greater "
                    "than 0, and only up to one of these may be zero."))

                #Save CCFs to disk, read them back in and perform cleaning steps.
                ut.tprint('---Writing CCFs to file and peforming cleaning steps.')
                if debug:
                    ut.tprint(prfx+'Cross-correlation complete. The next step is writing output '
                    'and cleaning.')
                    pdb.set_trace()
                for i in range(len(T_names)):
                    outpath_i = outpaths[i]/modelname
                    if do_xcor:
                        ut.tprint(f'------Writing {modelname}-injected CCF of {T_names[i]} '
                        f'to {str(outpath_i)}.')
                        if not os.path.exists(outpath_i):
                            ut.tprint("---------That path didn't exist, I made it now.")
                            os.makedirs(outpath_i)


                        ut.writefits(outpath_i/'ccf.fits',CCFs_i[i])
                        ut.writefits(outpath_i/'ccf_e.fits',CCF_Es_i[i])
                        ut.writefits(outpath_i/'RV.fits',RV)
                        ut.writefits(outpath_i/'Tsum.fits',T_sums_i[i])
                    else:
                        ut.tprint(f'---Reading CCF of template {T_names[i]} from '
                        f'{str(outpath_i)}.')
                        if os.path.isfile(outpath_i/'ccf.fits') == False:
                            raise FileNotFoundError(f'Injected CCF not located at '
                            f'{str(outpath_i)}. Rerun with do_xcor=True to create these files.')

                    #Read regardless of whether XCOR was performed or not. Less code to duplicate...
                    rv_i=ut.readfits(outpath_i/'RV.fits')
                    ccf_i = ut.readfits(outpath_i/'ccf.fits')
                    ccf_e_i = ut.readfits(outpath_i/'ccf_e.fits')
                    Tsums_i = ut.readfits(outpath_i/'Tsum.fits')


                    ut.tprint('---Cleaning injected CCFs')
                    ccf_n_i,ccf_ne_i,ccf_nn_i,ccf_nne_i = clean_ccf(rv_i,ccf_i,ccf_e_i,dp,intransit)


                    if skip_doppler_model == False:
                        # ut.tprint(f'---Reading doppler shadow model from {shadowname}')
                        # doppler_model,maskHW = shadow.read_shadow(dp,shadowname,rv,ccf)#This does not
                        #need to be repeated because it was already done during the correlation with
                        #the data.
                        if debug:
                            ut.tprint(prfx+'The next step application of the doppler-shadow.')
                            pdb.set_trace()
                        ccf_clean_i,matched_ds_model_i = shadow.match_shadow(rv_i,ccf_nn_i,dsmask,dp,
                        doppler_model)
                    else:
                        ut.tprint('---Not performing shadow correction on injected spectra either.')
                        ccf_clean_i = ccf_nn_i*1.0
                        matched_ds_model_i = ccf_clean_i*0.0


                    #High-pass filtering
                    if f_w > 0.0:
                        if debug:
                            ut.tprint(prfx+'The next step is high-pass filtering.')
                            pdb.set_trace()
                        ccf_clean_i_filtered,wiggles_i = filter_ccf(rv_i,ccf_clean_i,v_width = f_w)
                    else:
                        ut.tprint('---Skipping high-pass filter')
                        ccf_clean_i_filtered = ccf_clean_i*1.0
                        wiggles_i = ccf_clean*0.0


                    ut.tprint('---Weighing injected CCF rows by mean fluxes that were normalised out')
                    ccf_clean_i_weighted = np.transpose(np.transpose(ccf_clean_i_filtered) *
                    meanfluxes_norm_injected)
                    ccf_nne_i = np.transpose(np.transpose(ccf_nne_i)*meanfluxes_norm_injected)
                    ut.writefits(outpath_i/'ccf_cleaned_i.fits',ccf_clean_i_weighted)
                    ut.writefits(outpath_i/'ccf_cleaned_i_error.fits',ccf_nne)


                    #Disable KpVsys diagrams for now.
                    ut.tprint('---Constructing injected KpVsys')
                    Kp_i,KpVsys_i,KpVsys_e_i = construct_KpVsys(rv_i,ccf_clean_i_weighted,ccf_nne_i,dp,parallel=parallel,transit=intransit)
                    ut.writefits(outpath_i/'KpVsys_i.fits',KpVsys_i)
                    ut.writefits(outpath_i/'KpVsys_e_i.fits',KpVsys_e_i)
                    ut.writefits(outpath_i/'Kp.fits',Kp)

                    if debug:
                        ut.tprint('Halting at the end to give access to final variables.')
                        pdb.set_trace()
                    print('')
                    # if plot_xcor == True:
                    #     print('---Plotting KpVsys with '+modelname+' injected.')
                    #     analysis.plot_KpVsys(rv_i,Kp_i,KpVsys,dp,injected=KpVsys_i)
    print('')
    print('---Tayph run completed successfully.')
    print('')
    print('\a')
    """
            def do_xcor_inj_parallel(i, do_xcor=do_xcor, skip_doppler_model=skip_doppler_model, dsmask=dsmask, f_w=f_w):
                templatename = templatelist[i]
                wlt = list_of_wlts[i]
                T = list_of_templates[i]
                outpath_i = outpaths[i]/modelname
                #Save the correlation results in subfolders of the template, which was in:
                    #Path('output')/Path(dataname)/Path(libraryname)/Path(templatename)
                if do_xcor == True:
                    #ut.tprint('------Cross-correlating injected orders')
                    rv_i,ccf_i,ccf_e_i,Tsums_i=xcor(list_of_wls,list_of_orders_injected,np.flipud(
                        np.flipud(wlt)),T,drv,RVrange,list_of_errors=list_of_sigmas_injected)
                    ut.tprint(f'------Writing injected CCFs to {outpath_i}')
                    if not os.path.exists(outpath_i):
                        #ut.tprint("---------That path didn't exist, I made it now.")
                        os.makedirs(outpath_i)
                    ut.writefits(outpath_i/'ccf_i.fits',ccf_i)
                    ut.writefits(outpath_i/'ccf_e_i.fits',ccf_e_i)
                    ut.writefits(outpath_i/'RV.fits',rv_i)
                    ut.writefits(outpath_i/'Tsum.fits',Tsums_i)
                else:
                    #ut.tprint(f'---Reading injected CCFs from {outpath_i}')
                    if os.path.isfile(outpath_i/'ccf_i.fits') == False:
                        raise FileNotFoundError(f'Injected CCF not located at {str(outpath_i)} '
                        'Rerun do_xcor=True and inject_model=True to create these files.')
                    rv_i = fits.getdata(outpath_i/'RV.fits')
                    ccf_i = fits.getdata(outpath_i/'ccf_i.fits')
                    ccf_e_i = fits.getdata(outpath_i/'ccf_e_i.fits')
                    Tsums_i = fits.getdata(outpath_i/'Tsum.fits')



                #ut.tprint('---Cleaning injected CCFs')
                ccf_n_i,ccf_ne_i,ccf_nn_i,ccf_nne_i = clean_ccf(rv_i,ccf_i,ccf_e_i,dp)


                if skip_doppler_model == False:
                    # ut.tprint(f'---Reading doppler shadow model from {shadowname}')
                    # doppler_model,maskHW = shadow.read_shadow(dp,shadowname,rv,ccf)#This does not
                    #need to be repeated because it was already done during the correlation with
                    #the data.
                    ccf_clean_i,matched_ds_model_i = shadow.match_shadow(rv_i,ccf_nn_i,dsmask,dp,
                    doppler_model)
                else:
                    #ut.tprint('---Not performing shadow correction on injected spectra either.')
                    ccf_clean_i = ccf_nn_i*1.0
                    matched_ds_model_i = ccf_clean_i*0.0


                #High-pass filtering
                if f_w > 0.0:
                    ccf_clean_i_filtered,wiggles_i = filter_ccf(rv_i,ccf_clean_i,v_width = f_w)
                else:
                    #ut.tprint('---Skipping high-pass filter')
                    ccf_clean_i_filtered = ccf_clean_i*1.0
                    wiggles_i = ccf_clean*0.0


                #ut.tprint('---Weighing injected CCF rows by mean fluxes that were normalised out')
                ccf_clean_i_weighted = np.transpose(np.transpose(ccf_clean_i_filtered) *
                meanfluxes_norm_injected)
                ccf_nne_i = np.transpose(np.transpose(ccf_nne_i)*meanfluxes_norm_injected)

                ut.writefits(outpath_i/'ccf_cleaned_i.fits',ccf_clean_i_weighted)
                ut.writefits(outpath_i/'ccf_cleaned_i_error.fits',ccf_nne_i)


                #Disable KpVsys diagrams for now.
                # ut.tprint('---Constructing injected KpVsys')
                # Kp_i,KpVsys_i,KpVsys_e_i = construct_KpVsys(rv_i,ccf_clean_i_weighted,ccf_nne_i,dp)
                # ut.writefits(outpath_i/'KpVsys_i.fits',KpVsys_i)
                # ut.writefits(outpath_i/'KpVsys_e_i.fits',KpVsys_e_i)
                # ut.writefits(outpath_i/'Kp.fits',Kp)


                print('')
                # if plot_xcor == True:
                #     print('---Plotting KpVsys with '+modelname+' injected.')
                #     analysis.plot_KpVsys(rv_i,Kp_i,KpVsys,dp,injected=KpVsys_i)
                return 0

            result = Parallel(n_jobs=-1, verbose=5)(delayed(do_xcor_inj_parallel)(i) for i in range(len(list_of_wlts)))
    """




def read_e2ds(inpath,outname,read_s1d=True,instrument='HARPS',star='solar',
config=False,save_figure=True,skysub=False,rawpath=None):
    """This is the workhorse for reading in a time-series of archival 2D echelle
    spectra from a couple of instrument pipelines that produce a standard output,
    and formatting these into the order-wise FITS format that Tayph uses. These

    The user should point this script to a folder (located at inpath) that contains
    their pipeline-reduced echelle spectra. The script expects a certain data
    format, depending on the instrument in question. It is designed to accept
    pipeline products of the HARPS, HARPS-N, ESPRESSO, CARMENES and UVES instruments. In the
    case of HARPS, HARPS-N, CARMENES and ESPRESSO these may be downloaded from the archive.
    UVES is a bit special, because the 2D echelle spectra are not a standard
    pipeline output. Typical use cases are explained further below.

    Outname is to be set to the name of a datafolder to be located in the data/ subdirectory.
    An absolute path can also be set (in which case the path stars with a "/"). A relative path
    can also be set, but needs to start with a "." to denote the current path. Otherwise, Tayph
    will place a folder named "data" in the data/ subdirectory, which is probably not your
    intention.

    This script formats the time series of 2D echelle spectra into 2D FITS images,
    where each FITS file is the time-series of a single echelle order. If the
    spectrograph has N orders, an order spans npx pixels, and M exposures
    were taken during the time-series, there will be N FITS files, each measuring
    M rows by npx columns. This script will read the headers of the pipeline
    reduced data files to determine the date/time of each, the exposure time, the
    barycentric correction (without applying it) and the airmass, and writes
    these to an ASCII table along with the FITS files containing the spectral
    orders.

    A crucial functionality of Tayph is that it also acts as a wrapper
    for the Molecfit telluric correction software. If installed properly, the
    user can call this script with the read_s1d keyword to extract 1D spectra from
    the time-series. The tayph.run.molecfit function can then be used to
    let Molecfit loop ver the entire timeseries. To enable this functionality,
    the current script needs to read the full-width, 1D spectra that are output
    by the instrument pipelines. These are saved along with the 2D data.
    Tayph.run.molecfit can then be called separately on the output data-folder
    to apply molecfit to this time-series of 1D spectra, creating a
    time-series of models of the telluric absorption spectrum that is saved along
    with the 2D fits files. Tayph later interpolates these models onto the 2D
    spectra to telluric-correct them, as part of the main dataflow.
    Molecfit is called once in GUI-mode, allowing the user to select the
    relevant fitting regions and parameters, after which it is repeated
    automatically for the entire time series.

    The read_s1d keyword to activate this functionality is currently ignored.
    Instead, this switch is now done based on the mode keyword alone, but the keyword is left in
    for compatibility purposes. We may decide to make this optional again, later.

    Set the config keyword equal to true, if you want an example config file to be created in the
    data output folder, named config_empty. You can then fill in this file for your system, and
    this function will fill in the required keywords for the geographical coordinates and air,
    based on the instrument mode selected.


    By setting the measure_RV keyword, the code will run a preliminary
    cross-correlation with a stellar (PHOENIX) and a telluric template at both vaccuum
    and air wavelengths to provide the user with information about the nature of the
    adopted wavelength solution and possible wavelength shifts. the star keyword
    allows the user to switch between 3 stellar PHOENIX templates. A solar type (6000K),
    a hot (9000K) or a cool (4000K) template are available, and accessed by setting the
    star keyword to 'solar', 'hot' or 'cool' respectively. Set the save_figure keyword to save the
    plot of the spectra and the CCF to the data folder as a PDF.

    By setting rawpath some of the readers can access the RAW pre-pipeline fits file. This is
    necessary for pipelines that does not preserve necessary headers, like CRIRES pycrires



    About the different spectrographs:
    The processing of HARPS, HARPS-N and ESPRESSO data is executed in an almost
    identical manner, because the pipeline-reduced products are almost identical.
    To run on either of these instruments, the user simply downloads all pipeline
    products of a given time-series, and extracts these in the same folder, producing
    a sequence of ccfs, e2ds/s2d, s1d, blaze, wave files for each exposure.

    For UVES, the functionality is much more constricted because the pipeline
    reduced data in the ESO archive is generally not of sufficient stability to
    enable precise time-resolved spectroscopy. I designed this function therefore
    to run on intermediary pipeline-products produced by the Reflex (GUI) software. For this,
    a user should download the raw UVES data of their time series, letting ESO's
    calselector tool find the associated calibration files. This can easily be
    many GBs worth of data for a given observing program. The user should then
    reduce these data with the Reflex software. Reflex creates resampled, stitched
    1D spectra as its primary output. However, we will use the intermediate
    pipeline products, which include the 2D extracted orders, located in Reflex's
    working directory after the reduction process is completed.

    A further complication of UVES data is that it can be used with different
    dichroics and 'arms', leading to spectral coverage on the blue, and/or redu and redl
    chips. The user should take care that their time series contains only one blue or
    red types at any time. If they are mixed, this script will throw an exception.
    The blue and red arms are regarded as two different spectrographs, and they are. The most
    important complication this causes is that they have different read-out times, leading to
    differently sampled time-series. The two red chips (redu and redl) are combined when reading in
    the data.

    When reading ESPRESSO data, the user may choose to read sky-subtracted spectra by setting
    the skysub keyword to True. If not, Tayph will read the BLAZE files by default. Note that sky
    subtracted files are also blaze-corrected, meaning that the CCFs will be weighed poorly in
    Tayph. (Therefore, consider reading of sky-subtracted files to be an incomplete functionality
    at this time).
    """
    import pkg_resources
    import os
    import pdb
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    import copy
    import scipy.interpolate as interp
    from scipy import interpolate
    import pickle
    from pathlib import Path
    import warnings
    import glob
    import subprocess
    import textwrap

    import tayph.util as ut
    from tayph.vartests import typetest,dimtest
    import tayph.tellurics as mol
    import tayph.system_parameters as sp
    import tayph.functions as fun
    import tayph.operations as ops
    from tayph.ccf import xcor
    import tayph.masking as masking
    import tayph.models as models
    from tayph.read import read_harpslike, read_espresso, read_uves, read_carmenes,read_fies,read_foces
    from tayph.read import read_spirou, read_gianob, read_crires, read_hires_makee, read_nirps
    from astropy.io import fits
    import astropy.constants as const
    import astropy.units as u


    mode = copy.deepcopy(instrument)#Transfer from using the mode keyword to instrument keyword
    #to be compatible with Molecfit.
    print('\n \n \n')
    print('\n \n \n')
    print('\n \n \n')
    print('\n \n \n')
    print(' = = = = = = = = = = = = = = = = =')
    print(' = = = WELCOME TO READ_E2DS! = = =')
    print(' = = = = = = = = = = = = = = = = =')
    print('')
    print(f'    Reading {mode} data')
    print('')
    print(' = = = = = = = = = = = = = = = = =')
    print('\n \n \n')


    #First check the input:
    inpath=ut.check_path(inpath,exists=True)
    typetest(outname,str,'outname in read_e2ds()')
    typetest(read_s1d,bool,'read_s1d switch in read_e2ds()')
    typetest(mode,str,'mode in read_e2ds()')

    if mode not in ['HARPS','HARPSN','HARPS-N','ESPRESSO','UVES-red','UVES-blue',
        'CARMENES-VIS','CARMENES-NIR','SPIROU','GIANO-B','CRIRES','HIRES-MAKEE','FIES','NIRPS','FOCES']:
        raise ValueError("in read_e2ds: instrument needs to be set to HARPS, HARPSN, UVES-red, "
            "UVES-blue CARMENES-VIS, CARMENES-NIR, SPIROU, GIANO-B, CRIRES, HIRES-MAKEE, FIES, "
            "NIRPS or ESPRESSO.")



    if mode in ['HARPS','HARPSN','ESPRESSO','CARMENES-VIS','GIANO-B','CRIRES','HIRES-MAKEE','FIES','NIRPS','FOCES']:
        read_s1d = True
    else:
        read_s1d = False

    #if outname[0] in ['/','.']:#Test that this is an absolute path. If so, we trigger a warning.
        #ut.tprint(f'ERROR: The name of the dataset {outname} appears to be set as an absolute path.'
        #' or a relative path from the current directory. However, Tayph is designed to run in a '
        #'working directory in which datasets, models and cross-correlation output is bundled. To '
        #'that end, this variable is supposed to be set to a name, or a name with a subfolder (like '
        #'"WASP-123" or "WASP-123/night1"), which is to be placed in the data/ subdirectory of the '
        #'current folder. To initialise this file structure, please make an empty working directory '
        #'in e.g. /home/user/tayph/xcor_project/, start the python interpreter in this directory and '
        #'create a dummy file structure using the make_project_folder function, e.g. by importing '
        #'tayph.run and calling tayph.run.make_project_folder("/home/user/tayph/xcor_project/").')
        #sys.exit()

    if outname[0] in ['/','.']:#Test that this is an absolute path. If so, we trigger a warning.
        ut.tprint(f'WARNING: The name of the dataset {outname} appears to be set as an absolute '
        'path or a relative path from the current directory. However, Tayph is designed to run '
        'in a working directory in which datasets, models and cross-correlation output is bundled. '
        'To that end, this variable is recommended to be set to a name, or a name with a subfolder '
        'structure, (like "WASP-123" or "WASP-123/night1"), which is to be placed in the data/ '
        'subdirectory. To initialise this file structure, please make an empty working directory '
        'in e.g. /home/user/tayph/xcor_project/, start the python interpreter in this directory '
        'and create a dummy file structure using the make_project_folder function, e.g. by '
        'importing tayph.run and calling '
        'tayph.run.make_project_folder("/home/user/tayph/xcor_project/").\n\n'
        'Read_e2ds will proceed, assuming that you know what you are doing.')
        outpath = Path(outname)
    else:
        outpath = Path('data')/outname
    if os.path.exists(outpath) != True:
        os.makedirs(outpath)

    #Start reading data files.
    filelist_raw=os.listdir(inpath)#If mode == UVES, these are folders. Else, they are fits files.

    #Append only the things that are fits files and files that are not hidden. Deal with UVES later.
    filelist = []
    for f in filelist_raw:
        if Path(f).suffix.lower() == '.fits' and f.startswith('.') == False:
            filelist.append(f)

    if len(filelist) == 0:
        raise FileNotFoundError(f" in read_e2ds: input folder {str(inpath)} does not contain "
        "FITS files.")


    #MODE SWITCHING AND READING:
    #ADD READ_S1D OPTION. RIGHT NOW S1Ds ARE REQUIRED BY THE BELOW READING FUNCTIONS,
    #EVEN THOUGH THE USER HAS A SWITCH TO TURN THIS OFF.
    if mode=='HARPS-N': mode='HARPSN'#Guard against various ways of referring to HARPS-N.
    print(f'---Read_e2ds is attempting to read a {mode} datafolder at {str(inpath)}.')
    print('---These are the files encountered:')
    if mode in ['HARPS','HARPSN']:
        DATA = read_harpslike(inpath,filelist,mode,read_s1d=read_s1d)#This is a dictionary
        #containing all the e2ds files, wavelengths, s1d spectra, etc.
    elif mode in ['UVES-red','UVES-blue']:
        DATA = read_uves(inpath,filelist,mode)
    elif mode == 'ESPRESSO':
        DATA = read_espresso(inpath,filelist,read_s1d=read_s1d,skysub=skysub)
    elif mode == 'CARMENES-VIS':
        DATA = read_carmenes(inpath,filelist,'vis',construct_s1d=read_s1d)
    elif mode == 'CARMENES-NIR':
        DATA = read_carmenes(inpath,filelist,'nir',construct_s1d=read_s1d)
    elif mode == 'SPIROU':
        DATA = read_spirou(inpath, filelist, read_s1d=read_s1d)
    elif mode == 'GIANO-B':
        DATA = read_gianob(inpath, filelist, read_s1d=read_s1d)
    elif mode == 'CRIRES':
        DATA = read_crires(inpath, filelist, rawpath=rawpath, read_s1d=read_s1d)
    elif mode == 'HIRES-MAKEE':
        DATA = read_hires_makee(inpath, filelist, construct_s1d=read_s1d)
    elif mode == 'FIES':
        DATA = read_fies(inpath,filelist,mode)
    elif mode == 'NIRPS':
        DATA = read_nirps(inpath,filelist,read_s1d=read_s1d,skysub=skysub)
    elif mode == 'FOCES':
        DATA = read_foces(inpath, filelist, mode)
    else:
        raise ValueError(f'Error in read_e2ds: {mode} is not a valid instrument.')


    #There is a problem in this way of dealing with airmass and that is that the
    #derivative of the airmass may not be constant for higher airmass, meaning that
    #the average should be weighted in some way. This can probably be achieved by using
    #information about the start end and time duration between the two, but its a bit
    #more difficult. All the more reason to keep exposures short...


    wave    = DATA['wave']#List of 2D wavelength frames for each echelle order.
    e2ds    = DATA['e2ds']#List of 2D extracted echelle orders.
    norders     = DATA['norders']#Vertical size of each e2ds file.
    npx         = DATA['npx']#Horizontal size of each e2ds file.
    framename   = DATA['framename']#Name of file read.
    mjd     = DATA['mjd']#MJD of observation.
    header     = DATA['header']#List of header objects.
    berv    = DATA['berv']#Radial velocity of the Earth w.r.t. the solar system barycenter.
    airmass = DATA['airmass']#The average airmass of the observation ((end-start)/2)
    texp = DATA['texp']#The exposure time of each exposure in seconds.
    date = DATA['date']#The date of observation in yyyymmddThhmmss.s format.
    obstype = DATA['obstype']#Observation type, should be SCIENCE.
    if mode == 'ESPRESSO':
        SNR = DATA['snr550']
    if read_s1d:
        wave1d  = DATA['wave1d']#List of 1D stitched wavelengths.
        s1d     = DATA['s1d']#List of 1D stiched spectra.
        s1dmjd  = DATA['s1dmjd']#MJD of 1D stitched spectrum (should be the same as MJD).
        s1dhdr  = DATA['s1dhdr']#List of S1D headers. Used for passing weather information to Molecfit.
    del DATA#Free memory.

    #Typetest all of these:
    typetest(wave,list,'wave in read_e2ds()')
    typetest(e2ds,list,'e2ds in read_e2ds()')
    typetest(date,list,'date in read_e2ds()')
    typetest(norders,np.ndarray,'norders in read_e2ds()')
    typetest(npx,np.ndarray,'npx in read_e2ds()')
    typetest(mjd,np.ndarray,'mjd in read_e2ds()')
    typetest(berv,np.ndarray,'berv in read_e2ds()')
    typetest(airmass,np.ndarray,'airmass in read_e2ds()')
    typetest(texp,np.ndarray,'texp in read_e2ds()')
    if read_s1d:
        typetest(wave1d,list,'wave1d in read_e2ds()')
        typetest(s1d,list,'s1d in read_e2ds()')
        typetest(s1dmjd,np.ndarray,'s1dmjd in read_e2ds()')
    e2ds_count  = len(e2ds)



    #Now we catch some things that could be wrong with the data read:
    #1-The above should have read a certain nr of e2ds files that are classified as SCIENCE frames.
    #2-There should be equal numbers of wave and e2ds frames and wave1d as s1d frames.
    #3-The number of s1d files should be the same as the number of e2ds files.
    #4-All exposures should have the same number of spectral orders.
    #5-All orders should have the same number of pixels.
    #6-The wave frames should have the same dimensions as the e2ds frames.

    #Test 1
    if e2ds_count == 0:
        raise FileNotFoundError(f"in read_e2ds: The input folder {str(inpath)} does not contain "
            "files recognised as e2ds-format FITS files with SCIENCE keywords.")
    #Test 2,3
    if read_s1d:
        if len(wave1d) != len(s1d):
            raise ValueError(f"in read_e2ds: The number of 1D wavelength axes and S1D files does "
            f"not match ({len(wave1d)},{len(s1d)}). Between reading and this test, one of them has "
            "become corrupted. This should not realisticly happen.")
        if len(e2ds) != len(s1d):
            raise ValueError(f"in read_e2ds: The number of e2ds files and s1d files does not match "
            f"({len(e2ds)},{len(s1d)}). Make sure that s1d spectra matching the e2ds spectra are "
            "provided.")
    if len(wave) != len(e2ds):
        raise ValueError(f"in read_e2ds: The number of 2D wavelength frames and e2ds files does "
            f"not match ({len(wave)},{len(e2ds)}). Between reading and this test, one of them has "
            "become corrupted. This should not realisticly happen.")

    #Test 4 - and set norders to its constant value.
    if np.max(np.abs(norders-norders[0])) == 0:
        norders=int(norders[0])
    else:
        print('\n \n \n')
        print("These are the files and their numbers of orders:")
        for i in range(e2ds_count):
            print('   '+framename[i]+'  %s' % norders[i])
        raise ValueError("in read_e2ds: Not all e2ds files have the same number of orders. The "
            "list of frames is printed above.")
    #Test 5 - and set npx to its constant value.
    if np.max(np.abs(npx-npx[0])) == 0:
        npx=int(npx[0])
    else:
        print('\n \n \n')
        print("These are the files and their numbers of pixels:")
        for i in range(len(obstype)):
            print('   '+framename[i]+'  %s' % npx[i])
        raise ValueError("in read_e2ds: Not all e2ds files have the same number of pixels. The "
            "list of frames is printed above.")
    #Test 6
    for i in range(e2ds_count):
        if np.shape(wave[i])[0] != np.shape(e2ds[i])[0] or np.shape(wave[i])[1] != np.shape(
        e2ds[i])[1]:
            raise ValueError(f"in read_e2ds: wave[{i}] does not have the same dimensions as "
                f"e2ds[{i}] ({np.shape(wave)},{np.shape(e2ds)}).")

        if read_s1d == True:
            if len(wave1d[i]) != len(s1d[i]):
                raise ValueError(f"in read_e2ds: wave1d[{i}] does not have the same length as "
                f"s1d[{i}] ({len(wave)},{len(e2ds)}).")
            dimtest(wave1d[i],np.shape(s1d[i]),f'wave1d[{i}] and s1d[{i}] in read_e2ds()')
            #This error should never be triggered, but just in case.

    #Ok, so now we should have ended up with a number of lists that contain all
    #the relevant science data and associated information.
    #We determine how to sort the resulting lists in time:
    sorting = np.argsort(mjd)

    #If s1ds were read, we now sort them and save them to the output folder:
    if read_s1d == True:
        print('',end='\r')
        print('')
        print(f'---Saving S1D files to {str(outpath/"s1ds.pkl")}')
        s1dsorting = np.argsort(s1dmjd)
        if len(sorting) != len(s1dsorting):
            raise ValueError("in read_e2ds: Sorted science frames and sorted s1d frames are not of "
                "the same length. Telluric correction can't proceed. Make sure that the number of "
                "files of each type is correct.")
        s1dhdr_sorted=[]
        s1d_sorted=[]
        wave1d_sorted=[]
        for i in range(len(s1dsorting)):
            s1dhdr_sorted.append(s1dhdr[s1dsorting[i]])
            s1d_sorted.append(s1d[s1dsorting[i]])
            wave1d_sorted.append(wave1d[s1dsorting[i]])
            #Sort the s1d files for application of molecfit.
        #Store the S1Ds in a pickle file. I choose not to use a FITS image because the dimensions
        #of the s1ds can be different from one exposure to another.
        with open(outpath/'s1ds.pkl', 'wb') as f: pickle.dump([s1dhdr_sorted,s1d_sorted,
            wave1d_sorted],f)

    #Provide diagnostic output.
    ut.tprint('---These are the filetypes and observing date-times recognised.')
    ut.tprint('---These should all be SCIENCE and in chronological order.')
    for i in range(len(sorting)): print(f'------{obstype[sorting[i]]}  {date[sorting[i]]}  '
        f'{mjd[sorting[i]]}')


    #If we run diagnostic cross-correlations, we prepare to store output:
    if measure_RV:
        list_of_orders=[]
        list_of_waves=[]
    #Now we loop over all exposures and collect the i-th order from each exposure,
    #put these into a new matrix and save them to FITS images:
    f=open(outpath/'obs_times','w',newline='\n')

    headerline = 'MJD \t DATE \t EXPTIME \t MEAN AIRMASS \t BERV (km/s) \t FILE NAME'

    if mode == 'ESPRESSO':
        headerline+=' \t SNR550'

    n_1d = 0 #The number of orders for which the wavelength axes are all the same.
    #This should be equal to norders. If it is not, we will trigger a warning.
    for i in range(norders):
        order = np.zeros((len(sorting),npx))
        wave_order = copy.deepcopy(order)
        # print('CONSTRUCTING ORDER %s' % i)
        print(f"---Constructing order {i}", end="\r")
        for j in range(len(sorting)):#Loop over exposures
            exposure = e2ds[sorting[j]]
            wave_exposure = wave[sorting[j]]
            order[j,:] = exposure[i,:]
            wave_order[j,:] = wave_exposure[i,:]
            #Now I also need to write it to file.
            if i ==0:#Only do it the first time, not for every order.
                if mode == 'ESPRESSO':
                    line = (str(mjd[sorting[j]])+'\t'+date[sorting[j]]+'\t'+str(texp[sorting[j]])+'\t'+
                    str(np.round(airmass[sorting[j]],3))+'\t'+str(np.round(berv[sorting[j]],5))+'\t'
                    +framename[sorting[j]]+'\t'+str(np.round(SNR[sorting[j]],3))+'\n')
                else:
                    line = (str(mjd[sorting[j]])+'\t'+date[sorting[j]]+'\t'+str(texp[sorting[j]])+'\t'+
                    str(np.round(airmass[sorting[j]],3))+'\t'+str(np.round(berv[sorting[j]],5))+'\t'
                    +framename[sorting[j]]+'\n')
                f.write(line)


        if mode in ['UVES-red','UVES-blue','ESPRESSO']:#UVES and ESPRESSO pipelines pad with zeroes.
            #We remove these.
            # order[np.abs(order)<1e-10*np.nanmedian(order)]=0.0#First set very small values to zero.
            npx_order = np.shape(order)[1]
            sum=np.nansum(np.abs(order),axis=0)#Anything that is zero in this sum (i.e. the edges)
            #will be clipped.
            leftsize=npx_order-len(np.trim_zeros(sum,'f'))#Size of zero-padding on the left
            rightsize=npx_order-len(np.trim_zeros(sum,'b'))#Size of zero-padding on the right
            order = order[:,leftsize:npx_order-rightsize-1]
            wave_order=wave_order[:,leftsize:npx_order-rightsize-1]


        #Finally, we check whether or not the wavelength axes are in fact all the same. If they are,
        #we save only the first row:
        first_wl = np.tile(wave_order[0],(len(sorting),1))

        if measure_RV:
            list_of_orders.append(order)
            list_of_waves.append(wave_order)


        if np.allclose(wave_order,first_wl,rtol=2e-8):#If the wavelenght differences are smaller
        #than 1 in 50,000,000, i.e. R = 50 million, i.e delta-v = 6 m/s. Quite stringent here.
            n_1d+=1
            wave_order = wave_order[0]



        fits.writeto(outpath/('order_'+str(i)+'.fits'),order,overwrite=True)
        fits.writeto(outpath/('wave_'+str(i)+'.fits'),wave_order,overwrite=True)

    f.close()
    if n_1d > 0 and n_1d != norders:
        ut.tprint('---WARNING: In some orders, the wavelength axes were equal for all exposures, '
        f'but not for all orders ({n_1d} vs {norders}). For the orders where the wavelength axes '
        'were the same, the wavelength axis was saved only once. For the others, it was saved as '
        '2D arrays with shapes equal to the corresponding orders. This is not really a problem for '
        'tayph or for molecfit, but it may indicate that numerical errors are slipping through, or '
        'that there is an anomaly with the way the wavelengths are read from the FITS headers. '
        'You are advised to investigate this.')
    print('\n \n \n')
    ut.tprint(f'---Time-table written to {outpath/"obs_times"}.')


    if config:
        keywords=['P\t','a\t','aRstar\t','Mp\t','Rp\t','K\t','RpRstar\t','vsys\t',
        'RA\t-00:00:00.0','DEC\t-00:00:00.0','Tc\t','inclination\t','vsini\t']
        if mode in ['ESPRESSO']:
            keywords+=['resolution\t120000','long\t-70.4039','lat\t-24.6272','elev\t2635.0',
            'air\tTrue']
        if mode in ['UVES-red','UVES-blue']:
            keywords+=['resolution\t','long\t-70.4039','lat\t-24.6272','elev\t2635.0','air\tTrue']
        elif mode =='HARPS':
            keywords+=['resolution\t115000','long\t-70.7380','lat\t-29.2563','elev\t2387.2',
            'air\tTrue']
        elif mode =='NIRPS':
            keywords+=['resolution\t80000','long\t-70.7380','lat\t-29.2563','elev\t2387.2',
            'air\tTrue']
        elif mode in ['HARPSN','HARPS-N']:
            keywords+=['resolution\t115000','long\t-17.8850','lat\t28.7573','elev\t2396.0',
            'air\tTrue']
        elif mode in ['CARMENES-VIS','CARMENES-NIR']:
            keywords+=['resolution\t80000','long\t-2.5468','lat\t37.2208','elev\t2168.0','air\t']
        else:
            keywords+=['resolution\t','long\t','lat\t','elev\t','air\t']

        with open(outpath/'config_empty','w',newline='\n') as f:
            for keyword in keywords:
                f.write(keyword+'\n')
        ut.tprint(f'---Dummy config file written to {outpath/"config_empty"}.')


    print('\n \n \n')
    print('Read_e2ds completed successfully.')
    print('')



































































def measure_RV(dp,instrument='HARPS',star='solar',save_figure=True,air=True,air1D=True,
    ignore_s1d=False,parallel=True):
    """
    Does a rudimentary cleaning (de-colour, outlier correction) of data read by read_e2ds
    and performs cross-correlations with solar and telluric templates. Optionally attempts to
    re-align spectra to a common rest-frame.

    Set air to False to assume that the 2D orders are in vaccuum. Vactoair will be applied.
    Set air1D to False to assume that the 1D orders are in vaccuum. Vactoair will be applied.
    """

    import pdb
    import sys
    import copy
    import math
    import pickle
    import pathlib
    import warnings
    import numpy as np
    from pathlib import Path
    import astropy.io.fits as fits
    import matplotlib.pyplot as plt
    from contextlib import suppress
    import astropy.constants as const
    import scipy.interpolate as interp
    from astropy.utils.data import download_file

    import tayph.util as ut
    from tayph.ccf import xcor
    import tayph.models as models
    import tayph.functions as fun
    import tayph.operations as ops
    import tayph.masking as masking
    import tayph.system_parameters as sp
    from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
    from tayph.phoenix import get_phoenix_wavelengths, get_phoenix_model_spectrum
    from tayph.vartests import typetest,notnegativetest,nantest,postest,typetest_array,dimtest,lentest

    if parallel:
        from joblib import Parallel, delayed
        import multiprocessing
        ncores = int(multiprocessing.cpu_count())

    typetest(star,str,'star in measure_RV')
    typetest(instrument,str,'instrument in measure_RV')
    typetest(save_figure,bool,'save_figure in measure_RV')
    typetest(air,bool,'air in measure_RV')
    typetest(air1D,bool,'air1D in measure_RV')
    dp = ut.check_path(dp,exists=True)

    mode=instrument




    #Define the paths to the stellar and telluric templates if RV's need to be measured.
    if star.lower()=='solar' or star.lower() =='medium':
        T_eff = 6000
    elif star.lower() =='hot' or star.lower() =='warm':
        T_eff = 9000
    elif star.lower() == 'cool' or star.lower() == 'cold':
        T_eff = 4000
    else:
        warnings.warn(f"in read_e2ds: The star keyword was set to f{star} but only solar, hot "
            "or cold are allowed. Assuming a solar template.",RuntimeWarning)
        T_eff = 6000

    ut.tprint(f'---Downloading / reading diagnostic telluric and {star} PHOENIX models.')

    #Load the PHOENIX spectra from the Goettingen server:
    fxm = get_phoenix_model_spectrum(T_eff=T_eff)
    wlm = get_phoenix_wavelengths()/10.0#Angstrom to nm.

    #Load the telluric spectrum from my Google Drive or the cache:
    wlt,fxt = models.get_telluric()
    #Continuum-subtract the telluric model
    wlte,fxte=ops.envelope(wlt,fxt,2.0,selfrac=0.05,threshold=0.8)
    e_t = interp.interp1d(wlte,fxte,fill_value='extrapolate')(wlt)
    fxtn = fxt/e_t
    fxtn[fxtn>1]=1.0

    wlt=np.concatenate((np.array([100,200,300,400]),wlt))
    wlt=ops.vactoair(wlt)
    fxtn=np.concatenate((np.array([1.0,1.0,1.0,1.0]),fxtn))
    fxtn[wlt<500.1]=1.0

    #Continuum-subtract the PHOENIX model
    wlme,fxme=ops.envelope(wlm,fxm,5.0,selfrac=0.005)
    e_s = interp.interp1d(wlme,fxme,fill_value='extrapolate')(wlm)
    fxmn=fxm/e_s
    fxmn[fxmn>1.0]=1.0
    fxmn[fxmn<0.0]=0.0

    fxmn=fxmn[wlm>300.0]
    wlm=wlm[wlm>300.0]
    wlm=ops.vactoair(wlm)#Everything in the world is in air.
    #The stellar and telluric templates are now ready for application in cross-correlation at
    #the very end of this script.


    berv = sp.berv(dp)#This is used later to plot, but also to determine the number of exposures for
    #the filter width.
    airmass = sp.airmass(dp)


    npx_window=400#Total number of pixels that are included in a sliding window when measuring
    #the standard deviation or the MAD. I.e. the sample size.
    filter_width=np.max([np.min([math.ceil(npx_window/len(berv)),100]),10])#Make the window
    #dependent on how many exposures there are, such that the block has a total number of npx_window
    #pixels. Don't make the window wider than 100, and don't make it smaller than 10 either (in case
    #there are many spectra).



    if star =='hot':
        RVrange=500.0
        drv=2.0
    else:
        RVrange=250.0
        drv=1.0

    ut.tprint(f'---Preparing for diagnostic correlation with a telluric template and a {star} PHOENIX model.')
    ut.tprint(f'---The wavelength axes of the spectra will be shown here as they were saved by read_e2ds.')











    #After setup, we continue by reading in the data. First the 2D, and then the 1D if possible.
    ut.tprint(f'---Loading 2D orders from {dp}.')
    list_of_waves=[]
    list_of_orders=[]

    #Read in the file names from the datafolder.
    filelist_orders= [str(i) for i in dp.glob('order_*.fits')]
    if len(filelist_orders) == 0:#If no order FITS files are found:
        raise Exception(f'Runtime error: No orders_*.fits files were found in {dp}.')
    try:
        order_numbers = [int(i.split('order_')[1].split('.')[0]) for i in filelist_orders]
    except:
        raise Exception('Runtime error: Failed at casting fits filename numerals to ints. Are '
        'the filenames of all of the spectral orders correctly formatted (e.g. order_5.fits)?')
    order_numbers.sort()#This is the ordered list of numerical order IDs.
    n_orders = len(order_numbers)
    if n_orders == 0:
        raise Exception(f'Runtime error: n_orders may never have ended up as zero? ({n_orders})')


    for i in order_numbers:
        wavepath = ut.check_path(dp/f'wave_{i}.fits',exists=True)
        orderpath= ut.check_path(dp/f'order_{i}.fits',exists=True)
        wave_order = ut.readfits(wavepath)#2D or 1D?
        order_i = ut.readfits(orderpath)
        #Check dimensionality of wave axis and order. Either 2D or 1D.
        if wave_order.ndim == 2:
            if i == np.min(order_numbers):
                ut.tprint('------Wavelength axes provided are 2-dimensional.')
            n_px = np.shape(wave_order)[1]#Pixel width of the spectral order.
            dimtest(wave_order,np.shape(order_i),'wave_order in run.measure_RV()')
            #Need to have same shape.
        elif wave_order.ndim == 1:
            if i == np.min(order_numbers):
                ut.tprint('------Wavelength axes provided are 1-dimensional.')
            n_px = len(wave_order)
            dimtest(order_i,[0,n_px],f'order {i} in run.measure_RV()')
        else:
            raise Exception(f'Wavelength axis of order {i} is neither 1D nor 2D.')

        if i == np.min(order_numbers):
            n_exp = np.shape(order_i)[0]#For the first order, we fix n_exp.
            #All other orders should have the same n_exp.
            ut.tprint(f'------{n_exp} exposures recognised.')
        else:
            dimtest(order_i,[n_exp,n_px],f'order {i} in run_instance()')

        list_of_orders.append(order_i)

        #Deal with air or vaccuum wavelengths:
        if air == True:
            list_of_waves.append(copy.deepcopy(wave_order))
        else:
            if i == np.min(order_numbers):
                ut.tprint("------Applying vactoair correction.")
            list_of_waves.append(ops.vactoair(wave_order))

        list_of_orders_initial=copy.deepcopy(list_of_orders)
        list_of_waves_initial=copy.deepcopy(list_of_waves)#Save these for later.










    #Reading in the 1D data:
    s1d_path = dp/Path('s1ds.pkl')

    if s1d_path.exists() and not ignore_s1d:
        read_s1d=True
    else:
        read_s1d=False
    if read_s1d:
        print(f'---Presence of 1D spectra detected.')
        print(f'---Reading 1D spectra.')
        with open(s1d_path,"rb") as p:
            s1dhdr_sorted,s1d_sorted,wave1d_sorted = pickle.load(p)

        s1d_sorted_initial = copy.deepcopy(s1d_sorted)
        wave1d_sorted_initial = copy.deepcopy(wave1d_sorted)#Save these for later.





    #Here follow functions for cleaning the S1D and E2DS data. These will be called multiple times
    #if RV-correction is going to be done.
    def clean_s1d(wave1d_sorted,s1d_sorted,filter_width,parallel=False):
        """
        wave1d_sorted goes in angstroms here!
        """
        print('---Cleaning 1D spectra')
        wave_1d = wave1d_sorted[0]/10.0#Berv-un-corrected wavelength axis in nm in air.
        s1d_block=np.zeros((len(s1d_sorted),len(wave1d_sorted[0])))
        for i in range(0,len(s1d_sorted)):
            s1d_block[i]=interp.interp1d(wave1d_sorted[i]/10.0,s1d_sorted[i],bounds_error=False,
            fill_value='extrapolate')(wave_1d)
            #
            # plt.plot(wave_1d,s1d[0]/np.mean(s1d[0]),linewidth=0.5,label='Berv-un-corrected in nm.')
            # plt.plot(wave1d[1]/10.0,s1d[1]/np.mean(s1d[1]),linewidth=0.5,label=('Original '
            #'(bervcorrected) from A to nm.'))
            # plt.plot(wlt,fxtn,label='Telluric')
            # plt.legend()
            # plt.show()
            # pdb.set_trace()
        wave_1d,s1d_block,r1,r2=ops.clean_block(wave_1d,s1d_block,deg=4,verbose=True,renorm=False,
        w=filter_width,parallel=parallel)
        print(f"---Interpolating over NaN's")
        s1d_block=masking.interpolate_over_NaNs([s1d_block])[0]

        if not air1D:
            wave_1d = ops.vactoair(wave_1d)
        return(wave_1d,s1d_block)



    def clean_e2ds(list_of_waves,list_of_orders,filter_width,parallel=False):
        print(f'---Cleaning 2D orders for cross-correlation.')
        list_of_orders_trimmed=[]
        list_of_waves_trimmed=[]
        list_of_residuals=[]

        #For cleaning the 2D orders in parallel, I need to do a lot of error-catching because joblib
        #can be flaky. First define the for-looped function that cleans a single order. We'll loop over
        #orders to parallelise.
        def clean_order(i):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if np.ndim(list_of_waves[i]) == 2:
                    w_i = list_of_waves[i][0]
                    o_i = np.zeros((len(list_of_orders[i]),len(w_i)))
                    for j in range(len(list_of_orders[i])):
                        o_i[j]=interp.interp1d(list_of_waves[i][j],list_of_orders[i][j],
                        bounds_error=False,fill_value='extrapolate')(w_i)
                else:
                    w_i = list_of_waves[i]
                    o_i = list_of_orders[i]
                try:
                    w_trimmed,o_trimmed,r1,r2=ops.clean_block(w_i,o_i,w=filter_width,deg=4,renorm=True,
                    parallel=False)#SET PARALLEL TO FALSE BECAUSE WE MAY ALREDY BE IN PARALLEL
                except:
                    pdb.set_trace()
                if not parallel: ut.statusbar(i,len(list_of_orders))
                return(w_trimmed,o_trimmed,r2)

        #This little function of nothing will be needed later to seed a Parallel object to flush an
        #error.
        def void(i):
            return(i)

        #CARRY OUT THIS TIMING ON THE SERVER WITH LOKY BACKEND:
        # t1=ut.start()
        # list_of_waves_trimmed, list_of_orders_trimmed, list_of_residuals = zip(
        # *Parallel(n_jobs=ncores,backend="loky")(delayed(clean_order)(i) for i in range(len(list_of_orders))))
        # ut.end(t1)
        # t2=ut.start()
        # list_of_waves_trimmed, list_of_orders_trimmed, list_of_residuals = zip(
        # *[clean_order(i) for i in range(len(list_of_orders))])
        # ut.end(t2)
        # pdb.set_trace()

        if parallel:
            try:
                with Parallel(n_jobs=ncores,backend='loky') as P:
                    list_of_waves_trimmed, list_of_orders_trimmed, list_of_residuals = zip(
                    *P(delayed(clean_order)(i) for i in range(len(list_of_orders))))
            except:
                print('------Warning: Joblib triggered an error. Changing backend to threading.')
                #On my 8-core laptop, this yields only a 25% speedup compared to serial.
                #Under normal circumstances however, the loky backend will fare well, though.
                try:
                    with suppress(OSError, AttributeError):#This here is needed to flush out an OSError
                    #that is triggered by the failure of the loky backend above. I spawn a Parallel
                    #object that I let do nothing, supressing the error. Then the computation can
                    #continue with a new series of Parallel objects with the threading backend.
                    #This is a bit hacky, but hey...
                        with Parallel(n_jobs=ncores) as P:
                            a=P(delayed(void)(i) for i in range(len(list_of_orders)))
                    with Parallel(n_jobs=ncores,backend="threading") as P:
                        list_of_waves_trimmed, list_of_orders_trimmed, list_of_residuals = zip(
                        *P(delayed(clean_order)(i) for i in range(len(list_of_orders))))
                    print("------If you see an OSError here but the code keeps going, hang in there.")
                except:
                    print('------Warning: Unable to dislodge joblib by changing backend. Continuing '
                    'this computation in serial.')
                    with suppress(OSError, AttributeError):#I flush it again if there was a second fail.
                        with Parallel(n_jobs=ncores) as P:
                            a=P(delayed(void)(i) for i in range(len(list_of_orders)))
                    list_of_waves_trimmed, list_of_orders_trimmed, list_of_residuals = zip(
                    *[clean_order(i) for i in range(len(list_of_orders))])
        else:
            list_of_waves_trimmed, list_of_orders_trimmed, list_of_residuals = zip(
            *[clean_order(i) for i in range(len(list_of_orders))])
        #Done calling 2D order cleaning in parallel...

        print(f"---Interpolating over NaN's")
        list_of_orders_trimmed = masking.interpolate_over_NaNs(list_of_orders_trimmed)
        #Slow, also needs parallelising.
        return(list_of_waves_trimmed,list_of_orders_trimmed,list_of_residuals)


    def fit_ccf(rv,ccfs,mode='star',parallel=parallel,degree=2):
        if mode.lower() == 'star':
            def fit_stellar(i):
                spec = i/np.nanmean(i)
                fit = fun.fit_rotation_broadened_line(rv,spec,degree=degree)
                # if not parallel: ut.statusbar(i,len(ccf))
                return(fit[1],fit[2],fit[3],fit)

            #DO TIMING AGAIN ON THE SERVER!
            if parallel:#On my 8-core laptop, this yields a factor 2 speedup.
                # t1=ut.start()
                with Parallel(n_jobs=ncores) as P:
                    centroids, vsini, E2d, fit = zip(*P(delayed(fit_stellar)(i)
                for i in ccfs))
                # ut.end(t1)
            else:
                # t2=ut.start()
                centroids, vsini, E2d, fit = zip(*[fit_stellar(i) for i in ccfs])
                # ut.end(t2)
            return(centroids,vsini,E2d,fit)
        if mode.lower() == 'tel' or mode.lower() == 'telluric' or mode.lower() == 'tellurics':
            centroids=[]
            widths=[]
            fitparams = []
            for n,i in enumerate(ccfs):
                fit,void = fun.gaussfit(rv,i/np.nanmean(i),startparams=[-0.9,rv[np.argmin(i)],
                3.0,1.0,0.0,0.0])
                centroids.append(fit[1])
                widths.append(fit[2])
                fitparams.append(fit)
            return(centroids,widths,fitparams)



    def shift_spec_2D(low,dv):
        """
        The wavelength is shifted by an amount dv. If list_of_waves[i] is 1D, it is expanded
        into a 2D array. Dv represents the velocities of a time-series.
        """
        list_of_waves_shifted=[]
        for i in range(len(low)):
            wl=low[i]
            if np.ndim(wl) == 1:
                wl=np.tile(wl,(len(dv),1))#broadcast in vertical direction.
            dimtest(wl,[len(dv),0],'list_of_waves in shift_spec_2D in measure_RV')
            list_of_waves_shifted.append((wl.T*(1+dv/const.c.to('km/s').value)).T)
        return(list_of_waves_shifted)


    def shift_spec_1D(wave1d,dv):
        """
        wave1d corresponds to wave1d sorted. It is therefore a list of wavelengths with length
        equal to dv
        """
        lentest(wave1d,len(dv),'wave1d in shift_spec_1D in measure_RV')
        wave_shifted=[]
        for i in range(len(wave1d)):
            wave_shifted.append(wave1d[i]*(1+dv[i]/const.c.to('km/s').value))
        return(wave_shifted)


    class Theprocessor(object):
        """
        This object can only live inside this script because it uses a TON of global variables and
        functions. And it also changes them.
        If you want to revamp this in the future: Lots comes from global so good luck.
        """
        def __init__(self):
            self.list_of_waves = copy.deepcopy(list_of_waves)
            self.list_of_orders = copy.deepcopy(list_of_orders)
            if read_s1d:
                self.wave1d_sorted = copy.deepcopy(wave1d_sorted)
                self.s1d_sorted = copy.deepcopy(s1d_sorted)
            self.vsys = 0
            self.offset = 0
            self.fit_done = 0
            self.frame = 'star'#Shit to star by default. Altered by radiobutton.
            self.apply = 'both'
            self.source = '2D'
            self.bervmult = 0.0#Add no BERV by default. Altered by radiobutton.
            self.buttontrigger = 0
            self.degree = 2 #Degree of continuum polynomial fit for stellar line. Add support for
            #future freedom in this?
            self.adv_trigger = False
            self.centroids2d = None
            self.centroidsT2d = None
            self.recompute()
            self.fit()
            self.save_output()#Save the computed parameters.
            self.setup_figure()
            self.plot_data()
            self.decorate_figure()
            self.fig.tight_layout()
        def save_fit_result(self):
            """
            This saves the computed output to backup variables. Typically used to save the initial
            run such that we can operate a Reset button.
            """
            self.list_of_waves_trimmed_SAV2 = copy.deepcopy(self.list_of_waves_trimmed)
            self.list_of_orders_trimmed_SAV2 = copy.deepcopy(self.list_of_orders_trimmed)
            self.list_of_residuals_SAV2 = copy.deepcopy(self.list_of_residuals)
            if read_s1d:
                self.wave_1d_SAV2 = copy.deepcopy(self.wave_1d)
                self.s1d_block_SAV2 = copy.deepcopy(self.s1d_block)
            self.rv_SAV2 = copy.deepcopy(self.rv)
            self.ccf_SAV2 =  copy.deepcopy(self.ccf)
            self.rvT2D_SAV2 = copy.deepcopy(self.rvT2D)
            self.ccfT2D_SAV2 = copy.deepcopy(self.ccfT2D)
            if read_s1d:
                self.rv1d_SAV2 = copy.deepcopy(self.rv1d)
                self.ccf1d_SAV2 = copy.deepcopy(self.ccf1d)
                self.rvT1d_SAV2 = copy.deepcopy(self.rvT1d)
                self.ccfT1d_SAV2 = copy.deepcopy(self.ccfT1d)
            self.centroids2d_SAV2 = copy.deepcopy(self.centroids2d)
            self.vsini2d_SAV2 = copy.deepcopy(self.vsini2d)
            self.E2d_SAV2 = copy.deepcopy(self.E2d)
            self.fit2d_SAV2 = copy.deepcopy(self.fit2d)
            self.centroidsT2d_SAV2 = copy.deepcopy(self.centroidsT2d)
            self.widthsT2d_SAV2 = copy.deepcopy(self.widthsT2d)
            self.fitT2d_SAV2 = copy.deepcopy(self.fitT2d)
            if read_s1d:
                self.centroids1d_SAV2 = copy.deepcopy(self.centroids1d)
                self.centroidsT1d_SAV2 = copy.deepcopy(self.centroidsT1d)
        def save_output(self):
            """
            This saves the computed output to backup variables. Typically used to save the initial
            run such that we can operate a Reset button.
            """
            self.list_of_waves_trimmed_SAV = copy.deepcopy(self.list_of_waves_trimmed)
            self.list_of_orders_trimmed_SAV = copy.deepcopy(self.list_of_orders_trimmed)
            self.list_of_residuals_SAV = copy.deepcopy(self.list_of_residuals)
            if read_s1d:
                self.wave_1d_SAV = copy.deepcopy(self.wave_1d)
                self.s1d_block_SAV = copy.deepcopy(self.s1d_block)
            self.rv_SAV = copy.deepcopy(self.rv)
            self.ccf_SAV =  copy.deepcopy(self.ccf)
            self.rvT2D_SAV = copy.deepcopy(self.rvT2D)
            self.ccfT2D_SAV = copy.deepcopy(self.ccfT2D)
            if read_s1d:
                self.rv1d_SAV = copy.deepcopy(self.rv1d)
                self.ccf1d_SAV = copy.deepcopy(self.ccf1d)
                self.rvT1d_SAV = copy.deepcopy(self.rvT1d)
                self.ccfT1d_SAV = copy.deepcopy(self.ccfT1d)
            self.centroids2d_SAV = copy.deepcopy(self.centroids2d)
            self.vsini2d_SAV = copy.deepcopy(self.vsini2d)
            self.E2d_SAV = copy.deepcopy(self.E2d)
            self.fit2d_SAV = copy.deepcopy(self.fit2d)
            self.centroidsT2d_SAV = copy.deepcopy(self.centroidsT2d)
            self.widthsT2d_SAV = copy.deepcopy(self.widthsT2d)
            self.fitT2d_SAV = copy.deepcopy(self.fitT2d)
            if read_s1d:
                self.centroids1d_SAV = copy.deepcopy(self.centroids1d)
                self.centroidsT1d_SAV = copy.deepcopy(self.centroidsT1d)
        def load_fit_result(self):
            """
            This loads the computed output from backup variables. Typically used to save the initial
            run such that we can operate a Reset button.
            """
            self.list_of_waves_trimmed = copy.deepcopy(self.list_of_waves_trimmed_SAV2)
            self.list_of_orders_trimmed = copy.deepcopy(self.list_of_orders_trimmed_SAV2)
            self.list_of_residuals = copy.deepcopy(self.list_of_residuals_SAV2)
            if read_s1d:
                self.wave_1d = copy.deepcopy(self.wave_1d_SAV2)
                self.s1d_block = copy.deepcopy(self.s1d_block_SAV2)
            self.rv = copy.deepcopy(self.rv_SAV2)
            self.ccf =  copy.deepcopy(self.ccf_SAV2)
            self.rvT2D = copy.deepcopy(self.rvT2D_SAV2)
            self.ccfT2D = copy.deepcopy(self.ccfT2D_SAV2)
            if read_s1d:
                self.rv1d = copy.deepcopy(self.rv1d_SAV2)
                self.ccf1d = copy.deepcopy(self.ccf1d_SAV2)
                self.rvT1d = copy.deepcopy(self.rvT1d_SAV2)
                self.ccfT1d = copy.deepcopy(self.ccfT1d_SAV2)
            self.centroids2d = copy.deepcopy(self.centroids2d_SAV2)
            self.vsini2d = copy.deepcopy(self.vsini2d_SAV2)
            self.E2d = copy.deepcopy(self.E2d_SAV2)
            self.fit2d = copy.deepcopy(self.fit2d_SAV2)
            self.centroidsT2d = copy.deepcopy(self.centroidsT2d_SAV2)
            self.widthsT2d = copy.deepcopy(self.widthsT2d_SAV2)
            self.fitT2d = copy.deepcopy(self.fitT2d_SAV2)
            if read_s1d:
                self.centroids1d = copy.deepcopy(self.centroids1d_SAV2)
                self.centroidsT1d = copy.deepcopy(self.centroidsT1d_SAV2)
        def load_output(self):
            """
            This loads the computed output from backup variables. Typically used to save the initial
            run such that we can operate a Reset button.
            """
            self.list_of_waves_trimmed = copy.deepcopy(self.list_of_waves_trimmed_SAV)
            self.list_of_orders_trimmed = copy.deepcopy(self.list_of_orders_trimmed_SAV)
            self.list_of_residuals = copy.deepcopy(self.list_of_residuals_SAV)
            if read_s1d:
                self.wave_1d = copy.deepcopy(self.wave_1d_SAV)
                self.s1d_block = copy.deepcopy(self.s1d_block_SAV)
            self.rv = copy.deepcopy(self.rv_SAV)
            self.ccf =  copy.deepcopy(self.ccf_SAV)
            self.rvT2D = copy.deepcopy(self.rvT2D_SAV)
            self.ccfT2D = copy.deepcopy(self.ccfT2D_SAV)
            if read_s1d:
                self.rv1d = copy.deepcopy(self.rv1d_SAV)
                self.ccf1d = copy.deepcopy(self.ccf1d_SAV)
                self.rvT1d = copy.deepcopy(self.rvT1d_SAV)
                self.ccfT1d = copy.deepcopy(self.ccfT1d_SAV)
            self.centroids2d = copy.deepcopy(self.centroids2d_SAV)
            self.vsini2d = copy.deepcopy(self.vsini2d_SAV)
            self.E2d = copy.deepcopy(self.E2d_SAV)
            self.fit2d = copy.deepcopy(self.fit2d_SAV)
            self.centroidsT2d = copy.deepcopy(self.centroidsT2d_SAV)
            self.widthsT2d = copy.deepcopy(self.widthsT2d_SAV)
            self.fitT2d = copy.deepcopy(self.fitT2d_SAV)
            if read_s1d:
                self.centroids1d = copy.deepcopy(self.centroids1d_SAV)
                self.centroidsT1d = copy.deepcopy(self.centroidsT1d_SAV)
        def recompute(self):
            #Function definitions are finished, let's proceed to execute them:

            if self.apply in ['2D','both']:
                self.list_of_waves_trimmed,self.list_of_orders_trimmed,self.list_of_residuals = clean_e2ds(self.list_of_waves,
                self.list_of_orders,filter_width,parallel=parallel)

            if read_s1d and self.apply in ['1D','both']:
                self.wave_1d,self.s1d_block=clean_s1d(self.wave1d_sorted,self.s1d_sorted,filter_width,parallel=parallel)

            print(f'---Performing cross-correlation')
            if self.apply in ['2D','both']:
                self.rv,self.ccf,Tsums=xcor(self.list_of_waves_trimmed,self.list_of_orders_trimmed,[wlm],[fxmn-1.0],drv,RVrange)
                self.rvT2D,self.ccfT2D,TsusmT2D=xcor(self.list_of_waves_trimmed,self.list_of_orders_trimmed,[wlt],[fxtn-1.0],
                drv,RVrange)
                self.ccf=self.ccf[0]
                self.ccfT2D=self.ccfT2D[0]

            if read_s1d and self.apply in ['1D','both']:
                self.rv1d,self.ccf1d,Tsums1d=xcor([self.wave_1d],[self.s1d_block],[wlm],[fxmn-1.0],drv,RVrange)
                self.rvT1d,self.ccfT,TsusmT=xcor([self.wave_1d],[self.s1d_block],[wlt],[fxtn-1.0],drv,RVrange)
                self.ccf1d=self.ccf1d[0]
                self.ccfT1d=self.ccfT[0]
        def fit(self):
            if self.apply in ['2D','both']:
                print('---Fitting 2D centroids')
                self.centroids2d, self.vsini2d, self.E2d, self.fit2d = fit_ccf(self.rv,self.ccf,mode='star',parallel=parallel,degree=self.degree)
                self.centroidsT2d, self.widthsT2d, self.fitT2d = fit_ccf(self.rvT2D,self.ccfT2D,mode='tel')

            if read_s1d and self.apply in ['1D','both']:
                print('---Fitting 1D centroids')
                self.centroids1d, void1, void2, self.fit1d = fit_ccf(self.rv1d,self.ccf1d,mode='star',parallel=parallel,degree=self.degree)
                self.centroidsT1d, void1, self.fitT1d = fit_ccf(self.rvT1d,self.ccfT1d,mode='tel',parallel=parallel)
            print('---Velocity computation complete')
        def plot_data(self):
            #The rest is for plotting.
            minwl=np.inf#For determining in the for-loop what the min and max wl range of the data are.
            maxwl=0.0
            mean_of_orders = np.nanmean(np.hstack(self.list_of_orders_trimmed))
            for i in range(len(self.list_of_orders)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s_avg = np.nanmean(self.list_of_orders_trimmed[i],axis=0)
                minwl=np.min([np.min(self.list_of_waves[i]),minwl])
                maxwl=np.max([np.max(self.list_of_waves[i]),maxwl])
                if i == 0:
                    self.ax[0].plot(self.list_of_waves_trimmed[i],s_avg/mean_of_orders,color='red',
                    linewidth=0.9,alpha=0.5,label='2D echelle orders to be '
                    'cross-correlated')
                else:
                    self.ax[0].plot(self.list_of_waves_trimmed[i],s_avg/mean_of_orders,color='red',linewidth=0.7,
                    alpha=0.5)

            if read_s1d:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s1d_avg=np.nanmean(self.s1d_block,axis=0)
                self.ax[0].plot(self.wave_1d,s1d_avg/np.nanmean(s1d_avg),color='orange',
                label='1D spectrum to be cross-correlated',linewidth=0.9,alpha=0.5)
            self.ax[0].set_xlim(minwl-5,maxwl+5)#Always nm.

            for n,i in enumerate(self.ccf):#Overplot onto all of the lines and spans.
                if n == 0:
                    self.ax[1].plot(self.rv,i/np.nanmean(i),linewidth=0.7,alpha=0.3,color='red',
                    label='2D orders w. PHOENIX')
                else:
                    self.ax[1].plot(self.rv,i/np.nanmean(i),linewidth=0.7,alpha=0.3,color='red')
                self.ax[1].axvline(self.centroids2d[n],color='red',alpha=1.0/len(self.ccf))
            for n,i in enumerate(self.ccfT2D):
                if n == 0:
                    self.ax[1].plot(self.rvT2D,i/np.nanmean(i),linewidth=0.7,alpha=0.3,color='royalblue',
                    label='2D orders w. TELLURIC')
                else:
                    self.ax[1].plot(self.rvT2D,i/np.nanmean(i),linewidth=0.7,alpha=0.3,color='royalblue')
                self.ax[1].axvline(self.centroidsT2d[n],color='royalblue',alpha=1.0/len(self.ccfT2D))

            if read_s1d:
                for n,i in enumerate(self.ccf1d):
                    if n == 0:
                        self.ax[1].plot(self.rv1d,i/np.nanmean(i),linewidth=0.7,alpha=0.3,color='orange',
                        label='1D spectra w. PHOENIX')
                    else:
                        self.ax[1].plot(self.rv1d,i/np.nanmean(i),linewidth=0.7,alpha=0.3,color='orange')
                    self.ax[1].axvline(self.centroids1d[n],color='orange',alpha=1.0/len(self.ccf))
                for n,i in enumerate(self.ccfT1d):
                    if n == 0:
                        self.ax[1].plot(self.rvT1d,i/np.nanmean(i),linewidth=0.7,alpha=0.3,color='lightseagreen',
                        label='1D spectra w. TELLURIC')
                    else:
                        self.ax[1].plot(self.rvT1d,i/np.nanmean(i),linewidth=0.7,alpha=0.3,color='lightseagreen')
                    self.ax[1].axvline(self.centroidsT1d[n],color='lightseagreen',alpha=1.0/len(self.ccfT1d))

            if star == 'hot':
                sf = 4
            else:
                sf = 2.5

            if self.source == '2D':

                self.Gw = []
                self.Lw = []
                self.Sw = []
                print('------Showing 2D centroids in middle-left panels.')
                for i in range(len(self.fit2d)):
                    print(self.fit2d[i][4],self.fit2d[i][5])
                    self.Gw.append(self.fit2d[i][4])
                    self.Lw.append(self.fit2d[i][5])
                    self.Sw.append(self.fit2d[i][2])
                # rv_hi = np.linspace(np.min(self.rv),np.max(self.rv),len(self.rv)*5)#This creates a weird error. Nvm then...
                rv_hi = self.rv
                p_ccfs = [self.ccf[0],self.ccf[int(len(self.ccf)/2)],self.ccf[-1]]#First, middle last, to plot.
                p_fits = [fun.rotation_broadened_line(rv_hi,*self.fit2d[0]),
                fun.rotation_broadened_line(rv_hi,*self.fit2d[int(len(self.ccf)/2)]),
                fun.rotation_broadened_line(rv_hi,*self.fit2d[-1])]
                min_centroid_plot = np.min([self.fit2d[0][1],self.fit2d[1][1],self.fit2d[2][1]])
                max_centroid_plot = np.max([self.fit2d[0][1],self.fit2d[1][1],self.fit2d[2][1]])
                max_vsini_plot = np.max([np.max([self.fit2d[0][2],self.fit2d[1][2],self.fit2d[2][2]])*sf,25])#15km/s
                p_ccfsT = [self.ccfT2D[0],self.ccfT2D[int(len(self.ccfT2D)/2)],self.ccfT2D[-1]]#First, middle last, to plot.
                rv_hiT = np.linspace(np.min(self.rvT2D),np.max(self.rvT2D),len(self.rvT2D)*5)
                p_fitsT = [fun.gaussian(rv_hiT,*self.fitT2d[0]),
                fun.gaussian(rv_hiT,*self.fitT2d[int(len(self.ccfT2D)/2)]),
                fun.gaussian(rv_hiT,*self.fitT2d[-1])]
                min_centroid_plotT = np.min([self.fitT2d[0][1],self.fitT2d[1][1],self.fitT2d[2][1]])
                max_centroid_plotT = np.max([self.fitT2d[0][1],self.fitT2d[1][1],self.fitT2d[2][1]])
                width_plotT = np.max([np.max([self.fitT2d[0][2],self.fitT2d[1][2],self.fitT2d[2][2]])*5,10])#10km/s
            if self.source == '1D' and read_s1d == True:
                print('------Showing 1D centroids in middle-left panels.')
                p_ccfs = [self.ccf1d[0],self.ccf1d[int(len(self.ccf1d)/2)],self.ccf1d[-1]]#First, middle last, to plot.
                # rv_hi = np.linspace(np.min(self.rv1d),np.max(self.rv1d),len(self.rv1d)*5)
                rv_hi = self.rv1d
                p_fits = [fun.rotation_broadened_line(rv_hi,*self.fit1d[0]),
                fun.rotation_broadened_line(rv_hi,*self.fit1d[int(len(self.ccf1d)/2)]),
                fun.rotation_broadened_line(rv_hi,*self.fit1d[-1])]
                min_centroid_plot = np.min([self.fit1d[0][1],self.fit1d[1][1],self.fit1d[2][1]])
                max_centroid_plot = np.max([self.fit1d[0][1],self.fit1d[1][1],self.fit1d[2][1]])
                max_vsini_plot = np.max([np.max([self.fit1d[0][2],self.fit1d[1][2],self.fit1d[2][2]])*sf,25])#15km/s
                p_ccfsT = [self.ccfT1d[0],self.ccfT1d[int(len(self.ccfT1d)/2)],self.ccfT1d[-1]]#First, middle last, to plot.
                rv_hiT = np.linspace(np.min(self.rvT1d),np.max(self.rvT1d),len(self.rvT1d)*5)
                p_fitsT = [fun.gaussian(rv_hiT,*self.fitT1d[0]),
                fun.gaussian(rv_hiT,*self.fitT1d[int(len(self.ccfT1d)/2)]),
                fun.gaussian(rv_hiT,*self.fitT1d[-1])]
                min_centroid_plotT = np.min([self.fitT1d[0][1],self.fitT1d[1][1],self.fitT1d[2][1]])
                max_centroid_plotT = np.max([self.fitT1d[0][1],self.fitT1d[1][1],self.fitT1d[2][1]])
                width_plotT = np.max([np.max([self.fitT1d[0][2],self.fitT1d[1][2],self.fitT1d[2][2]])*5,10])#10km/s

            #The plot that shows the stellar fits.
            self.ax[2].plot(self.rv,p_ccfs[0]/np.nanmean(p_ccfs[0]),'.',label='First',alpha=0.4,color='C0')
            self.ax[2].plot(self.rv,p_ccfs[1]/np.nanmean(p_ccfs[1]),'.',label='Middle',alpha=0.4,color='C1')
            self.ax[2].plot(self.rv,p_ccfs[2]/np.nanmean(p_ccfs[2]),'.',label='Last',alpha=0.4,color='C2')
            self.ax[2].plot(rv_hi,p_fits[0],'--',alpha=1,color='C0')
            self.ax[2].plot(rv_hi,p_fits[1],'--',alpha=1,color='C1')
            self.ax[2].plot(rv_hi,p_fits[2],'--',alpha=1,color='C2')
            self.ax[2].set_xlim(min_centroid_plot-max_vsini_plot,max_centroid_plot+max_vsini_plot)
            self.ax[3].plot(np.arange(len(self.centroids2d)),self.centroids2d,'.',label='E2DS centroids of star',color='red')
            self.ax[3].plot(np.arange(len(berv)),-1*(berv-np.mean(berv))+np.mean(self.centroids2d),'--',label='Barycentric RV (inv. & offset)',color='red')

            self.ax[4].plot(self.rvT2D,p_ccfsT[0]/np.nanmean(p_ccfsT[0]),'.',label='First',alpha=0.4,color='C0')
            self.ax[4].plot(self.rvT2D,p_ccfsT[1]/np.nanmean(p_ccfsT[1]),'.',label='Middle',alpha=0.4,color='C1')
            self.ax[4].plot(self.rvT2D,p_ccfsT[2]/np.nanmean(p_ccfsT[2]),'.',label='Last',alpha=0.4,color='C2')
            self.ax[4].plot(rv_hiT,p_fitsT[0],'--',alpha=1,color='C0')
            self.ax[4].plot(rv_hiT,p_fitsT[1],'--',alpha=1,color='C1')
            self.ax[4].plot(rv_hiT,p_fitsT[2],'--',alpha=1,color='C2')
            self.ax[4].set_xlim(min_centroid_plotT-width_plotT,max_centroid_plotT+width_plotT)
            self.ax[5].plot(np.arange(len(self.centroidsT2d)),self.centroidsT2d,'.',label='E2DS centroids of tellurics',color='royalblue')
            self.ax[5].plot(np.arange(len(berv)),1*(berv-np.mean(berv))+np.mean(self.centroidsT2d),'--',label='Barycentric RV (offset)',color='royalblue')

            if read_s1d:
                self.ax[6].plot(np.arange(len(self.centroids1d)),self.centroids1d,'.',label='S1D centroids of star',color='orange')
                self.ax[6].plot(np.arange(len(berv)),-1*(berv-np.mean(berv))+np.mean(self.centroids1d),'--',label='Barycentric RV (inv. & offset)',color='orange')
                self.ax[7].plot(np.arange(len(self.centroidsT1d)),self.centroidsT1d,'.',label='S1D centroids of tellurics',color='lightseagreen')
                self.ax[7].plot(np.arange(len(berv)),1*(berv-np.mean(berv))+np.mean(self.centroidsT1d),'--',label='Barycentric RV (offset)',color='lightseagreen')
            #The plot that shows the telluric fits.
            #Start the figure
        def setup_figure(self):
            if read_s1d:
                self.fig, axs = plt.subplot_mosaic([['top','top','top','top'],['lower left', 'middle middle',
                'middle right1','middle right2'],['lower left', 'lower middle','lower right1','lower right2']],figsize=(14,7))
                self.ax=[axs['top'],axs['lower left'],axs['middle middle'],axs['middle right1'],axs['lower middle'],axs['lower right1'],axs['middle right2'],axs['lower right2']]
            else:
                self.fig, axs = plt.subplot_mosaic([['top','top','top'],['lower left', 'middle middle',
                'middle right1'],['lower left', 'lower middle','lower right1']],figsize=(14,7))
                self.ax=[axs['top'],axs['lower left'],axs['middle middle'],axs['middle right1'],axs['lower middle'],axs['lower right1']]
        def decorate_figure(self):
            self.ax[0].set_title(f'Time-averaged spectral orders and {star} PHOENIX model.')
            self.ax[0].set_xlabel('Wavelength (nm)')
            self.ax[0].set_ylabel('Normalised flux')
            self.ax[0].legend(loc='upper right',fontsize=6)
            self.ax[0].tick_params(labelsize=10)
            self.ax[0].plot(wlm,fxmn,color='green',linewidth=0.7,label='PHOENIX template (air)',alpha=0.5)#These two are globals and dont change.
            self.ax[0].plot(wlt,fxtn,color='blue',linewidth=0.7,label='Skycalc telluric model (air)',alpha=0.6)
            self.ax[1].set_title(f'CCF between data with PHOENIX and telluric models',fontsize=9)
            self.ax[1].set_xlabel('Radial velocity (km/s)')
            self.ax[1].set_ylabel('Mean flux')
            legend=self.ax[1].legend(fontsize=6)
            for lh in legend.legendHandles:
                try:
                    lh._legmarker.set_alpha(1.0)
                except:
                    pass
            for lh in legend.get_lines():
                try:
                    lh.set_linewidth(2)
                except:
                    pass
            self.ax[2].set_xlabel('Radial velocity (km/s)')
            if self.source == '2D':
                self.ax[2].set_title('Fits of stellar line centroid (E2DS)',fontsize=9)
            if self.source == '1D' and read_s1d == True:
                self.ax[2].set_title('Fits of stellar line centroid (S1D)',fontsize=9)
            self.ax[2].legend(fontsize=6)
            self.ax[3].set_xlabel('Exposure number')
            self.ax[3].set_title('Evolution/drift of stellar centroid RV (E2DS)',fontsize=8)
            self.ax[3].legend(fontsize=6)
            self.ax[4].set_xlabel('Radial velocity (km/s)')
            if self.source == '2D':
                self.ax[4].set_title('Fits of telluric line centroid (E2DS)',fontsize=9)
            if self.source == '1D' and read_s1d == True:
                self.ax[4].set_title('Fits of telluric line centroid (S1D)',fontsize=9)
            self.ax[4].legend(fontsize=6)
            self.ax[5].legend(fontsize=6)
            self.ax[5].set_xlabel('Exposure number')
            self.ax[5].set_title('Evolution/drift of telluric centroid RV (E2DS)',fontsize=8)
            if read_s1d:
                self.ax[6].set_xlabel('Exposure number')
                self.ax[6].set_title('Evolution/drift of stellar centroid RV (S1D)',fontsize=8)
                self.ax[6].legend(fontsize=6)
                self.ax[7].legend(fontsize=6)
                self.ax[7].set_xlabel('Exposure number')
                self.ax[7].set_title('Evolution/drift of telluric centroid RV (S1D)',fontsize=8)
            plt.subplots_adjust(left=0.05)#Make them more tight, we need all the space we can get.
            plt.subplots_adjust(right=0.85)
        def replot(self):
            for a in self.ax:
                a.clear()
            self.plot_data()
            self.decorate_figure()
            self.fig.canvas.draw_idle()
        def align_spectra(self,event):
            """
            This here is the core functionality.
            It shifts the spectra and recomputes to show the result.
            The shifting is only a applied to a certain set of spectra (2D or 1D) if so chosen.
            The shifting can also be chosen to put the star in the restframe or the tellurics.
            The user may also choose to add/subtract the BERV, or to add a systemic velocity.
            """
            if self.source == '2D' and self.frame == 'star':
                cent = self.centroids2d
            elif self.source == '1D' and self.frame == 'star':
                cent = self.centroids1d
            elif self.source == '2D' and self.frame == 'tel':
                cent = self.centroidsT2d
            elif self.source == '1D' and self.frame == 'tel':
                cent = self.centroidsT1d

            #Reset to initial. We dont want cumulative shifts. But you can do one after the other..!
            self.load_output()


            self.offset = self.vsys+berv*self.bervmult

            if self.apply in ['2D','both']:
                # print(np.min(self.list_of_waves[0]),np.array(cent)*(-1),self.offset)
                self.list_of_waves = shift_spec_2D(self.list_of_waves,np.array(cent)*-1+self.offset)
            if read_s1d and self.apply in ['1D','both']:
                self.wave1d_sorted = shift_spec_1D(self.wave1d_sorted,np.array(cent)*-1+self.offset)
            self.recompute()
            self.fit()
            self.fit_done+=1
            self.save_fit_result()
            self.replot()

            # plt.close('all')#We continue by closing the figure, ending the callback.
            # self.shifted_spectra = shift_spec
            #AFTER SHIFTING, I NEED TO RERUN ALL OF THE ABOVE.
            #THAT MEANS THAT ALL CLEANING AND XCOR STEPS NEED TO BE PUT IN FUNCTIONS, TO BE
            #REPEATED HERE. LOTS OF WORK, BUT WILL SIMPLIFY THE CODE.
        def activate_back_to_fit_button(self):
            self.bbacktofit.color='lightgray'
            self.bbacktofit.on_clicked(callback.reset_to_fit)
        def replot_button(self,event):
            print('------Replotting')
            self.replot()
        def refit(self,event):
            self.fit()
            self.fit_done+=1
            self.save_fit_result()
            self.replot()
        def reset_to_fit(self,event):
            """Goes with the add_back_to_fit button"""
            print('------Resetting to fit result')
            self.fit_done+=1
            self.load_fit_result()
            self.replot()
        def reset_to_initial(self,event):
            print('------Resetting to initial result')
            if self.fit_done >0 and self.buttontrigger ==0:#only do this once.
                self.activate_back_to_fit_button()
                self.buttontrigger=1
            self.fit_done = 0#if closing in this state, this prevents writing files.
            self.load_output()
            self.replot()
        def go_to_adv(self,event):
            self.adv_trigger = True
            Npx = len(self.list_of_orders[0][0])
            for i in range(len(self.list_of_orders)):#Check that all orders are defined on the same
                #wavelength frame.
                if len(self.list_of_orders[i][0]) != Npx: #If that is not the case, cancel this.
                    self.adv_trigger = False
            if self.adv_trigger == False:
                ut.tprint('---Not all orders have the same length. This means that the original '
                'extraction is not preserved, and 2D wavelength correction cannot be performed.')
            else:
                ut.tprint('---Proceeding with 2D wavelength correction.')
                plt.close('all')
        def close(self,event):
            if self.fit_done > 0:
                print(f'---Saving to {dp}')
                for i in range(len(self.list_of_waves)):
                    fits.writeto(dp/('wave_'+str(i)+'.fits'),self.list_of_waves[i],overwrite=True)
                if read_s1d:
                    with open(dp/'s1ds.pkl', 'wb') as f: pickle.dump([s1dhdr_sorted,s1d_sorted,self.wave1d_sorted],f)
            else:
                print('---Accepting wavelength solutions without re-alignment')
            plt.close('all')

            print('---Exiting')






    callback=Theprocessor()

    if save_figure:
        plt.savefig(dp/'read_data.pdf', dpi=400)#Global dp.
    plt.subplots_adjust(left=0.05)#Make them more tight, we need all the space we can get.
    plt.subplots_adjust(right=0.85)


    if read_s1d:
        ax_source = callback.fig.add_axes([0.87, 0.85, 0.11, 0.1])
        ax_target = callback.fig.add_axes([0.87, 0.75, 0.11, 0.1])
    ax_addberv = callback.fig.add_axes([0.87, 0.65, 0.11, 0.1])
    ax_startel = callback.fig.add_axes([0.87, 0.55, 0.11, 0.1])

    # ax_slider1 = callback.fig.add_axes([0.87, 0.50, 0.11, 0.02])
    # ax_slider2 = callback.fig.add_axes([0.87, 0.45, 0.11, 0.02])
    ax_vsys = callback.fig.add_axes([0.93,0.45,0.05,0.05])
    ax_shift = callback.fig.add_axes([0.87, 0.35, 0.11, 0.07])
    ax_refit = callback.fig.add_axes([0.93, 0.27, 0.05, 0.07])  #Not yet implemented
    ax_replot = callback.fig.add_axes([0.87, 0.27, 0.05, 0.07]) #Not yet implemented
    ax_backtofit = callback.fig.add_axes([0.93, 0.19, 0.05, 0.07])
    ax_reset = callback.fig.add_axes([0.87, 0.19, 0.05, 0.07])
    ax_close  = callback.fig.add_axes([0.87, 0.11, 0.11, 0.07])
    ax_adv  = callback.fig.add_axes([0.87, 0.03, 0.11, 0.07])

    if read_s1d:
        ax_source.set_title('')
        ax_target.set_title('')
    ax_addberv.set_title('')
    ax_startel.set_title('')
    ax_vsys.set_title('')
    ax_shift.set_title('')
    ax_reset.set_title('')
    ax_close.set_title('')
    ax_adv.set_title('')

    clabels = ['Align to star', 'Align to telluric']
    clabels2 = ['No BERV', 'Subtract BERV']
    clabels3 = ['To both', 'To only E2DS', 'To only S1D']
    clabels4 = ['From E2DS', 'From S1D']

    def framechoice(label):
        index = clabels.index(label)
        if index == 0:
            print('---Align to star')
            callback.frame = 'star'
        if index == 1:
            print('---Align to tellurics')
            callback.frame = 'tel'

    def bervchoice(label):
        index = clabels2.index(label)
        if index == 0:
            print('---No berv')
            callback.bervmult = 0
        if index == 1:
            print('---Subtract berv')
            callback.bervmult = -1.0
        if index == 2:
            print('---Add berv')
            callback.bervmult = 1.0

    def targetchoice(label):
        index = clabels3.index(label)
        if index == 0:
            print('---Apply shifts to both')
            callback.apply = 'both'
        if index == 1:
            print('---Subtract berv')
            callback.apply = '2D'
        if index == 2:
            print('---Add berv')
            callback.apply = '1D'

    def sourcechoice(label):
        index = clabels4.index(label)
        if index == 0:
            print('---Use E2DS')
            callback.source = '2D'
        if index == 1:
            print('---Use S1D')
            callback.source = '1D'

    def update_rv_offset(text):
        if np.char.isnumeric(text):
            callback.vsys = np.float(text)
            print(f'------Changed vsys to {callback.vsys} km/s')
    # def update_rv_offset_whole(val):
    #     rest = callback.offset % 1
    #     callback.offset = val+rest
    # def update_rv_offset_decimal(val):
    #     callback.offset=math.floor(np.abs(callback.offset))*np.sign(callback.offset)+val



    if read_s1d:
        rsource    = RadioButtons(ax_source,clabels4)
        rtarget    = RadioButtons(ax_target,clabels3)
    raddberv   = RadioButtons(ax_addberv,clabels2)
    rstartel   = RadioButtons(ax_startel,clabels)
    tvsys      = TextBox(ax_vsys,'Vsys (km/s)  ',initial=0.0)
    # offset_slider1 = Slider(ax_slider1,'',-80,80,valinit=0.0,valstep=1.0)
    # offset_slider2 = Slider(ax_slider2,'',-1,1,valinit=0.0,valstep=0.01)
    callback.bbacktofit = Button(ax_backtofit, 'Back \n to fit',color='lightcoral',hovercolor='orangered')
    brefit     = Button(ax_refit, 'Re-fit',color='lightcoral') #Done after changing fitting params. In future.
    breplot    = Button(ax_replot, 'Refresh')
    bshift     = Button(ax_shift, 'Align spectra')#This fills in a button object in the above axes.
    breset     = Button(ax_reset, 'Reset to \n initial')
    bclose     = Button(ax_close,'Save & Close')
    badv       = Button(ax_adv,'Adv correction',color='lightcoral',hovercolor='orangered')


    if read_s1d:
        rsource.on_clicked(sourcechoice)
        rtarget.on_clicked(targetchoice)
    raddberv.on_clicked(bervchoice)
    rstartel.on_clicked(framechoice)
    # offset_slider1.on_changed(update_rv_offset_whole)
    # offset_slider2.on_changed(update_rv_offset_decimal)
    tvsys.on_submit(update_rv_offset)
    # brefit.on_clicked(callback.refit)
    breplot.on_clicked(callback.replot_button)
    bshift.on_clicked(callback.align_spectra)#Now this goes back into the class object.
    breset.on_clicked(callback.reset_to_initial)
    bclose.on_clicked(callback.close)
    # badv.on_clicked(callback.go_to_adv)

    plt.show()

    plt.plot(callback.Gw)
    plt.show()

    plt.plot(callback.Lw)
    plt.show()
    pdb.set_trace()
    sys.exit()


    class advcorrector(object,c2d,cT2d):
        def __init__(self):
            if c2d and cT2d:
                self.mode='both'
            elif c2d:
                self.mode='star'
            elif cT2D:
                self.mode='tel'
            else:
                raise Exception('Either centroids 2d or telluric centroids2d is supposed to be set')

            self.list_of_waves = copy.deepcopy(list_of_waves)
            self.list_of_orders = copy.deepcopy(list_of_orders)



        def setup_plot(self):
            self.fig, axs = plt.subplot_mosaic([['top','top','top'],['lower left', 'middle middle',
            'middle right1'],['lower left', 'lower middle','lower right1']],figsize=(14,7))
            self.ax=[axs['top'],axs['lower left'],axs['middle middle'],axs['middle right1'],axs['lower middle'],axs['lower right1']]
            #START HERE





    if M.adv_trigger == True:
        callback2 = advcorrector(copy.deepcopy(M.centroids2d),copy.deepcopy(M.centroidsT2d))
        del callback #This was probably quite big in memory but all we need are the centroids.



    # if read_s1d:
    #     print('---Fitting 1D centroids')
    #     centroids1d=[]
    #     centroidsT1d=[]
    #     widthsT1d=[]
    #     for n,i in enumerate(ccfT):
    #         fit,void = fun.gaussfit(rvT,i/np.nanmean(i),startparams=[-0.9,
    #         rvT[np.argmin(i)],3.0,1.0,0.0,0.0])
    #         centroidsT1d.append(fit[1])
    #         widthsT1d.append(fit[2])
    #
    #     if parallel:#On my 8-core laptop, this yields a factor 2 speedup.
    #         centroids1d, void1, void2, void3 = zip(*Parallel(n_jobs=ncores)(delayed(fit_ccf)(i)
    #         for i in ccf1d))
    #     else:
    #         centroids1d, void1, void2, void3 = zip(*[fit_ccf(i) for i in ccf1d])








    #
    #
    #
    #
    #
    # if instrument == 'ESPRESSO':
    #     explanation=[(f'For ESPRESSO, the S1D and S2D spectra are typically provided in the '
    #     'barycentric frame, in air. Because the S1D spectra are used for telluric correction, '
    #     'this code undoes this barycentric correction so that the S1Ds are saved in the '
    #     'telluric rest-frame while the S2D spectra remain in the barycentric frame.'),'',
    #     ('Therefore, you should see the following values for the the measured line centers:'),
    #     ('- The 1D spectra correlated with PHOENIX should peak at the systemic velocity minus '
    #     f'the BERV correction (equal to {np.round(np.nanmean(berv),2)} km/s on average).'),
    #     '- The 1D spectra correlated with the telluric model should peak at 0 km/s.',
    #     ('- The 2D spectra correlated with PHOENIX should peak the systemic velocity of this '
    #     'star.'),('- The 2D spectra correlated with the tellurics should peak at the '
    #     'barycentric velocity.'),'',('If this is correct, in the Tayph runfile of this dataset,'
    #     ' do_berv_correction should be set to False and air in the config file of this dataset '
    #     'should be set to True.')]
    #
    #
    # if instrument in ['HARPS','HARPSN','GIANO-B']:
    #     explanation=[(f'For {mode}, the s1d spectra are typically provided in the barycentric '
    #     'restframe, while the e2ds spectra are left in the observatory frame, both in air. '
    #     'Because the s1d spectra are used for telluric correction, this code undoes this '
    #     'barycentric correction so that the s1ds are saved in the telluric rest-frame so that '
    #     'they end up in the same frame as the e2ds spectra.'),'',('Therefore, you should see '
    #     'the following values for the the measured line centers:'),('- The 1D spectra '
    #     'correlated with PHOENIX should peak at the systemic velocity minus the BERV correction'
    #     f' (equal to {np.round(np.nanmean(berv),2)} km/s on average).'),('- The 1D spectra '
    #     'correlated with the telluric model should peak at 0 km/s.'),('- The 2D spectra '
    #     'correlated with PHOENIX should peak the systemic velocity minus the BERV correction '
    #     f'(equal to {np.round(np.nanmean(berv),2)} km/s on average).'),('- The 2D spectra '
    #     'correlated with the tellurics should peak at 0 km/s.'),'',('If this is correct, in '
    #     'the Tayph runfile of this dataset, do_berv_correction should be set to True and air '
    #     'in the config file of this dataset should be set to True.')]
    #
    #
    # if instrument in ['SPIROU']:
    #     explanation=[('For SPIROU, no s1d spectra are provided, and the telluric correction is'
    #     ' carried out by the pipeline on the e2ds spectra which are in the observatory '
    #     'rest-frame. The pipeline originally provided these in vaccuum wavelengths, but the '
    #     'read_spirou function should have converted these to air.'),'',('Therefore, you should'
    #     ' see the following values for the the measured line centers:'),('- The 2D spectra '
    #     'correlated with PHOENIX should peak the systemic velocity minus the BERV correction '
    #     f'(equal to {np.round(np.nanmean(berv),2)} km/s on average).'),('- The 2D spectra '
    #     'correlated with the tellurics should peak at 0 km/s.'),'',('If this is correct, in '
    #     'the Tayph runfile of this dataset, do_berv_correction should be set to True and air '
    #     'in the config file of this dataset should be set to True.')]
    #
    # if instrument in ['CRIRES','CARMENES-VIS']:
    #     explanation=[('No explanation provided yet.')]
    #
    # if instrument in ['HIRES-MAKEE']:
    #     explanation=[('do_berv_correction should be set to True and air should be set to'
    #     ' True. Evidently, this information message still needs to be expanded.')]
    #
    #     for s in explanation: ut.tprint(s)
    #
    # print('\n \n \n')
    #
    # print('The derived line positions are as follows:')
    #
    # if read_s1d:
    #     print(f'1D spectra with PHOENIX:  Line center near RV = '
    #     f'{int(np.round(np.nanmedian(centroids1d),1))} km/s.')
    #     print(f'1D spectra with tellurics:  Line center near RV = '
    #     f'{int(np.round(np.nanmedian(centroidsT1d),1))} km/s.')
    # print(f'2D orders with PHOENIX:  Line center near RV = '
    # f'{int(np.round(np.nanmedian(centroids2d),1))} km/s.')
    # print(f'2D orders with tellurics:  Line center near RV = '
    # f'{int(np.round(np.nanmedian(centroidsT2d),1))} km/s.')
    # print('\n \n \n')
    #
    # final_notes = ('Large deviations can occur if the wavelength solution from the pipeline is '
    # 'incorrectly assumed to be in air, or if for whatever reason, the wavelength solution is '
    # 'provided in the reference frame of the star. In the former case, you need to specify in '
    # 'the config file of the data that the wavelength solution written by this file is in '
    # 'vaccuum. This wil be read by Tayph and Molecfit. In the latter case, barycentric '
    # 'correction (and possibly Keplerian corrections) are likely implicily taken into account '
    # 'in the wavelength solution, meaning that these corrections need to be switched off when '
    # "running Tayph's cascade. Molecfit can't be run in the default way in this case, because "
    # 'the systemic velocity is offsetting the telluric spectrum. If no peak is visible at all, '
    # 'the spectral orders are suffering from unknown velocity shifts or continuum fluctuations '
    # 'that this code was not able to take out. In this case, please inspect your data carefully '
    # 'to determine whether you can locate the source of the problem. Continuing to run Tayph '
    # 'from this point on would probably make it very difficult to obtain meaningful '
    # 'cross-correlations.')
    #
    # ut.tprint(final_notes)






















#MAKE SURE THAT WE DO A VACTOAIR IF THIS IS SET IN THE CONFIG FILE.
def molecfit(dataname,mode='both',instrument='HARPS',save_individual='',configfile=None,
plot_spec=False,time_average=False,guide_plot=False):
    """This is the main wrapper for molecfit that pipes a list of s1d spectra and
    executes it. It first launces the molecfit gui on the middle spectrum of the
    sequence, and then loops through the entire list, returning the transmission
    spectra of the Earths atmosphere in the same order as the list provided.
    These can then be used to correct the s1d spectra or the e2ds spectra.
    Note that the s1d spectra are assumed to be in the barycentric frame in vaccuum,
    but that the output transmission spectrum is in the observers frame, and e2ds files
    are in air wavelengths by default.

    You can also set save_individual to a path to an existing folder to which the
    transmission spectra of the time-series can be written one by one.

    Set the mode keyword to either 'GUI', 'batch' or 'both', to run molecfit in respectively
    GUI mode (requiring connection to an X-window), batch mode (which can be run in the background
    without access to a window environment, or both, in which the GUI and the batch are executed
    in the same call to molecfit.)

    Set the time-average keyword to do a fit on the time-average spectrum. This is useful in case
    the individual spectra are too noisy to see the telluric lines well. Although this will help
    you to better set the inclusion and exclusion regions, the resulting fit may not be very good
    because the telluric lines shift due to the BERV, and because the time of the observation
    changes during the run. Therefore, after selecting inclusion regions, you are advised to re-run
    this function in GUI mode with time_average switched off, to make sure that you are getting a
    good fit on individual spectra.
    """
#MAKE SURE THAT WE DO A VACTOAIR IF THIS IS SET IN THE CONFIG FILE.
#MAKE SURE THAT WE DO A VACTOAIR IF THIS IS SET IN THE CONFIG FILE.
#MAKE SURE THAT WE DO A VACTOAIR IF THIS IS SET IN THE CONFIG FILE.

    import pdb
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    import os.path
    import pickle
    import copy
    from pathlib import Path
    import tayph.util as ut
    import tayph.system_parameters as sp
    import tayph.models as models
    import astropy.io.fits as fits
    import pkg_resources
    import tayph.tellurics  as tel
    from tayph.vartests import typetest
    import astropy.units as u
    import astropy.constants as const
    from tqdm import tqdm

    #The DP contains the S1D files and the configile of the data (air or vaccuum)
    if instrument=='HARPS-N': instrument='HARPSN'#Guard against various ways of spelling HARPS-N.

    if dataname[0] in ['/','.']:#Test that this is an absolute path. If so, we trigger a warning.
        ut.tprint(f'WARNING: The name of the dataset {dataname} appears to be set as an absolute '
        'path or a relative path from the current directory. However, Tayph is designed to run '
        'in a working directory in which datasets, models and cross-correlation output is bundled. '
        'To that end, this variable is recommended to be set to a name, or a name with a subfolder '
        'structure, (like "WASP-123" or "WASP-123/night1"), which is to be placed in the data/ '
        'subdirectory. To initialise this file structure, please make an empty working directory '
        'in e.g. /home/user/tayph/xcor_project/, start the python interpreter in this directory '
        'and create a dummy file structure using the make_project_folder function, e.g. by '
        'importing tayph.run and calling '
        'tayph.run.make_project_folder("/home/user/tayph/xcor_project/").\n\n'
        'Molecfit will proceed, assuming that you know what you are doing.')
        dp = ut.check_path(Path(dataname),exists=True)
    else:
        dp = ut.check_path(Path('data')/dataname,exists=True)

    typetest(mode,str,'mode in molecfit()')
    typetest(instrument,str,'mode in molecfit()')

    s1d_path=ut.check_path(dp/'s1ds.pkl',exists=True)
    if not configfile:
        molecfit_config=tel.get_molecfit_config()#Path at which the system-wide molecfit
        #configuration file is supposed to be located, packaged within Tayph.
    else:
        molecfit_config=ut.check_path(configfile)




    if mode.lower() not in ['gui','batch','both']:
        raise ValueError("Molecfit mode should be set to 'GUI', 'batch' or 'both.'")


    #Now we know what mode we are in, do the standard molecfit setups:
    #If this file doesn't exist (e.g. due to accidental corruption) the user needs to supply these
    #parameters.
    if molecfit_config.exists() == False:
        ut.tprint('Molecfit configuration file does not exist. Making a new file at '
        f'{molecfit_config}.')
        tel.set_molecfit_config(molecfit_config)
    else:        #Otherwise we test its contents.
        tel.test_molecfit_config(molecfit_config)
    molecfit_input_folder = Path(sp.paramget('molecfit_input_folder',molecfit_config,
        full_path=True))
    molecfit_prog_folder = Path(sp.paramget('molecfit_prog_folder',molecfit_config,full_path=True))
    python_alias = sp.paramget('python_alias',molecfit_config,full_path=True)
    #If this passes, the molecfit confugration file appears to be set correctly.



    #We check if the parameter file for this instrument exists.
    parname=Path(instrument+'.par')
    parfile = molecfit_input_folder/parname#Path to molecfit parameter file.
    ut.check_path(molecfit_input_folder,exists=True)#This should be redundant, but still check.
    ut.check_path(parfile,exists=True)#Test that the molecfit parameter file
    #for this instrument exists.



    #If that passes, we proceed with loading the S1D files.
    with open(s1d_path,"rb") as p:
        s1dhdr_sorted,s1d_sorted,wave1d_sorted = pickle.load(p)
        N=len(s1dhdr_sorted)
        if N==1:
            wrn_msg=("WARNING in molecfit(): The length of the time series is found to be 1. Are "
                "you certain that there is only one s1d file to apply Molecfit to?")
            ut.tprint(wrn_msg)
            middle_i=0
        else:
            middle_i = int(round(0.5*N))#Initialise molecfit on the middle spectrum of the series.

    middle_date = s1dhdr_sorted[middle_i]['DATE-OBS']
    ut.tprint('Molecfit will first be initialised in GUI mode on the spectrum with the following'
        f' date: {middle_date}')

    print('\n \n')
    ut.tprint('Then, it will be executed automatically onto the entire time series, with dates in'
        'this order:')
    for x in s1dhdr_sorted:
        try:
            print(f"{x['MJD-OBS']} --- {x['DATE-OBS']}")
        except:
            print(f"{x['DATE-OBS']}")
    print('')
    ut.tprint("If these are not chronologically ordered, there is a problem with the way dates are "
        "formatted in the header and you are advised to abort this program.")
    print('\n \n')




    if len(str(save_individual)) > 0:
        ut.check_path(save_individual,exists=True)



    #====== ||  START OF PROGRAM   ||======#

    list_of_wls = []
    list_of_fxc = []
    list_of_trans = []


    if guide_plot:
        tel.guide_plot(dp)

    if mode.lower() == 'gui' or mode.lower()=='both':
        tel.write_file_to_molecfit(molecfit_input_folder,instrument+'.fits',s1dhdr_sorted,
            wave1d_sorted,s1d_sorted,middle_i,plot=plot_spec,time_average=time_average)
        tel.execute_molecfit(molecfit_prog_folder,parfile,molecfit_input_folder,gui=True,
        alias=python_alias)
        wl,fx,trans = tel.retrieve_output_molecfit(molecfit_input_folder/instrument)
        tel.remove_output_molecfit(molecfit_input_folder,instrument)



    if mode.lower() == 'batch' or mode.lower()=='both':
        print('Fitting %s spectra. Here is your progress bar:' % N)
        for i in tqdm(range(N)):#range(len(spectra)):
            #print('Fitting spectrum %s from %s' % (i+1,N))
            #t1=ut.start()
            tel.write_file_to_molecfit(molecfit_input_folder,instrument+'.fits',s1dhdr_sorted,
            wave1d_sorted,s1d_sorted,int(i))
            tel.execute_molecfit(molecfit_prog_folder,parfile,molecfit_input_folder,gui=False)
            wl,fx,trans = tel.retrieve_output_molecfit(molecfit_input_folder/instrument)
            tel.remove_output_molecfit(molecfit_input_folder,instrument)
            #list_of_wls.append(wl*1000.0)#Convert to nm.
            if instrument == 'ESPRESSO':
                berv_corr =  s1dhdr_sorted[i]['HIERARCH ESO QC BERV']
                list_of_wls.append(wl*1000.0*(1.0+(berv_corr*u.km/u.s/const.c)))#Convert to nm. # correct for berv.
            else:
                list_of_wls.append(wl*1000.0)#Convert to nm.
            list_of_fxc.append(fx/trans)
            list_of_trans.append(trans)
            #ut.end(t1)
            if len(str(save_individual)) > 0:
                indv_outpath=Path(save_individual)/f'tel_{i}.fits'
                indv_out = np.zeros((2,len(trans)))
                indv_out[0]=wl*1000.0
                indv_out[1]=trans
                fits.writeto(indv_outpath,indv_out)
        tel.write_telluric_transmission_to_file(list_of_wls,list_of_trans,list_of_fxc,
        dp/'telluric_transmission_spectra.pkl')

def check_molecfit(dp,instrument='HARPS',configfile=None):
    """This allows the user to visually inspect the telluric correction performed by Molecfit, and
    select individual spectra that need to be refit. Each of these spectra will then be fit with
    molecfit in GUI mode."""
    import tayph.util as ut
    import tayph.tellurics as tel
    from pathlib import Path
    import pickle
    import astropy.units as u
    import astropy.constants as const

    dp=ut.check_path(dp,exists=True)
    telpath = ut.check_path(Path(dp)/'telluric_transmission_spectra.pkl',exists=True)
    list_of_wls,list_of_trans,list_of_fxc=tel.read_telluric_transmission_from_file(telpath)
    to_do_manually = tel.check_fit_gui(list_of_wls,list_of_fxc,list_of_trans)

    s1d_path=ut.check_path(dp/'s1ds.pkl',exists=True)
    with open(s1d_path,"rb") as p:
        s1dhdr_sorted,s1d_sorted,wave1d_sorted = pickle.load(p)

    if len(to_do_manually) > 0:
        if not configfile:
            molecfit_config=tel.get_molecfit_config()#Path at which the system-wide molecfit
            #configuration file is supposed to be located, packaged within Tayph.
        else:
            molecfit_config=ut.check_path(configfile,exists=True)

        tel.test_molecfit_config(molecfit_config)
        molecfit_input_folder = Path(sp.paramget('molecfit_input_folder',molecfit_config,
            full_path=True))
        molecfit_prog_folder = Path(sp.paramget('molecfit_prog_folder',molecfit_config,full_path=True))
        python_alias = sp.paramget('python_alias',molecfit_config,full_path=True)
        #If this passes, the molecfit confugration file appears to be set correctly.

        parname=Path(instrument+'.par')
        parfile = molecfit_input_folder/parname#Path to molecfit parameter file.

        print('The following spectra were selected to be redone manually:')
        print(to_do_manually)
        for i in to_do_manually:
            tel.write_file_to_molecfit(molecfit_input_folder,instrument+'.fits',s1dhdr_sorted,wave1d_sorted,s1d_sorted,int(i))
            tel.execute_molecfit(molecfit_prog_folder,parfile,molecfit_input_folder,gui=True,alias=python_alias)
            wl,fx,trans = tel.retrieve_output_molecfit(molecfit_input_folder/instrument)
            if instrument == 'ESPRESSO':
                berv_corr =  s1dhdr_sorted[i]['HIERARCH ESO QC BERV']
                list_of_wls[int(i)] = wl*1000.0 * ((1.0+(berv_corr*u.km/u.s/const.c))) #Convert to nm. # correct for berv.
            else:
                list_of_wls[int(i)] = wl*1000.0#Convert to nm.
            list_of_fxc[int(i)] = fx/trans
            list_of_trans[int(i)] = trans
        tel.write_telluric_transmission_to_file(list_of_wls,list_of_trans,list_of_fxc,dp/'telluric_transmission_spectra.pkl')


    # return(list_of_wls,list_of_trans)
