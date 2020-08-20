#This package contains high-level wrappers for running the entire sequence.

__all__ = [
    'start_run',
    'run_instance',
]

def start_run(configfile):
    """
    This is the main command-line initializer of the cross-correlation routine provided by Tayph.
    It parses a configuration file located at configfile which should contain predefined keywords.
    These keywords are passed on to the run_instance routine, which executes the analysis cascade.
    A call to this function is called a "run" of Tayph. A run has this configuration file as input,
    as well as a dataset, a (list of) cross-correlation template(s) and a (list of) models for injection
    purposes.
    """
    import tayph.system_parameters as sp
    cf = configfile



    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print(' = = = = WELCOME TO TAYPH = = = =')
    print('')
    print(f'    Running {cf}')
    print('')
    print('')
    print('')

    print('---Start')
    print('---Load parameters from config file')
    modellist = sp.paramget('model',cf,full_path=True).split(',')
    templatelist = sp.paramget('model',cf,full_path=True).split(',')

    params={'dp':sp.paramget('datapath',cf,full_path=True),
            'shadowname':sp.paramget('shadowname',cf,full_path=True),
            'maskname':sp.paramget('maskname',cf,full_path=True),
            'RVrange':sp.paramget('RVrange',cf,full_path=True),
            'drv':sp.paramget('drv',cf,full_path=True),
            'f_w':sp.paramget('f_w',cf,full_path=True),
            'do_colour_correction':sp.paramget('do_colour_correction',cf,full_path=True),
            'do_telluric_correction':sp.paramget('do_telluric_correction',cf,full_path=True),
            'do_xcor':sp.paramget('do_xcor',cf,full_path=True),
            'plot_xcor':sp.paramget('plot_xcor',cf,full_path=True),
            'make_mask':sp.paramget('make_mask',cf,full_path=True),
            'apply_mask':sp.paramget('apply_mask',cf,full_path=True),
            'do_berv_correction':sp.paramget('do_berv_correction',cf,full_path=True),
            'do_keplerian_correction':sp.paramget('do_keplerian_correction',cf,full_path=True),
            'make_doppler_model':sp.paramget('make_doppler_model',cf,full_path=True),
            'skip_doppler_model':sp.paramget('skip_doppler_model',cf,full_path=True),
            'modellist':modellist,
            'templatelist':templatelist,
            'template_library':sp.paramget('template_library',cf,full_path=True),
            'model_library':sp.paramget('model_library',cf,full_path=True),
    }
    run_instance(params)


def run_instance(p):
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
    import distutils.util
    import pickle
    import copy
    from pathlib import Path

    import tayph.util as ut
    import tayph.operations as ops
    import tayph.functions as fun
    import tayph.system_parameters as sp
    import tayph.tellurics as telcor
    import tayph.masking as masking
    from tayph.vartests import typetest,notnegativetest,nantest,postest,typetest_array,dimtest
    # from lib import models
    # from lib import analysis
    # from lib import cleaning
    # from lib import read_data as rd
    # from lib import masking as masking
    # from lib import shadow as shadow
    # from lib import molecfit as telcor


#First parse the parameter dictionary into required variables and test them.
    typetest(p,dict,'params in run_instance()')

    dp = Path(p['dp'])
    ut.check_path(dp,exists=True)


    modellist = p['modellist']
    templatelist = p['templatelist']
    model_library = p['model_library']
    template_library = p['template_library']
    typetest(modellist,[str,list],'modellist in run_instance()')
    typetest(templatelist,[str,list],'templatelist in run_instance()')
    typetest(model_library,str,'model_library in run_instance()')
    typetest(template_library,str,'template_library in run_instance()')
    ut.check_path(model_library,exists=True)
    ut.check_path(template_library,exists=True)
    if type(modellist) == str:
        modellist = [modellist]#Force this to be a list
    if type(templatelist) == str:
        templatelist = [templatelist]#Force this to be a list
    typetest_array(modellist,str,'modellist in run_instance()')
    typetest_array(templatelist,str,'modellist in run_instance()')


    shadowname = p['shadowname']
    maskname = p['maskname']
    typetest(shadowname,str,'shadowname in run_instance()')
    typetest(maskname,str,'shadowname in run_instance()')


    RVrange = p['RVrange']
    drv = p['drv']
    f_w = p['f_w']
    typetest(RVrange,[int,float],'RVrange in run_instance()')
    typetest(drv,[int,float],'drv in run_instance()')
    typetest(f_w,[int,float],'f_w in run_instance()')
    nantest(RVrange,'RVrange in run_instance()')
    nantest(drv,'drv in run_instance()')
    nantest(f_w,'f_w in run_instance()')
    postest(RVrange,'RVrange in run_instance()')
    postest(drv,'drv in run_instance()')
    notnegativetest(f_w,'f_w in run_instance()')


    do_colour_correction=p['do_colour_correction']
    do_telluric_correction=p['do_telluric_correction']
    do_xcor=p['do_xcor']
    plot_xcor=p['plot_xcor']
    make_mask=p['make_mask']
    apply_mask=p['apply_mask']
    do_berv_correction=p['do_berv_correction']
    do_keplerian_correction=p['do_keplerian_correction']
    make_doppler_model=p['make_doppler_model']
    skip_doppler_model=p['skip_doppler_model']
    typetest(do_colour_correction,bool, 'do_colour_correction in run_instance()')
    typetest(do_telluric_correction,bool,'do_telluric_correction in run_instance()')
    typetest(do_xcor,bool,              'do_xcor in run_instance()')
    typetest(plot_xcor,bool,            'plot_xcor in run_instance()')
    typetest(make_mask,bool,            'make_mask in run_instance()')
    typetest(apply_mask,bool,           'apply_mask in run_instance()')
    typetest(do_berv_correction,bool,   'do_berv_correction in run_instance()')
    typetest(do_keplerian_correction,bool,'do_keplerian_correction in run_instance()')
    typetest(make_doppler_model,bool,   'make_doppler_model in run_instance()')
    typetest(skip_doppler_model,bool,   'skip_doppler_model in run_instance()')



#We start by defining constants and preparing for generating output.
    c=const.c.value/1000.0#in km/s
    colourdeg = 3#A fitting degree for the colour correction.


    print(f'---Passed parameter input tests. Creating output folder tree in {Path("output")/dp}.')
    libraryname=str(template_library).split('/')[-1]
    if str(dp).split('/')[0] == 'data':
        dataname=str(dp).replace('data/','')
        print(f'------Data is located in data/ folder. Assuming output name for this dataset as {dataname}')
    else:
        dataname=dp
        print(f'------Data is NOT located in data/ folder. Assuming output name for this dataset as {dataname}')

    for templatename in templatelist:
        outpath=Path('output')/Path(dataname)/Path(libraryname)/Path(templatename)
        print(outpath)#NEED TO CONTINUE HERE TO LOOP OVER TEMPLATES PROBABLY NEED TO MOVE THIS LOOP DOWNWARDS

        if not os.path.exists(outpath):
            print(f"------The output location ({outpath}) didn't exist, I made it now.")
            os.makedirs(outpath)



    list_of_wls=[]#This will store all the data.
    list_of_orders=[]#All of it needs to be loaded into your memory.
    list_of_sigmas=[]

    trigger2 = 0#These triggers are used to limit the generation of output in the forloop.
    trigger3 = 0
    n_negative_total = 0#This will hold the total number of pixels that were set to NaN because they were zero when reading in the data.
    air = sp.paramget('air',dp)#Read bool from str in config file.
    typetest(air,bool,'air in run_instance()')

    filelist_orders= [str(i) for i in Path(dp).glob('order_*.fits')]
    if len(filelist_orders) == 0:
        raise Exception(f'Runtime error: No orders_*.fits files were found in {dp}.')
    try:
        order_numbers = [int(i.split('order_')[1].split('.')[0]) for i in filelist_orders]
    except:
        raise Exception('Runtime error: Failed casting fits filename numerals to ints. Are the filenames of the spectral orders correctly formatted?')
    order_numbers.sort()#This is the ordered list of numerical order IDs.
    n_orders = len(order_numbers)
    if n_orders == 0:
        raise Exception(f'Runtime error: n_orders may not have ended up as zero. ({n_orders})')








#Loading the data from the datafolder.
    if do_xcor == True or plot_xcor == True or make_mask == True:
        print(f'---Loading orders from {dp}.')

        # for i in range(startorder,endorder+1):
        for i in order_numbers:
            wavepath = dp/f'wave_{i}.fits'
            orderpath= dp/f'order_{i}.fits'
            sigmapath= dp/f'sigma_{i}.fits'
            ut.check_path(wavepath,exists=True)
            ut.check_path(orderpath,exists=True)
            ut.check_path(sigmapath,exists=False)
            wave_axis = fits.getdata(wavepath)
            dimtest(wave_axis,[0],'wavelength grid in run_instance()')
            n_px = len(wave_axis)#Pixel width of the spectral order.
            if air == False:
                if i == np.min(order_numbers):
                    print("------Assuming wavelengths are in vaccuum.")
                list_of_wls.append(1.0*wave_axis)
            else:
                if i == np.min(order_numbers):
                    print("------Applying airtovac correction.")
                list_of_wls.append(ops.airtovac(wave_axis))

            order_i = fits.getdata(orderpath)
            if i == np.min(order_numbers):
                dimtest(order_i,[0,n_px],f'order {i} in run_instance()')#For the first order, check that it is 2D and that is has a width equal to n_px.
                n_exp = np.shape(order_i)[0]#then fix n_exp. All other orders should have the same n_exp.
                print(f'------{n_exp} exposures recognised.')
            else:
                dimtest(order_i,[n_exp,n_px],f'order {i} in run_instance()')

            #Now test for negatives, set them to NaN and track them.
            n_negative = len(order_i[order_i <= 0])
            if trigger3 == 0 and n_negative > 0:
                print("------Setting negative values to NaN.")
                trigger3 = -1
            n_negative_total+=n_negative
            order_i[order_i <= 0] = np.nan
            postest(order_i,f'order {i} in run_instance().')#make sure whatever comes out here is strictly positive.
            list_of_orders.append(order_i)

            try:#Try to get a sigma file. If it doesn't exist, we raise a warning. If it does, we test its dimensions and append it.
                sigma_i = fits.getdata(sigmapath)
                dimtest(sigma_i,[n_exp,n_px],f'order {i} in run_instance().')
                list_of_sigmas.append(sigma_i)
            except FileNotFoundError:
                if trigger2 == 0:
                    print('------WARNING: Sigma (flux error) files not provided. Assuming sigma = sqrt(flux). This is standard practise for HARPS data, but e.g. ESPRESSO has a pipeline that computes standard errors on each pixel for you.')
                    trigger2=-1
                list_of_sigmas.append(np.sqrt(order_i))
        print(f"------{n_negative_total} negative values set to NaN ({np.round(100.0*n_negative_total/n_exp/n_px/len(order_numbers),2)}% of total spectral pixels in dataset.)")

    if len(list_of_orders) != n_orders:
        raise Exception('Runtime error: n_orders is not equal to the length of list_of_orders. Something went wrong when reading them in?')

    print('---Finished loading dataset to memory.')



    #Apply telluric correction file or not.
    # plt.plot(list_of_wls[60],list_of_orders[60][10],color='red')
    # plt.plot(list_of_wls[60],list_of_orders[60][10]+list_of_sigmas[60][10],color='red',alpha=0.5)#plot corrected spectra
    # plt.plot(list_of_wls[60],list_of_orders[60][10]/list_of_sigmas[60][10],color='red',alpha=0.5)#plot SNR
    if do_telluric_correction == True and n_orders > 0:
        print('---Applying telluric correction')
        telpath = dp/'telluric_transmission_spectra.pkl'
        list_of_orders,list_of_sigmas = telcor.apply_telluric_correction(telpath,list_of_wls,list_of_orders,list_of_sigmas)

    # plt.plot(list_of_wls[60],list_of_orders[60][10],color='blue')
    # plt.plot(list_of_wls[60],list_of_orders[60][10]+list_of_sigmas[60][10],color='blue',alpha=0.5)#plot corrected spectra

    # plt.plot(list_of_wls[60],list_of_orders[60][10]/list_of_sigmas[60][10],color='blue',alpha=0.5) #plot SNR
    # plt.show()
    # pdb.set_trace()

#Do velocity correction of wl-solution. Explicitly after telluric correction
#but before masking! Because the cross-correlation relies on columns being masked.
#Then if you start to move the CCFs around before removing the time-average,
#each masked column becomes slanted. Bad deal.
    rv_cor = 0
    if do_berv_correction == True:
        rv_cor += sp.berv(dp)
    if do_keplerian_correction == True:
        rv_cor-=sp.RV_star(dp)*(1.0)

    if type(rv_cor) != int and len(list_of_orders) > 0:
        print('---Reinterpolating data to correct velocities')
        list_of_orders_cor = []
        list_of_sigmas_cor = []
        for i in range(len(list_of_wls)):
            order = list_of_orders[i]
            sigma = list_of_sigmas[i]
            order_cor = order*0.0
            sigma_cor = sigma*0.0
            for j in range(len(list_of_orders[0])):
                wl_i = interp.interp1d(list_of_wls[i],order[j],bounds_error=False)
                si_i = interp.interp1d(list_of_wls[i],sigma[j],bounds_error=False)
                wl_cor = list_of_wls[i]*(1.0-(rv_cor[j]*u.km/u.s/const.c))#The minus sign was tested on a slow-rotator.
                order_cor[j] = wl_i(wl_cor)
                sigma_cor[j] = si_i(wl_cor)#I checked that this works because it doesn't affect the SNR, apart from wavelength-shifting it.
            list_of_orders_cor.append(order_cor)
            list_of_sigmas_cor.append(sigma_cor)
            ut.statusbar(i,fun.findgen(len(list_of_wls)))
        # plt.plot(list_of_wls[60],list_of_orders[60][10]/list_of_sigmas[60][10],color='blue')
        # plt.plot(list_of_wls[60],list_of_orders_cor[60][10]/list_of_sigmas_cor[60][10],color='red')
        # plt.show()
        # sys.exit()
        list_of_orders = list_of_orders_cor
        list_of_sigmas = list_of_sigmas_cor


    if len(list_of_orders) != n_orders:
        raise Exception('Runtime error: n_orders is no longer equal to the length of list_of_orders, though it was before. Something went wrong during telluric correction or velocity correction.')

#Continue populating masking (most is there), add normalise_orders to ops; and write documentation and tests for plotting_scales_2D and the masking routines.


#Do masking or not.
    if make_mask == True and len(list_of_orders) > 0:
        if do_colour_correction == True:
            print('---Constructing mask with intra-order colour correction applied')
            masking.mask_orders(list_of_wls,ops.normalize_orders(list_of_orders,list_of_sigmas,colourdeg)[0],dp,maskname,40.0,5.0,manual=True)
        else:
            print('---Constructing mask WITHOUT intra-order colour correction applied.')
            print('---Switch on colour correction if you see colour variations in the 2D spectra.')
            masking.mask_orders(list_of_wls,list_of_orders,dp,maskname,40.0,5.0,manual=True)
        if apply_mask == False:
            print('---Warning in run_instance: Mask was made but is not applied to data (apply_mask = False)')

    if apply_mask == True and len(list_of_orders) > 0:
        print('---Applying mask')
        list_of_orders = masking.apply_mask_from_file(dp,maskname,list_of_orders)

    if do_xcor == True:
        print('---Healing NaNs')
        list_of_orders = masking.interpolate_over_NaNs(list_of_orders)#THERE IS AN ISSUE HERE: INTERPOLATION SHOULD ALSO HAPPEN ON THE SIGMAS ARRAY!


        #Normalize the orders to their average flux in order to effectively apply
        #a broad-band colour correction (colour is a function of airmass and seeing).
    if do_colour_correction == True:
        print('---Normalizing orders to common flux level')
        # plt.plot(list_of_wls[60],list_of_orders[60][10]/list_of_sigmas[60][10],color='blue',alpha=0.4)
        list_of_orders_normalised,list_of_sigmas_normalised,meanfluxes = ops.normalize_orders(list_of_orders,list_of_sigmas,colourdeg)#I tested that this works because it doesn't alter the SNR.

        meanfluxes_norm = meanfluxes/np.nanmean(meanfluxes)
    else:
        meanfluxes_norm = fun.findgen(len(list_of_orders[0]))*0.0+1.0#All unity.
        # plt.plot(list_of_wls[60],list_of_orders_normalised[60][10]/list_of_sigmas[60][10],color='red',alpha=0.4)
        # plt.show()
        # sys.exit()

    sys.exit()
#Construct the cross-correlation template in case we will be doing or plotting xcor.
    if do_xcor == True or plot_xcor == True:
        print('---Building template')
        wlt,T=models.build_template(templatename,binsize=0.5,maxfrac=0.01,resolution=120000.0,template_library=template_library)
        T*=(-1.0)
#CAN'T I OUTSOURCE THIS TO DANIEL AND SIMON? I mean for each spectrum they could also
#generate a binary mask by putting delta's instead of.. well... not precisely
#because of masking by line wings; that's the whole trick of opacity calculation.
#And then Daniel can subtract the continuum himself...
#I can rewrite this to include the name pointing to another model to be
#used as continuum normalization.


#Perform the cross-correlation on the entire list of orders.
    if do_xcor == True:
        print('---Cross-correlating spectra')
        rv,ccf,ccf_e,Tsums=analysis.xcor(list_of_wls,list_of_orders_normalised,np.flipud(np.flipud(wlt)),T,drv,RVrange,list_of_errors=list_of_sigmas_normalised)
        print('------Writing CCFs to '+outpath)
        ut.writefits(outpath+'ccf.fits',ccf)
        ut.writefits(outpath+'ccf_e.fits',ccf_e)
        ut.writefits(outpath+'RV.fits',rv)
        ut.writefits(outpath+'Tsum.fits',Tsums)
    else:
        print('---Reading CCFs from '+outpath)
        if os.path.isfile(outpath+'ccf.fits') == False:
            print('------ERROR: CCF output not located at '+outpath+'. Set do_xcor to True to create these files?')
            sys.exit()
        rv=fits.getdata(outpath+'rv.fits')
        ccf = fits.getdata(outpath+'ccf.fits')
        ccf_e = fits.getdata(outpath+'ccf_e.fits')
        Tsums = fits.getdata(outpath+'Tsum.fits')

    if plot_xcor == True and inject_model == False:
        print('---Plotting orders and XCOR')
        fitdv = sp.paramget('fitdv',dp)
        analysis.plot_XCOR(list_of_wls,list_of_orders_normalised,wlt,T,rv,ccf,Tsums,dp,CCF_E=ccf_e,dv=fitdv)


    ccf_cor = ccf*1.0
    ccf_e_cor = ccf_e*1.0

    print('---Cleaning CCFs')
    ccf_n,ccf_ne,ccf_nn,ccf_nne= cleaning.clean_ccf(rv,ccf_cor,ccf_e_cor,dp)
    ut.writefits(outpath+'ccf_normalized.fits',ccf_nn)
    ut.writefits(outpath+'ccf_ne.fits',ccf_ne)
    if make_doppler_model == True and skip_doppler_model == False:
        # pdb.set_trace()
        shadow.construct_doppler_model(rv,ccf_nn,dp,shadowname,xrange=[-200,200],Nxticks=20.0,Nyticks=10.0)
    if skip_doppler_model == False:
        print('---Reading doppler shadow model from '+shadowname)
        doppler_model,maskHW = shadow.read_shadow(dp,shadowname,rv,ccf)
        ccf_clean,matched_ds_model = shadow.match_shadow(rv,ccf_nn,dp,doppler_model,maskHW)#THIS IS AN ADDITIVE CORRECTION, SO CCF_NNE DOES NOT NEED TO BE ALTERED AND IS STILL VALID VOOR CCF_CLEAN
    else:
        print('---Not performing shadow correction')
        ccf_clean = ccf_nn*1.0
        matched_ds_model = ccf_clean*0.0

    if f_w > 0.0:
        ccf_clean_filtered,wiggles = cleaning.filter_ccf(rv,ccf_clean,v_width = f_w)#THIS IS ALSO AN ADDITIVE CORRECTION, SO CCF_NNE IS STILL VALID.
    else:
        ccf_clean_filtered = ccf_clean*1.0
        wiggles = ccf_clean*0.0

    ut.save_stack(outpath+'cleaning_steps.fits',[ccf,ccf_cor,ccf_nn,ccf_clean,matched_ds_model,ccf_clean_filtered,wiggles])
    ut.writefits(outpath+'ccf_cleaned.fits',ccf_clean_filtered)
    ut.writefits(outpath+'ccf_cleaned_error.fits',ccf_nne)

    print('---Weighing CCF rows by mean fluxes that were normalised out')
    ccf_clean_filtered = np.transpose(np.transpose(ccf_clean_filtered)*meanfluxes_norm)#MULTIPLYING THE AVERAGE FLUXES BACK IN! NEED TO CHECK THAT THIS ALSO GOES PROPERLY WITH THE ERRORS!
    ccf_nne = np.transpose(np.transpose(ccf_nne)*meanfluxes_norm)

    print('---Constructing KpVsys')
    Kp,KpVsys,KpVsys_e = analysis.construct_KpVsys(rv,ccf_clean_filtered,ccf_nne,dp)
    ut.writefits(outpath+'KpVsys.fits',KpVsys)
    ut.writefits(outpath+'KpVsys_e.fits',KpVsys_e)
    ut.writefits(outpath+'Kp.fits',Kp)
    if plot_xcor == True and inject_model == False:
        print('---Plotting KpVsys')
        analysis.plot_KpVsys(rv,Kp,KpVsys,dp)







    #Now repeat it all for the model injection.
    if inject_model == True:
        for modelname in modellist:
            outpath_i = outpath+modelname+'/'
            if do_xcor == True:
                print('---Injecting model '+modelname)
                list_of_orders_injected=models.inject_model(list_of_wls,list_of_orders,dp,modelname,model_library=model_library)#Start with the unnormalised orders from before.
                #Normalize the orders to their average flux in order to effectively apply
                #a broad-band colour correction (colour is a function of airmass and seeing).
                if do_colour_correction == True:
                    print('------Normalizing injected orders to common flux level')
                    list_of_orders_injected,list_of_sigmas_injected,meanfluxes_injected = ops.normalize_orders(list_of_orders_injected,list_of_sigmas,colourdeg)
                    meanfluxes_norm_injected = meanfluxes_injected/np.mean(meanfluxes_injected)
                else:
                    meanfluxes_norm_injected = fun.findgen(len(list_of_orders_injected[0]))*0.0+1.0#All unity.
                # plt.plot(list_of_wls[60],list_of_orders[60][10]/list_of_sigmas[60][10],color='red',alpha=0.4)
                # plt.show()
                # sys.exit()

                print('------Cross-correlating injected orders')
                rv_i,ccf_i,ccf_e_i,Tsums_i=analysis.xcor(list_of_wls,list_of_orders_injected,np.flipud(np.flipud(wlt)),T,drv,RVrange,list_of_errors=list_of_sigmas_injected)
                print('------Writing injected CCFs to '+outpath_i)
                if not os.path.exists(outpath_i):
                    print("---------That path didn't exist, I made it now.")
                    os.makedirs(outpath_i)
                ut.writefits(outpath_i+'/'+'ccf_i_'+modelname+'.fits',ccf_i)
                ut.writefits(outpath_i+'/'+'ccf_e_i_'+modelname+'.fits',ccf_e_i)
            else:
                print('---Reading injected CCFs from '+outpath_i)
                # try:

                    # f = open(outpath_i+'ccf_i_'+modelname+'.fits', 'r')
                    # print(outpath_i+'ccf_i_'+modelname+'.fits')
                    # f2 = open(outpath+'ccf_e_i_'+modelname+'.fits','r')
                # except FileNotFoundError:
                if os.path.isfile(outpath_i+'ccf_i_'+modelname+'.fits') == False:
                    print('------ERROR: Injected CCF not located at '+outpath_i+'ccf_i_'+modelname+'.fits'+'. Set do_xcor and inject_model to True?')
                    sys.exit()
                if os.path.isfile(outpath_i+'ccf_e_i_'+modelname+'.fits') == False:
                    print('------ERROR: Injected CCF error not located at '+outpath_i+'ccf_e_i_'+modelname+'.fits'+'. Set do_xcor and inject_model to True?')
                    sys.exit()
                # f.close()
                # f2.close()
                ccf_i = fits.getdata(outpath_i+'ccf_i_'+modelname+'.fits')
                ccf_e_i = fits.getdata(outpath_i+'ccf_e_i_'+modelname+'.fits')



            print('---Cleaning injected CCFs')
            ccf_n_i,ccf_ne_i,ccf_nn_i,ccf_nne_i = cleaning.clean_ccf(rv,ccf_i,ccf_e_i,dp)
            ut.writefits(outpath_i+'ccf_normalized_i.fits',ccf_nn_i)
            ut.writefits(outpath_i+'ccf_ne_i.fits',ccf_ne_i)

            # if make_doppler_model == True and skip_doppler_model == False:
                # shadow.construct_doppler_model(rv,ccf_nn,dp,shadowname,xrange=[-200,200],Nxticks=20.0,Nyticks=10.0)
            if skip_doppler_model == False:
                # print('---Reading doppler shadow model from '+shadowname)
                # doppler_model,maskHW = shadow.read_shadow(dp,shadowname,rv,ccf)
                ccf_clean_i,matched_ds_model_i = shadow.match_shadow(rv,ccf_nn_i,dp,doppler_model,maskHW)
            else:
                print('---Not performing shadow correction on injected spectra either.')
                ccf_clean_i = ccf_nn_i*1.0
                matched_ds_model_i = ccf_clean_i*0.0

            if f_w > 0.0:
                ccf_clean_i_filtered,wiggles_i = cleaning.filter_ccf(rv,ccf_clean_i,v_width = f_w)
            else:
                ccf_clean_i_filtered = ccf_clean_i*1.0



            ut.writefits(outpath_i+'ccf_cleaned_i.fits',ccf_clean_i_filtered)
            ut.writefits(outpath+'ccf_cleaned_i_error.fits',ccf_nne)

            print('---Weighing injected CCF rows by mean fluxes that were normalised out')
            ccf_clean_i_filtered = np.transpose(np.transpose(ccf_clean_i_filtered)*meanfluxes_norm_injected)#MULTIPLYING THE AVERAGE FLUXES BACK IN! NEED TO CHECK THAT THIS ALSO GOES PROPERLY WITH THE ERRORS!
            ccf_nne_i = np.transpose(np.transpose(ccf_nne_i)*meanfluxes_norm_injected)

            print('---Constructing injected KpVsys')
            Kp,KpVsys_i,KpVsys_e_i = analysis.construct_KpVsys(rv,ccf_clean_i_filtered,ccf_nne_i,dp)
            ut.writefits(outpath_i+'KpVsys_i.fits',KpVsys_i)
            # ut.writefits(outpath+'KpVsys_e_i.fits',KpVsys_e_i)
            if plot_xcor == True:
                print('---Plotting KpVsys with '+modelname+' injected.')
                analysis.plot_KpVsys(rv,Kp,KpVsys,dp,injected=KpVsys_i)
    # print('OK')
    # sys.exit()



    # if plot_xcor == True:
    #     print('---Plotting 2D CCF')
    #     print("---THIS NEEDS TO BE REVAMPED!")
        # analysis.plot_ccf(rv,ccf_nn,dp,xrange=[-200,200],Nticks=20.0,doppler_model = doppler_rv)
        # analysis.plot_ccf(rv,ccf_ds_model,dp,xrange=[-200,200],Nticks=20.0,doppler_model = doppler_rv)
        # analysis.plot_ccf(rv,ccf_clean,dp,xrange=[-200,200],Nticks=20.0,doppler_model = doppler_rv)

    # ut.save_stack('test.fits',[ccf_n,ccf_nn,ccf_ne,ccf_nne])
    # pdb.set_trace()




#The following tests XCOR on synthetic orders.
#wlm = fun.findgen(2e6)/10000.0+650.0
#fxm = wlm*0.0
#fxm[[(fun.findgen(400)*4e3+1e3).astype(int)]] = 1.0
#px_scale=wlm[1]-wlm[0]
#dldv = np.min(wlm) / c /px_scale
#T=ops.smooth(fxm,dldv * 5.0)
#fxm_b=ops.smooth(fxm,dldv * 20.0,mode='gaussian')
#plt.plot(wlm,T)
#plt.plot(wlm,fxm_b)
#plt.show()
#ii = interpolate.interp1d(wlm,fxm_b)
#dspec = ii(wl)
#order = fun.rebinreform(dspec/np.max(dspec),30)
#fits.writeto('test.fits',order,overwrite=True)
#ccf=analysis.xcor([wl,wl,wl+0.55555555555],[order,order*3.0,order*5.0],wlm,T,drv,RVrange)

# meanspec=np.mean(order1,axis=0)
# meanspec-=min(meanspec)
# meanspec/=np.max(meanspec)
# T-=np.min(T)
# T/=np.median(T)
# T-=np.max(T[(wlt >= min(wl1)) & (wlt <= max(wl1))])
# plt.plot(wl1,meanspec)
# plt.plot(wlt*(1.0-200.0/300000.0),T)
# plt.xlim((min(wl1),max(wl1)))
# plt.show()


# plt.plot(wlt,T)
# plt.show()


# analysis.plot_RV_star(dp,rv,ccf,RVrange=[-50,100]) #THis entire function is probably obsolete.

# sel = (doppler_rv > -100000.0)
# ccf_ds = ops.shift_ccf(rv,ccf_nn[sel,:],(-1.0)*doppler_rv[sel])
# vsys = sp.paramget('vsys',dp)
# sel = ((rv >= -20) & (rv <= 60))
#
# plt.plot(rv[sel],np.nanmean(ccf_ds[:,sel],axis=0))
# # plt.axvline(x=vsys)
# plt.show()
# pdb.set_trace()

#The following reads in a binary mask and plots it to search for the velocity shift
#by comparing with a Kelt-9 model.
#start = time.time()
#wlb,fxb=models.read_binary_mask('models/A0v2.mas')
#end = time.time()
#print(end-start)
#FeIImodel=fits.getdata('models/FeII_4500_c.fits')
#wlm=FeIImodel[0,:]
#fxm=FeIImodel[1,:]
#pylab.plot(wlm,300.0*(fxm-np.median(fxm)))
#pylab.plot(wlb,fxb)
#pylab.show()



#The following tests blurring.
#spec = fun.findgen(len(wl))*0.0
#spec[500] = 1
#spec[1000] = 1
#spec[3000] = 1
#spec[3012] = 1
#spec_b=ops.blur_rotate(wl,spec,3.0,1.5,1.5,90.0)
#spec_b2=ops.blur_rotate(wl,spec,3.0,2.0,1.0,90.0)
#t1=ut.start()
#spec_b=ops.blur_spec(wl,spec,20.0,mode='box')
#spec_b2=ops.blur_spec(wl,spec,20.0,mode='box')
#dt1=ut.end(t1)
#pylab.plot(wl,spec)
#pylab.plot(wl,spec_b)
#pylab.plot(wl,spec_b2)
#pylab.show()




#This times the gaussian function.
#x=fun.findgen(10000)
#start = time.time()
#for i in range(0,10000):
#    g=fun.gaussian(x,20.0,50000.0,10000.0)
#end = time.time()
#print(end - start)
#plot(x,g)
#show()
