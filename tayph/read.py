


def read_HARPS_e2ds(inpath,outname,nowave=False,molecfit=False,mode='HARPS',ignore_exp=[]):
    """This reads pipeline-reduced HARPS or HARPS-N data as downloaded from the ESO
    or TNG archives, and formats them into spectral orders. It can also call molecfit
    to perform telluric corrections; but for this, molecfit needs to be installed in a very
    specific manner.

    Set the nowave keyword to True if the dataset has no wave files associated with it.
    This may happen if you downloaded ESO Advanced Data Products, which include
    reduced science e2ds's but not reduced wave e2ds's. The wavelength solution
    is still encoded in the fits header however, so we take it from there, instead.

    Set the ignore_exp keyword to a list of exposures (start counting at 0) that
    need to be ignored when reading, e.g. because they are bad for some reason.
    If you have set molecfit to True, this becomes an expensive parameter to
    play with in terms of computing time.

    IF IN THE FUTURE A BERV KEYWORD WOULD BE MISSING, I HAVE INCLUDED AN ASTROPY
    IMPLEMENTATION THAT ACCURATELY CALCULATES THE BERV FROM THE MJD. SEE SYSTEM_PARAMETERS.PY


    """

    import os
    import pdb
    from astropy.io import fits
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    import tayph.util as ut
    from tayph.vartests import typetest
    import tayph.tellurics as mol
    import pyfits
    import copy
    import scipy.interpolate as interp
    import pickle
    from pathlib import Path
    import warnings

    molecfit = False
    #First check the input:
    inpath=ut.check_path(inpath,exists=True)
    typetest(outname,str,'outname in read_HARPS_e2ds()')
    typetest(nowave,bool,'nowave switch in read_HARPS_e2ds()')
    typetest(molecfit,bool,'molecfit switch in read_HARPS_e2ds()')
    typetest(ignore_exp,list,'ignore_exp in read_HARPS_e2ds()')
    typetest(mode,str,'mode in read_HARPS_e2ds()')

    filelist=os.listdir(inpath)
    N=len(filelist)

    if len(filelist) == 0:
        raise FileNotFoundError(f" in read_HARPS_e2ds: input folder {str(inpath)} is empty.")


    if mode not in ['HARPS','HARPSN','HARPS-N']:
        raise ValueError("in read_HARPS_e2ds: mode needs to be set to HARPS or HARPSN.")
    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    type=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    norders=np.array([])
    e2ds=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    blaze=[]
    wavefile_used = []
    outpath = Path('data/'+outname)
    if os.path.exists(outpath) != True:
        os.makedirs(outpath)


    e2ds_count = 0
    sci_count = 0
    wave_count = 0
    blaze_count = 0
    s1d_count = 0


    #MODE SWITCHING HERE:
    if mode == 'HARPS':
        catkeyword = 'HIERARCH ESO DPR CATG'
        bervkeyword = 'HIERARCH ESO DRS BERV'
        thfilekeyword = 'HIERARCH ESO DRS CAL TH FILE'
        Zstartkeyword = 'HIERARCH ESO TEL AIRM START'
        Zendkeyword = 'HIERARCH ESO TEL AIRM END'
    if mode == 'HARPSN' or mode == 'HARPS-N':
        catkeyword = 'OBS-TYPE'
        bervkeyword = 'HIERARCH TNG DRS BERV'
        thfilekeyword = 'HIERARCH TNG DRS CAL TH FILE'
        Zstartkeyword = 'AIRMASS'
        Zendkeyword = 'AIRMASS'#These are the same because HARPSN doesnt have start and end keywords.
        #Down there, the airmass is averaged, so there is no problem in taking the average of the same number.



    for i in range(N):
        if filelist[i].endswith('e2ds_A.fits'):
            e2ds_count += 1
            print(filelist[i])

            hdul = fits.open(inpath/filelist[i])
            data = copy.deepcopy(hdul[0].data)
            hdr = hdul[0].header
            hdul.close()

            del hdul[0].data
            if hdr[catkeyword] == 'SCIENCE':
                framename.append(filelist[i])
                header.append(hdr)
                type.append(hdr[catkeyword])
                texp=np.append(texp,hdr['EXPTIME'])
                date.append(hdr['DATE-OBS'])
                mjd=np.append(mjd,hdr['MJD-OBS'])
                npx=np.append(npx,hdr['NAXIS1'])
                norders=np.append(norders,hdr['NAXIS2'])
                e2ds.append(data)

                sci_count += 1
                berv=np.append(berv,hdr[bervkeyword])
                airmass=np.append(airmass,0.5*(hdr[Zstartkeyword]+hdr[Zendkeyword]))#This is an approximation where we take the mean airmass.

                if nowave == True:
                    #Record which wavefile was used by the pipeline to
                        #create the wavelength solution.
                    wavefile_used.append(hdr[thfilekeyword])

                    wavedata=ut.read_wave_from_HARPS_header(hdr,mode=mode)
                    wave.append(wavedata)
            # else:
                # berv=np.append(berv,np.nan)
                # airmass=np.append(airmass,np.nan)
        if filelist[i].endswith('wave_A.fits'):
            print(filelist[i]+' (wave)')
            if nowave == True:
                warnings.warn(" in read_HARPS_e2ds: nowave was set to True but a wave_A file was detected. This wave file is now ignored in favor of the header.",RuntimeWarning)
            else:
                wavedata=fits.getdata(inpath/filelist[i])
                wave.append(wavedata)
                wave_count += 1
        if filelist[i].endswith('blaze_A.fits'):
                print(filelist[i]+' (blaze)')
                blazedata=fits.getdata(inpath/filelist[i])
                blaze.append(blazedata)
                blaze_count += 1
        if filelist[i].endswith('s1d_A.fits'):
            hdul = fits.open(inpath/filelist[i])
            data_1d = copy.deepcopy(hdul[0].data)
            hdr = hdul[0].header
            hdul.close()
            del hdul[0].data
            if hdr[catkeyword] == 'SCIENCE':
                s1d.append(data_1d)
                s1dhdr.append(hdr)
                s1dmjd=np.append(s1dmjd,hdr['MJD-OBS'])
                s1d_count += 1
    #Now we catch some errors:
    #-The above should have read a certain number of e2ds files.
    #-A certain number of these should be SCIENCE frames.
    #-There should be at least one WAVE file.
    #-All exposures should have the same number of spectral orders.
    #-All orders should have the same number of pixels (this is true for HARPS).
    #-The wave frame should have the same dimensions as the order frames.
    #-If nowave is set, test that all frames used the same wave_A calibrator.
    #-The blaze file needs to have the same shape as the e2ds files.
    #-The number of s1d files should be the same as the number of e2ds files.



    if e2ds_count == 0:
        raise FileNotFoundError(f"in read_HARPS_e2ds: The input folder {str(inpath)} does not contain files ending in e2ds.fits.")
    if sci_count == 0:
        print('')
        print('')
        print('')
        print("These are the files and their types:")
        for i in range(len(type)):
            print('   '+framename[i]+'  %s' % type[i])
        raise ValueError("in read_HARPS_e2ds: The input folder (%2) contains e2ds files, but none of them are classified as SCIENCE frames with the HIERARCH ESO DPR CATG/OBS-TYPE keyword. The list of frames is printed above.")
    if np.max(np.abs(norders-norders[0])) == 0:
        norders=int(norders[0])
    else:
        print('')
        print('')
        print('')
        print("These are the files and their number of orders:")
        for i in range(len(type)):
            print('   '+framename[i]+'  %s' % norders[i])
        raise ValueError("in read_HARPS_e2ds: Not all files have the same number of orders. The list of frames is printed above.")
    if np.max(np.abs(npx-npx[0])) == 0:
        npx=int(npx[0])
    else:
        print('')
        print('')
        print('')
        print("These are the files and their number of pixels:")
        for i in range(len(type)):
            print('   '+framename[i]+'  %s' % npx[i])
        raise ValueError("in read_HARPS_e2ds: Not all files have the same number of pixels. The list of frames is printed above.")
    if wave_count >= 1:
        wave=wave[0]#SELECT ONLY THE FIRST WAVE FRAME. The rest is ignored.
    else:
        if nowave == False:
            print('')
            print('')
            print('')
            print("ERROR in read_HARPS_e2ds: No wave_A.fits file was detected.")
            print("These are the files in the folder:")
            for i in range(N):
                print(filelist[i])
            print("This may have happened if you downloaded the HARPS data from the")
            print("ADP query form, which doesn't include wave_A files (as far as I")
            print("have seen). Set the /nowave keyword in your call to read_HARPS_e2ds")
            print("if you indeed do not expect a wave_A file to be present.")
            raise FileNotFoundError("No wave_A.fits file was detected. More details are printed above.")
    if nowave == True:
        if all(x == wavefile_used[0] for x in wavefile_used):
            print("Nowave is set, and simple wavelength calibration extraction")
            print("works, as all files in the dataset used the same wave_A file.")
            wave=wave[0]
        else:
            print("WARNING IN read_HARPS_e2ds: Nowave is set, but not all files")
            print("in the dataset used the same wave_A file when the pipeline was")
            print("run. Catching this requres an interpolation step that is currently")
            print("not yet implemented. These are the filenames and their")
            print("wave_A file used:")
            for i in range(N-1):
                print('   '+framename[i]+'  %s' % wavefile_used[0])
            wave=wave[0]
            print("I ALLOW YOU TO CONTINUE BUT USING ONLY THE FIRST WAVELENGTH")
            print("SOLUTION. A PART OF THE DATA MAY BE AFFECTED BY HAVING ASSUMED")
            print("THE WRONG SOLUTION FILE. Though due to the stability of HARPS,")
            print("this MAY not be an issue.")
            raise Exception("")
    if blaze_count >= 1:
        blaze=blaze[0]#SELECT ONLY THE FIRST blaze FRAME. The rest is ignored.
    if np.shape(wave) != np.shape(e2ds[0]):
        raise ValueError(f"in read_HARPS_e2ds: A wave file was detected but its shape ({np.shape(wave)[0]},{np.shape(wave)[1]}) does not match that of the orders ({np.shape(e2ds[0])[0]},{np.shape(e2ds[0])[1]})")
    if np.shape(blaze) != np.shape(e2ds[0]) and blaze_count > 0:
        raise ValueError(f"in read_HARPS_e2ds: A blaze file was detected but its shape ({np.shape(blaze)[0]},{np.shape(wave)[1]}) does not match that of the orders ({np.shape(e2ds[0])[0]},{np.shape(e2ds[0])[1]})")
    if len(s1dhdr) != len(e2ds) and molecfit == True:
        raise ValueError(f'in read_HARPS_e2ds: The number of s1d SCIENCE files and e2ds SCIENCE files is not the same. ({len(s1dhdr)} vs {len(e2ds)})')


    #Ok, so now we should have ended up with a number of lists that contain all
    #the relevant information of our science frames.
    #We determine how to sort the resulting lists in time:
    sorting = np.argsort(mjd)
    s1dsorting = np.argsort(s1dmjd)

    if len(ignore_exp) > 0:
        sorting = [x for i,x in enumerate(sorting) if i not in ignore_exp]
        s1dsorting = [x for i,x in enumerate(s1dsorting) if i not in ignore_exp]


    if mode == 'HARPSN':
        for i in range(len(header)):
            s1dhdr[i]['TELALT'] = np.degrees(float(s1dhdr[i]['EL']))
            s1dhdr[i]['UTC'] = (float(s1dhdr[i]['MJD-OBS'])%1.0)*86400.0


    #First sort the s1d files for application of molecfit.
    if molecfit == True:
        if len(sorting) != len(s1dsorting):
            raise ValueError("in read_HARPS_e2ds: Sorted science frames and sorted s1d frames are not of the same length. Telluric correction can't proceed.")

        s1dhdr_sorted=[]
        s1d_sorted=[]
        for i in range(len(s1dsorting)):
            s1dhdr_sorted.append(s1dhdr[s1dsorting[i]])
            s1d_sorted.append(s1d[s1dsorting[i]])

        # print('Molecfit will be executed onto the files in this order:')
        # for x in s1dhdr_sorted:
        #     print(x['DATE-OBS'])
        list_of_wls,list_of_trans = mol.do_molecfit(s1dhdr_sorted,s1d_sorted,load_previous=False,mode=mode)

        if len(list_of_trans) != len(sorting):
            raise ValueError("ERROR in read_HARPS_e2ds: Molecfit did not produce the same number of spectra as there are in the e2ds spectra.")
        mol.write_telluric_transmission_to_file(list_of_wls,list_of_trans,outpath+'telluric_transmission_spectra.pkl')


    #Now we loop over all exposures and collect the i-th order from each exposure,
    #put these into a new matrix and save them to FITS images:
    f=open(outpath/'obs_times','w',newline='\n')
    headerline = 'MJD'+'\t'+'DATE'+'\t'+'EXPTIME'+'\t'+'MEAN AIRMASS'+'\t'+'BERV (km/s)'+'\t'+'FILE NAME'
    for i in range(norders):
        order = np.zeros((len(sorting),npx))
        wave_axis = wave[i,:]/10.0#Convert to nm.
        print('CONSTRUCTING ORDER %s' % i)
        c = 0#To count the number of science frames that have passed. The counter
        # c is not equal to j because the list of files contains not only SCIENCE
        # frames.

        for j in range(len(sorting)):#Loop over exposures
            if i ==0:
                print('---'+type[sorting[j]]+'  '+date[sorting[j]])
            if type[sorting[j]] == 'SCIENCE':#This check may be redundant.
                exposure = e2ds[sorting[j]]
                order[c,:] = exposure[i,:]
                #T_i = interp.interp1d(list_of_wls[j],list_of_trans[j])#This should be time-sorted, just as the e2ds files.
                #Do a manual check here that the MJDs are identical.
                #Also, determiine what to do with airtovac.
                #tel_order[c,:] = T_i[wave_axis]
                #Now I also need to write it to file.
                if i ==0:#Only do it the first time, not for every order.
                    line = str(mjd[sorting[j]])+'\t'+date[sorting[j]]+'\t'+str(texp[sorting[j]])+'\t'+str(np.round(airmass[sorting[j]],3))+'\t'+str(np.round(berv[sorting[j]]/1000.0,5))+'\t'+framename[sorting[j]]+'\n'
                    f.write(line)
                c+=1

        fits.writeto(outpath/('order_'+str(i)+'.fits'),order,overwrite=True)
        fits.writeto(outpath/('wave_'+str(i)+'.fits'),wave_axis,overwrite=True)
    f.close()
    print(f'Time-table written to {outpath/"obs_times"}.')
