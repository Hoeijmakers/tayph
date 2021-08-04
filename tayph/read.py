
import pkg_resources
import os
import pdb
from astropy.io import fits
import astropy.constants as const
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import sys
import tayph.util as ut
from tayph.vartests import typetest,dimtest
import tayph.tellurics as mol
import tayph.system_parameters as sp
import tayph.functions as fun
import tayph.operations as ops
from tayph.ccf import xcor
import copy
import scipy.interpolate as interp
import pickle
from pathlib import Path
import warnings
import glob
from scipy import interpolate
import tayph.masking as masking
import subprocess
import textwrap
from astropy.utils.data import download_file
from .phoenix import get_phoenix_wavelengths, get_phoenix_model_spectrum
from scipy.ndimage import uniform_filter1d


__all__ = [
    "read_harpslike",
    "read_espresso",
    "read_uves"
]



def read_harpslike(inpath,filelist,mode,read_s1d=True):
    """
    This reads a folder of HARPS or HARPSN data. Input is a list of filepaths and the mode (HARPS
    or HARPSN).
    """

    if mode=='HARPS':
        catkeyword = 'HIERARCH ESO DPR CATG'
        bervkeyword = 'HIERARCH ESO DRS BERV'
        thfilekeyword = 'HIERARCH ESO DRS CAL TH FILE'
        Zstartkeyword = 'HIERARCH ESO TEL AIRM START'
        Zendkeyword = 'HIERARCH ESO TEL AIRM END'
    elif mode=='HARPSN':
        catkeyword = 'OBS-TYPE'
        bervkeyword = 'HIERARCH TNG DRS BERV'
        thfilekeyword = 'HIERARCH TNG DRS CAL TH FILE'
        Zstartkeyword = 'AIRMASS'
        Zendkeyword = 'AIRMASS'#These are the same because HARPSN doesnt have start and end keywords.
        #Down there, the airmass is averaged, so there is no problem in taking the average of the same number.
    else:
        raise ValueError(f"Error in read_harpslike: mode should be set to HARPS or HARPSN ({mode})")

    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
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
    # wavefile_used = []
    for i in range(len(filelist)):
        if filelist[i].endswith('e2ds_A.fits'):
            print(f'------{filelist[i]}', end="\r")
            hdul = fits.open(inpath/filelist[i])
            data = copy.deepcopy(hdul[0].data)
            hdr = hdul[0].header
            hdul.close()
            del hdul[0].data
            if hdr[catkeyword] == 'SCIENCE':
                framename.append(filelist[i])
                header.append(hdr)
                obstype.append(hdr[catkeyword])
                texp=np.append(texp,hdr['EXPTIME'])
                date.append(hdr['DATE-OBS'])
                mjd=np.append(mjd,hdr['MJD-OBS'])
                npx=np.append(npx,hdr['NAXIS1'])
                norders=np.append(norders,hdr['NAXIS2'])
                e2ds.append(data)
                berv=np.append(berv,hdr[bervkeyword])
                airmass=np.append(airmass,0.5*(hdr[Zstartkeyword]+hdr[Zendkeyword]))#This is an approximation where we take the mean airmass.
                # if nowave == True:
                # wavefile_used.append(hdr[thfilekeyword])
                #Record which wavefile was used by the pipeline to
                #create the wavelength solution.
                wavedata=ut.read_wave_from_e2ds_header(hdr,mode=mode)/10.0#convert to nm.
                wave.append(wavedata)
                # if filelist[i].endswith('wave_A.fits'):
                #     print(filelist[i]+' (wave)')
                #     if nowave == True:
                #         warnings.warn(" in read_e2ds: nowave was set to True but a wave_A file was detected. This wave file is now ignored in favor of the header.",RuntimeWarning)
                #     else:
                #         wavedata=fits.getdata(inpath/filelist[i])
                #         wave.append(wavedata)

                if read_s1d:
                    s1d_path=inpath/Path(str(filelist[i]).replace('e2ds_A.fits','s1d_A.fits'))
                    ut.check_path(s1d_path,exists=True)#Crash if the S1D doesn't exist.
        # if filelist[i].endswith('s1d_A.fits'):
                    hdul = fits.open(s1d_path)
                    data_1d = copy.deepcopy(hdul[0].data)
                    hdr1d = hdul[0].header
                    hdul.close()
                    del hdul
            # if hdr[catkeyword] == 'SCIENCE':
                    s1d.append(data_1d)
                    if mode == 'HARPSN':#In the case of HARPS-N we need to convert the units of the
                        #elevation and provide a UTC keyword.
                        hdr1d['TELALT'] = np.degrees(float(hdr1d['EL']))
                        hdr1d['UTC'] = (float(hdr1d['MJD-OBS'])%1.0)*86400.0
                    s1dhdr.append(hdr1d)
                    s1dmjd=np.append(s1dmjd,hdr1d['MJD-OBS'])
                    berv1d = hdr1d[bervkeyword]
                    if berv1d != hdr[bervkeyword]:
                        wrn_msg = ('WARNING in read_harpslike(): BERV correction of s1d file is not'
                        f'equal to that of the e2ds file. {berv1d} vs {hdr[bervkeyword]}')
                        ut.tprint(wrn_msg)
                    gamma = (1.0-(berv1d*u.km/u.s/const.c).decompose().value)#Doppler factor BERV.
                    wave1d.append((hdr1d['CDELT1']*np.arange(len(data_1d), dtype=float)+hdr1d['CRVAL1'])*gamma)

    #Check that all exposures have the same number of pixels, and clip s1ds if needed.
    # min_npx1d = int(np.min(np.array(npx1d)))
    # if np.sum(np.abs(np.array(npx1d)-npx1d[0])) != 0:
    #     warnings.warn("in read_e2ds when reading HARPS data: Not all s1d files have the same number of pixels. This could have happened if the pipeline has extracted one or two extra pixels in some exposures but not others. The s1d files will be clipped to the smallest length.",RuntimeWarning)
    #     for i in range(len(s1d)):
    #         wave1d[i]=wave1d[i][0:min_npx1d]
    #         s1d[i]=s1d[i][0:min_npx1d]
    #         npx1d[i]=min_npx1d
    output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,
    'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
    'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    return(output)

def read_carmenes(inpath,filelist,channel,construct_s1d=True):
    """
    This reads a folder of CARMENES visible (VIS) or infra-red (NIR) channel data. Input is a list
    of filepaths and the mode ('VIS' or 'NIR').
    """



    catkeyword = 'HIERARCH CAHA INS ICS IMAGETYP'
    bervkeyword = 'HIERARCH CARACAL BERV'
    thfilekeyword = 'HIERARCH CARACAL WAVE FILE'
    Zstartkeyword = 'AIRMASS'
    Zendkeyword = 'AIRMASS'#These are the same because CARMENES doesnt have start and end keywords.

    if channel not in ['vis','nir']:
        raise ValueError(f"Error in read_carmenes: channel should be set to VIS or NIR ({channel})")

    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    norders=np.array([])
    e2ds=[]
    blaze=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    # wavefile_used = []
    for i in range(len(filelist)):
        if filelist[i].endswith('-'+channel.lower()+'_A.fits'):
            print(f'------{filelist[i]}', end="\r")
            hdul = fits.open(inpath/filelist[i])
            data = copy.deepcopy(hdul[1].data)
            cont = copy.deepcopy(hdul[2].data)
            sigma = copy.deepcopy(hdul[3].data)
            wavedata = copy.deepcopy(hdul[4].data)/10.0
            hdr = hdul[0].header
            spechdr = hdul[1].header
            hdul.close()
            del hdul[1].data
            del hdul[2].data
            del hdul[3].data
            del hdul[4].data
            if hdr[catkeyword] == 'SCIENCE' or hdr[catkeyword] == "Science":
                framename.append(filelist[i])
                header.append(hdr)
                obstype.append(hdr[catkeyword])
                texp=np.append(texp,hdr['EXPTIME'])
                date.append(hdr['DATE-OBS'])
                mjd=np.append(mjd,hdr['MJD-OBS'])
                npx=np.append(npx,spechdr['NAXIS1'])
                norders=np.append(norders,spechdr['NAXIS2'])
                e2ds.append(data)
                blaze.append(data/sigma**2)
                berv=np.append(berv,hdr[bervkeyword])
                airmass=np.append(airmass,0.5*(hdr[Zstartkeyword]+hdr[Zendkeyword]))#This is an approximation where we take the mean airmass.
                wave.append(ops.vactoair(wavedata))


                if construct_s1d:
                    wave_1d, data_1d = spec_stich_n_norm(data,wavedata,cont,sigma)

                    s1d.append(data_1d)

                    hdr1d = copy.deepcopy(hdr)
                    hdr1d['UTC'] = (float(hdr1d['MJD-OBS'])%1.0)*86400.0
                    s1dhdr.append(hdr1d)
                    s1dmjd=np.append(s1dmjd,hdr1d['MJD-OBS'])
                    berv1d = hdr1d[bervkeyword]
                    gamma = (1.0-(berv1d*u.km/u.s/const.c).decompose().value)#Doppler factor BERV.
                    gamma = 1 #turning berv off
                    wave1d.append(ops.vactoair(wave_1d)*gamma*10)

    BLAZE_Model = blaze_model(np.nanmean(blaze, axis = 0)) #Calucalting blaze model
    e2ds = list(e2ds*BLAZE_Model[np.newaxis,:]) #Deblazing the data

    if construct_s1d:
        output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,
        'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
        'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    else:
        output = {'wave':wave,'e2ds':e2ds,'header':header,
        'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
        'norders':norders,'berv':berv,'airmass':airmass}
    return(output)


def spec_stich_n_norm(spec, wave, cont, sig):
    """This stitches and continuum normalises CARMENES E2DS spectra into 1D spectra for use in
    molecfit.
    N. Borsato - 24-02-2021"""

    import numpy as np
    from astropy.io import fits
    from scipy.interpolate import interp1d

    #These arrays will be filled with the stiched data.
    Total_Specs = np.array([])
    Total_Waves = np.array([])
    Total_Cont = np.array([])

    step_size = np.diff(wave)
    step_size = np.min(step_size[step_size>0])/2

    for i in range(len(spec)-1):

        waves = np.linspace(np.min(wave[i]), np.max(wave[i+1]), 4*wave[i:i+1].size) #Specifiying wavelength grid

        #Interpolating the spectral orders to the new grid for both the current and proceeding order
        I_spectra_1 = interp1d(wave[i], spec[i], bounds_error = False) #Intep for spectra
        I_spectra_2 = interp1d(wave[i+1], spec[i+1], bounds_error = False)

        I_sig_1 = interp1d(wave[i], sig[i], bounds_error = False) #Interp for sig vals
        I_sig_2 = interp1d(wave[i+1], sig[i+1], bounds_error = False)

        I_cont_1 = interp1d(wave[i], cont[i], bounds_error = False) #Interp for continuum vals
        I_cont_2 = interp1d(wave[i+1], cont[i+1], bounds_error = False)

        #Using interpolator to create a vecotr of the same length for both orders.
        ## Note: If the order doesn't span the wavelength range it creates a nan
        I_spectra_1 = I_spectra_1(waves)
        I_spectra_2 = I_spectra_2(waves)

        I_sig_1 = I_sig_1(waves)
        I_sig_2 = I_sig_2(waves)

        I_cont_1 = I_cont_1(waves)
        I_cont_2 = I_cont_2(waves)

        #If a nan value is present, replace the value with the corresponding value of the other interpolated vecotor
        ## This creates an overlapping vecotr of the same value, which means you can take an average and get the same value
        I_spectra_1 = np.nan_to_num(I_spectra_1, nan = I_spectra_2)
        I_spectra_2 = np.nan_to_num(I_spectra_2, nan = I_spectra_1)

        I_sig_1 = np.nan_to_num(I_sig_1, nan = I_sig_2)
        I_sig_2 = np.nan_to_num(I_sig_2, nan = I_sig_1)

        I_cont_1 = np.nan_to_num(I_cont_1, nan = I_cont_2)
        I_cont_2 = np.nan_to_num(I_cont_2, nan = I_cont_1)

        Spec_Combo = np.array([I_spectra_1, I_spectra_2]) #Combing spectra as 2d array
        Spec_Combo = Spec_Combo.T #Take transpose to pair the data values

        #Do the same for the sig values and continuum
        Sig_Combo = np.array([I_sig_1, I_sig_2])
        Sig_Combo = Sig_Combo.T

        Cont_Combo = np.array([I_cont_1, I_cont_2])
        Cont_Combo = Cont_Combo.T

        #Money is made here, a weighted average is taken using the sig values than normalise by dividing by the continuu,
        Ave_Spec = np.average(Spec_Combo, weights = 1/(Sig_Combo**2), axis = 1)/np.average(Cont_Combo, axis = 1)
        Ave_Spec = Ave_Spec.T

        #Averages are append with their corresponding wavlength
        Total_Specs = np.append(Total_Specs, Ave_Spec)
        Total_Waves = np.append(Total_Waves, waves)
        #Total_Cont = np.append(Total_Cont, np.average(Cont_Combo, axis = 1))

    #Reaorder values to a new grid to create the final stiched spectra
    waves = np.arange(np.min(Total_Waves), np.max(Total_Waves), step_size)
    I_T_Spectra = interp1d(Total_Waves, Total_Specs)

    return waves, I_T_Spectra(waves) #returns wavelength grid and the normalised flux values

def blaze_model(blaze,sdev=3):
    """Applies running average fit of the blaze order data.
        args:
            blaze: the average escelle blaze data
            sdev: stanrard deviation cutoff

        returns:
            a: the mean blaze model
    """

    def nan_helper(data): #Function which allows interpolation though nans
        return lambda z: z.nonzero()[0]

    blaze = blaze.copy()
    data = blaze.copy()

    nan_mask = np.isnan(blaze)
    no_nan = nan_helper(blaze)

    #This will re-create the dataset but will interpolate though the nans giving it a place holder average value
    blaze[nan_mask]= np.interp(no_nan(nan_mask), no_nan(~nan_mask), blaze[~nan_mask], period = 1)

    a = uniform_filter1d(blaze,size=300,mode="nearest")#Applies moving averages on data

    #Takes difference the replaces datavalues which fall less 3 std of mean trend with mean
    diff = blaze - a
    cleaned_blaze_1 = blaze
    cleaned_blaze_1[diff<-sdev*np.std(diff)] = a[diff<-sdev*np.std(diff)]

    #Repeats process on the new data but masks out datavalues which fall outside 3std in both directions
    a = uniform_filter1d(cleaned_blaze_1,size=300,mode="nearest")
    diff = blaze - a
    cleaned_blaze_2 = cleaned_blaze_1
    cleaned_blaze_2[np.absolute(diff)>sdev*np.std(diff)] = a[np.absolute(diff)>sdev*np.std(diff)]
    a = uniform_filter1d(cleaned_blaze_1,size=300,mode="nearest")

    #Replace all positions that contanined nan values initially with nans again
    a[np.isnan(data)] = data[np.isnan(data)]

    return a



def read_uves(inpath,filelist,mode):
    """This reads a folder of UVES-blue or UVES-red intermediate pipeline products. Input is a list of filepaths and the mode."""
    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    npx1d=np.array([])
    norders=np.array([])
    e2ds=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    catkeyword = 'HIERARCH ESO DPR CATG'
    bervkeyword = 'HIERARCH ESO DRS BERV'
    thfilekeyword = 'HIERARCH ESO DRS CAL TH FILE'
    Zstartkeyword = 'HIERARCH ESO TEL AIRM START'
    Zendkeyword = 'HIERARCH ESO TEL AIRM END'
    for i in range(len(filelist)):
        print(f'------{filelist[i]}', end="\r")
        if (inpath/Path(filelist[i])).is_dir():
            tmp_products = [i for i in (inpath/Path(filelist[i])).glob('resampled_science_*.fits')]
            tmp_products1d = [i for i in (inpath/Path(filelist[i])).glob('red_science_*.fits')]
            if mode == 'UVES-red' and len(tmp_products) != 2:
                raise ValueError(f"in read_e2ds: When mode=UVES-red there should be 2 resampled_science files (redl and redu), but {len(tmp_products)} were detected in {str(inpath/Path(filelist[i]))}.")
            if mode == 'UVES-blue' and len(tmp_products) != 1:
                raise ValueError(f"in read_e2ds: When mode=UVES-rblue there should be 1 resampled_science files (blue), but {len(tmp_products)} were detected in {str(inpath/Path(filelist[i]))}.")
            if mode == 'UVES-red' and len(tmp_products1d) != 2:
                raise ValueError(f"in read_e2ds: When mode=UVES-red there should be 2 red_science files (redl and redu), but {len(tmp_products1d)} were detected in {str(inpath/Path(filelist[i]))}.")
            if mode == 'UVES-blue' and len(tmp_products1d) != 1:
                raise ValueError(f"in read_e2ds: When mode=UVES-rblue there should be 1 red_science files (blue), but {len(tmp_products1d)} were detected in {str(inpath/Path(filelist[i]))}.")

            data_combined = []#This will store the two chips (redu and redl) in case of UVES_red, or simply the blue chip if otherwise.
            wave_combined = []
            wave1d_combined=[]
            data1d_combined=[]
            norders_tmp = 0
            for tmp_product in tmp_products:
                hdul = fits.open(tmp_product)
                data = copy.deepcopy(hdul[0].data)
                hdr = hdul[0].header
                hdul.close()
                del hdul[0].data
                if not hdr['HIERARCH ESO PRO SCIENCE']:#Only add if it's actually a science product:#I force the user to supply only science exposures in the input  folder. No BS allowed... UVES is hard enough as it is.
                    raise ValueError(f' in read_e2ds: UVES file {tmp_product} is not classified as a SCIENCE file, but should be. Remove it from the folder?')
                wavedata=ut.read_wave_from_e2ds_header(hdr,mode='UVES')/10.0#Convert to nm.
                data_combined.append(data)
                wave_combined.append(wavedata)
                norders_tmp+=np.shape(data)[0]

            for tmp_product in tmp_products1d:
                hdul = fits.open(tmp_product)
                data_1d = copy.deepcopy(hdul[0].data)
                hdr1d = hdul[0].header
                hdul.close()
                del hdul[0].data
                if not hdr1d['HIERARCH ESO PRO SCIENCE']:#Only add if it's actually a science product:#I force the user to supply only science exposures in the input  folder. No BS allowed... UVES is hard enough as it is.
                    raise ValueError(f' in read_e2ds: UVES file {tmp_product} is not classified as a SCIENCE file, but should be. Remove it from the folder?')
                npx_1d = hdr1d['NAXIS1']
                wavedata = np.arange(npx_1d, dtype=float)*hdr1d['CDELT1']+hdr1d['CRVAL1']
                data1d_combined.append(data_1d)
                wave1d_combined.append(wavedata)

            if len(data_combined) < 1 or len(data_combined) > 2:#Double-checking that length here...
                raise ValueError(f'in read_e2ds(): Expected 1 or 2 chips, but {len(data_combined)} files were somehow read.')
            #The chips generally don't give the same size. Therefore I will pad the smaller one with NaNs to make it fit:
            if len(data_combined) != len(data1d_combined):
                raise ValueError(f'in read_e2ds(): The number of chips in the 1d and 2d spectra is not the same {len(data1d_combined)} vs {len(data_combined)}.')

            if len(data_combined) == 2:
                chip1 = data_combined[0]
                chip2 = data_combined[1]
                wave1 = wave_combined[0]
                wave2 = wave_combined[1]
                npx_1 = np.shape(chip1)[1]
                npx_2 = np.shape(chip2)[1]
                no_1 = np.shape(chip1)[0]
                no_2 = np.shape(chip2)[0]
                npx_max = np.max([npx_1,npx_2])
                npx_min = np.min([npx_1,npx_2])
                diff = npx_max-npx_min
                #Pad the smaller one with NaNs to match the wider one:
                if npx_1 < npx_2:
                    chip1=np.hstack([chip1,np.zeros((no_1,diff))*np.nan])
                    wave1=np.hstack([wave1,np.zeros((no_1,diff))*np.nan])
                else:
                    chip2=np.hstack([chip2,np.zeros((no_2,diff))*np.nan])
                    wave2=np.hstack([wave2,np.zeros((no_2,diff))*np.nan])
                #So now they can be stacked:
                e2ds_stacked = np.vstack((chip1,chip2))
                wave_stacked = np.vstack((wave1,wave2))
                if np.shape(e2ds_stacked)[1] != np.shape(wave_stacked)[1]:
                    raise ValueError("Width of stacked e2ds and stacked wave frame are not the same. Is the wavelength solution in the header of this file correct?")
                npx=np.append(npx,np.shape(e2ds_stacked)[1])

                e2ds.append(e2ds_stacked)
                wave.append(wave_stacked)
                chip1_1d = data1d_combined[0]
                chip2_1d = data1d_combined[1]
                wave1_1d = wave1d_combined[0]
                wave2_1d = wave1d_combined[1]
                if np.nanmean(wave1_1d) < np.nanmean(wave2_1d):
                    combined_data_1d = np.concatenate((chip1_1d,chip2_1d))
                    combined_wave_1d = np.concatenate((wave1_1d,wave2_1d))
                else:
                    combined_data_1d = np.concatenate((chip2_1d,chip1_1d))
                    combined_wave_1d = np.concatenate((wave2_1d,wave1_1d))
                wave1d.append(combined_wave_1d)
                s1d.append(combined_data_1d)
                npx1d=np.append(npx1d,len(combined_wave_1d))
            else:
                e2ds.append(data_combined[0])
                wave.append(wave_combined[0])
                npx=np.append(npx,np.shape(data_combined[0])[1])
                wave1d.append(wave1d_combined[0])
                s1d.append(data1d_combined[0])
                npx1d=np.append(npx1d,len(combined_wave_1d))
            #Only using the keyword from the second header in case of redl,redu.
            s1dmjd=np.append(s1dmjd,hdr1d['MJD-OBS'])
            framename.append(hdr['ARCFILE'])
            header.append(hdr)
            obstype.append('SCIENCE')
            texp=np.append(texp,hdr['EXPTIME'])
            date.append(hdr['DATE-OBS'])
            mjd=np.append(mjd,hdr['MJD-OBS'])
            norders=np.append(norders,norders_tmp)
            airmass=np.append(airmass,0.5*(hdr[Zstartkeyword]+hdr[Zendkeyword]))#This is an approximation where we take the mean airmass.
            berv_i=sp.calculateberv(hdr['MJD-OBS'],hdr['HIERARCH ESO TEL GEOLAT'],hdr['HIERARCH ESO TEL GEOLON'],hdr['HIERARCH ESO TEL GEOELEV'],hdr['RA'],hdr['DEC'])
            berv = np.append(berv,berv_i)
            hdr1d['HIERARCH ESO QC BERV']=berv_i#Append the berv here using the ESPRESSO berv keyword, so that it can be used in molecfit later.
            s1dhdr.append(hdr1d)

    #Check that all exposures have the same number of pixels, and clip orders if needed.
    min_npx = int(np.min(np.array(npx)))
    min_npx1d = int(np.min(np.array(npx1d)))
    if np.sum(np.abs(np.array(npx)-npx[0])) != 0:
        warnings.warn("in read_e2ds when reading UVES data: Not all e2ds files have the same number of pixels. This could have happened if the pipeline has extracted one or two extra pixels in some exposures but not others. The e2ds files will be clipped to the smallest width.",RuntimeWarning)
        for i in range(len(e2ds)):
            wave[i]=wave[i][:,0:min_npx]
            e2ds[i]=e2ds[i][:,0:min_npx]
            npx[i]=min_npx
    if np.sum(np.abs(np.array(npx1d)-npx1d[0])) != 0:
        warnings.warn("in read_e2ds when reading UVES data: Not all s1d files have the same number of pixels. This could have happened if the pipeline has extracted one or two extra pixels in some exposures but not others. The s1d files will be clipped to the smallest width.",RuntimeWarning)
        for i in range(len(s1d)):
            wave1d[i]=wave1d[i][0:min_npx1d]
            s1d[i]=s1d[i][0:min_npx1d]
            npx1d[i]=min_npx1d
    output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,'npx1d':npx1d,'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    return(output)



def read_espresso(inpath,filelist,read_s1d=True,skysub=True):
    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
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
    catkeyword = 'EXTNAME'
    bervkeyword = 'HIERARCH ESO QC BERV'
    airmass_keyword1 = 'HIERARCH ESO TEL'
    airmass_keyword2 = ' AIRM '
    airmass_keyword3_start = 'START'
    airmass_keyword3_end = 'END'

    if skysub:
        type_suffix = 'S2D_SKYSUB_A.fits'
    else:
        type_suffix = 'S2D_BLAZE_A.fits'

    for i in range(len(filelist)):
        if filelist[i].endswith(type_suffix):
            hdul = fits.open(inpath/filelist[i])
            data = copy.deepcopy(hdul[1].data)
            hdr = hdul[0].header
            hdr2 = hdul[1].header
            wavedata=copy.deepcopy(hdul[5].data)
            hdul.close()
            del hdul

            if hdr2[catkeyword] == 'SCIDATA':
                # print('science keyword found')
                print(f'------{filelist[i]}', end="\r")
                framename.append(filelist[i])
                header.append(hdr)
                obstype.append('SCIENCE')
                texp=np.append(texp,hdr['EXPTIME'])
                date.append(hdr['DATE-OBS'])
                mjd=np.append(mjd,hdr['MJD-OBS'])
                npx=np.append(npx,hdr2['NAXIS1'])
                norders=np.append(norders,hdr2['NAXIS2'])
                e2ds.append(data)
                berv=np.append(berv,hdr[bervkeyword])#in km.s.
                telescope = hdr['TELESCOP'][-1]
                airmass = np.append(airmass,0.5*(hdr[airmass_keyword1+telescope+' AIRM START']+hdr[airmass_keyword1+telescope+' AIRM END']))
                wave.append(wavedata/10.0)#*(1.0-(hdr[bervkeyword]*u.km/u.s/const.c).decompose().value))
                #Ok.! So unlike HARPS, ESPRESSO wavelengths are actually BERV corrected in the S2Ds.
                #WHY!!!?. WELL SO BE IT. IN ORDER TO HAVE E2DSes THAT ARE ON THE SAME GRID, AS REQUIRED, WE UNDO THE BERV CORRECTION HERE.
                #WHEN COMPARING WAVE[0] WITH WAVE[1], YOU SHOULD SEE THAT THE DIFFERENCE IS NILL.
                #THATS WHY LATER WE CAN JUST USE WAVE[0] AS THE REPRESENTATIVE GRID FOR ALL.
                #BUT THAT IS SILLY. JUST SAVE THE WAVELENGTHS!

                if read_s1d:
                    s1d_path=inpath/Path(str(filelist[i]).replace('_'+type_suffix,'_S1D_A.fits'))
                    #Need the blazed files. Not the S2D_A's by themselves.
                    ut.check_path(s1d_path,exists=True)#Crash if the S1D doesn't exist.
                    hdul = fits.open(s1d_path)
                    data_table = copy.deepcopy(hdul[1].data)
                    hdr1d = hdul[0].header
                    hdul.close()
                    del hdul
                    s1d.append(data_table.field(2))

                    berv1d = hdr1d[bervkeyword]
                    if berv1d != hdr[bervkeyword]:
                        wrn_msg = ('WARNING in read_espresso(): BERV correction of S1D file is not'
                        f'equal to that of the S2D file. {berv1d} vs {hdr[bervkeyword]}')
                        ut.tprint(wrn_msg)
                    gamma = (1.0-(berv1d*u.km/u.s/const.c).decompose().value)
                    wave1d.append(data_table.field(1)*gamma)#This is in angstroms.
                    #We need to check to which UT ESPRESSO was connected, so that we can read
                    #the weather information (which is UT-specific) and parse them into the
                    #header using UT-agnostic keywords that are in the ESPRESSO.par file.
                    TELESCOP = hdr1d['TELESCOP'].split('U')[1]#This is the number of the UT, either 1, 2, 3 or 4.
                    if TELESCOP not in ['1','2','3','4']:
                        raise ValueError(f"in read_e2ds when reading ESPRESSO data. The UT telescope is not recognised. (TELESCOP={hdr['TELESCOP']})")
                    else:
                        hdr1d['TELALT']     = hdr1d[f'ESO TEL{TELESCOP} ALT']
                        hdr1d['RHUM']       = hdr1d[f'ESO TEL{TELESCOP} AMBI RHUM']
                        hdr1d['PRESSURE']   = (hdr1d[f'ESO TEL{TELESCOP} AMBI PRES START']+
                                            hdr1d[f'ESO TEL{TELESCOP} AMBI PRES END'])/2.0
                        hdr1d['AMBITEMP']   = hdr1d[f'ESO TEL{TELESCOP} AMBI TEMP']
                        hdr1d['M1TEMP']     = hdr1d[f'ESO TEL{TELESCOP} TH M1 TEMP']
                    s1dhdr.append(hdr1d)
                    s1dmjd=np.append(s1dmjd,hdr1d['MJD-OBS'])
    if read_s1d:
        output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,
        'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
        'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    else:
        output = {'wave':wave,'e2ds':e2ds,'header':header,
        'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
        'norders':norders,'berv':berv,'airmass':airmass}
    return(output)
