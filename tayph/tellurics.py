__all__ = [
    "do_molecfit",
    "execute_molecfit",
    "remove_output_molecfit",
    "retrieve_output_molecfit",
    "write_file_to_molecfit",
    "molecfit_gui",
    "write_telluric_transmission_to_file",
    "read_telluric_transmission_from_file",
    "apply_telluric_correction"
]



def do_molecfit(headers,spectra,configfile,wave=[],mode='HARPS',load_previous=False,save_individual=''):
    """This is the main wrapper for molecfit that pipes a list of s1d spectra and
    executes it. It first launces the molecfit gui on the middle spectrum of the
    sequence, and then loops through the entire list, returning the transmission
    spectra of the Earths atmosphere in the same order as the list provided.
    These can then be used to correct the s1d spectra or the e2ds spectra.
    Note that the s1d spectra are assumed to be in the barycentric frame in vaccuum,
    but that the output transmission spectrum is in the observers frame, and e2ds files
    are in air wavelengths by default.

    If you have run do_molecfit before, and want to reuse the output of the previous run
    for whatever reason (i.e. due to a crash), set the load_previous keyword to True.
    This will reload the list of transmission spectra created last time, if available.

    You can also set save_individual to a path to an existing folder to which the
    transmission spectra of the time-series can be written one by one.
    """

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
    import astropy.io.fits as fits
    configfile=ut.check_path(configfile,exists=True)

    molecfit_input_folder=sp.paramget('molecfit_input_folder',configfile,full_path=True)
    molecfit_prog_folder=sp.paramget('molecfit_prog_folder',configfile,full_path=True)
    temp_specname = copy.deepcopy(mode)#The name of the temporary file used (without extension).
    #The spectrum will be named like this.fits There should be a this.par file as well,
    #that contains a line pointing molecfit to this.fits:
    parname=temp_specname+'.par'


    #====== ||  START OF PROGRAM   ||======#
    N = len(headers)
    if N != len(spectra):
        raise RuntimeError(f' in prep_for_molecfit: Length of list of headers is not equal to length of list of spectra ({N},{len(spectra)})')

    #Test that the input root and molecfit roots exist; that the molecfit root contains the molecfit executables.
    #that the input root contains the desired parfile and later fitsfile.
    molecfit_input_root=Path(molecfit_input_folder)
    molecfit_prog_root=Path(molecfit_prog_folder)

    ut.check_path(molecfit_input_root,exists=True)
    ut.check_path(molecfit_prog_root,exists=True)
    ut.check_path(molecfit_input_root/parname,exists=True)#Test that the molecfit parameter file for this mode exists.
    ut.check_path(molecfit_prog_root/'molecfit',exists=True)
    ut.check_path(molecfit_prog_root/'molecfit_gui',exists=True)
    if len(save_individual) > 0:
        ut.check_path(save_individual,exists=True)

    pickle_outpath = molecfit_input_root/'previous_run_of_do_molecfit.pkl'


    if load_previous == True:
        if os.path.isfile(pickle_outpath) ==  False:
            print('WARNING in do_molecfit(): Previously saved run was asked for but is not available.')
            print('The program will proceed to re-fit. That run will then be saved.')
            load_previous = False
        else:
            pickle_in = open(pickle_outpath,"rb")
            list_of_wls,list_of_fxc,list_of_trans = pickle.load(pickle_in)

    if load_previous == False:
        list_of_wls = []
        list_of_fxc = []
        list_of_trans = []

        middle_i = int(round(0.5*N))#We initialize molecfit on the middle spectrum of the time series.
        write_file_to_molecfit(molecfit_input_root,temp_specname+'.fits',headers,spectra,middle_i,mode=mode,wave=wave)
        print(molecfit_input_root)
        print(temp_specname+'.fits')
        execute_molecfit(molecfit_prog_root,molecfit_input_root/parname,gui=True)
        wl,fx,trans = retrieve_output_molecfit(molecfit_input_root/temp_specname)
        remove_output_molecfit(molecfit_input_root,temp_specname)
        for i in range(N):#range(len(spectra)):
            print('Fitting spectrum %s from %s' % (i+1,len(spectra)))
            t1=ut.start()
            write_file_to_molecfit(molecfit_input_root,temp_specname+'.fits',headers,spectra,i,mode=mode,wave=wave)
            execute_molecfit(molecfit_prog_root,molecfit_input_root/parname,gui=False)
            wl,fx,trans = retrieve_output_molecfit(molecfit_input_root/temp_specname)
            remove_output_molecfit(molecfit_input_root,temp_specname)
            list_of_wls.append(wl*1000.0)#Convert to nm.
            list_of_fxc.append(fx/trans)
            list_of_trans.append(trans)
            ut.end(t1)
            if len(save_individual) > 0:
                indv_outpath=Path(save_individual)/f'tel_{i}.fits'
                indv_out = np.zeros((2,len(trans)))
                indv_out[0]=wl*1000.0
                indv_out[1]=trans
                fits.writeto(indv_outpath,indv_out)



        pickle_outpath = molecfit_input_root/'previous_run_of_do_molecfit.pkl'
        with open(pickle_outpath, 'wb') as f: pickle.dump((list_of_wls,list_of_fxc,list_of_trans),f)

    to_do_manually = check_fit_gui(list_of_wls,list_of_fxc,list_of_trans)
    if len(to_do_manually) > 0:
        print('The following spectra were selected to redo manually:')
        print(to_do_manually)
        #CHECK THAT THIS FUNCIONALITY WORKS:
        for i in to_do_manually:
            write_file_to_molecfit(molecfit_input_root,temp_specname+'.fits',headers,spectra,int(i),mode=mode,wave=wave)
            execute_molecfit(molecfit_prog_root,molecfit_input_root/parname,gui=True)
            wl,fx,trans = retrieve_output_molecfit(molecfit_input_root/temp_specname)
            list_of_wls[int(i)] = wl*1000.0#Convert to nm.
            list_of_fxc[int(i)] = fxc
            list_of_trans[int(i)] = trans
    return(list_of_wls,list_of_trans)


def remove_output_molecfit(path,name):
    """This cleans the molecfit project folder"""
    import os
    from pathlib import Path
    root = Path(path)
    files = ['_out.fits','_out_gui.fits','_out_molecfit_out.txt','_out_fit.par','_out_fit.atm','_out_apply_out.txt','_out_fit.atm.fits','_out_fit.res','_out_tac.asc','_TAC.fits','_out_fit.res.fits','_out_fit.asc','_out_fit.fits']

    for f in files:
        try:
            os.remove(root/(name+f))
        except:
            pass
    return


def retrieve_output_molecfit(path):
    """This collects the actual transmission model created after a pass through
    molecfit is completed"""
    import astropy.io.fits as fits
    import os.path
    import sys
    from pathlib import Path
    import tayph.util as ut
    file = Path(str(Path(path))+'_out_tac.fits')
    ut.check_path(file,exists=True)

    with fits.open(file) as hdul:
        wl=hdul[1].data['lambda']
        fx=hdul[1].data['flux']
        trans=hdul[1].data['mtrans']
    return(wl,fx,trans)

def execute_molecfit(molecfit_prog_root,molecfit_input_file,gui=False):
    """This actually calls the molecfit command in bash"""
    import os
    from pathlib import Path
    if gui == False:
        command = str(Path(molecfit_prog_root)/'molecfit')+' '+str(molecfit_input_file)
        command2 = str(Path(molecfit_prog_root)/'calctrans')+' '+str(molecfit_input_file)
        os.system(command)
        os.system(command2)
    if gui == True:
        command = 'python3 '+str(molecfit_prog_root/'molecfit_gui')+' '+str(molecfit_input_file)
        os.system(command)
    #python3 /Users/hoeijmakers/Molecfit/bin/molecfit_gui /Users/hoeijmakers/Molecfit/share/molecfit/spectra/cross_cor/test.par

def write_file_to_molecfit(molecfit_file_root,name,headers,spectra,ii,mode='HARPS',wave=[]):
    """This is a wrapper for writing a spectrum from a list to molecfit format.
    name is the filename of the fits file that is the output.
    headers is the list of astropy header objects associated with the list of spectra
    in the spectra variable. ii is the number from that list that needs to be written (meaning
    that this routine is expected to be called as part of a loop).

    The mode keyword will determine how to handle the spectra provided, notably what to do
    with the FITS headers and the wavelengths.

    In the case of HARPS and ESPRESSO, s1d spectra are normally in air and in the barycenter.
    Using the BERV correction in the header, they are shifted back to the Earths frame to put
    the tellurics back to 0km/s.

    An arbitrary mode can be set, but in that case the user must make sure that their wavelength
    axes are provided correctly, either encoded in the FITS header in the observatory restframe;
    or via the wave keyword.

    The wave keyword is used for when the s1d headers do not contain wavelength information like HARPS does.
    (for instance, ESPRESSO). The wave keyword needs to be set in this case, to the wavelength array as extracted from FITS files or smth.
    If you do that for HARPS or HARPSN and also set the wave keyword, this code will still
    grab it from the header and ignore it.
    """
    import astropy.io.fits as fits
    from scipy import stats
    import copy
    import tayph.functions as fun
    import astropy.constants as const
    import astropy.units as u
    import numpy as np
    from tayph.vartests import typetest
    import tayph.util as ut
    import sys
    typetest(ii,int,'ii write_file_to_molecfit')
    molecfit_file_root=ut.check_path(molecfit_file_root,exists=True)
    spectrum = spectra[int(ii)]
    npx = len(spectrum)

    if mode == 'HARPS':
        berv = headers[ii]['HIERARCH ESO DRS BERV']#Need to un-correct the s1d spectra to go back to the frame of the Earths atmosphere.
        wave = (headers[ii]['CDELT1']*fun.findgen(len(spectra[ii]))+headers[ii]['CRVAL1'])*(1.0-(berv*u.km/u.s/const.c).decompose().value)
    elif mode == 'HARPSN':
        berv = headers[ii]['HIERARCH TNG DRS BERV']#Need to un-correct the s1d spectra to go back to the frame of the Earths atmosphere.
        wave = (headers[ii]['CDELT1']*fun.findgen(len(spectra[ii]))+headers[ii]['CRVAL1'])*(1.0-(berv*u.km/u.s/const.c).decompose().value)
    elif mode in ['ESPRESSO','UVES-red','UVES-blue']:
        if len(wave) == 0:
            raise ValueError('in write_file_to_molecfit(): When mode in [ESPRESSO,UVES-red,UVES-blue], the 1D wave axis needs to be provided.')
        #WAVE VARIABLE NEEDS TO BE PASSED NOW.
        berv = headers[ii]['HIERARCH ESO QC BERV']#Need to un-correct the s1d spectra to go back to the frame of the Earths atmosphere.
        wave = copy.deepcopy(wave*(1.0-(berv*u.km/u.s/const.c).decompose().value))

    err = np.sqrt(spectrum)

    #Write out the s1d spectrum in a format that molecfit eats.
    #This is a fits file with an empty primary extension that contains the header of the original s1d file.
    #Plus an extension that contains a binary table with 3 columns.
    #The names of these columns need to be indicated in the molecfit parameter file,
    #as well as the name of the file itself. This is currently hardcoded.
    col1 = fits.Column(name = 'wavelength', format = '1D', array = wave)
    col2 = fits.Column(name = 'flux', format       = '1D', array = spectrum)
    col3 = fits.Column(name = 'err_flux', format   = '1D', array = err)
    cols = fits.ColDefs([col1, col2, col3])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    prihdr = fits.Header()
    prihdr = copy.deepcopy(headers[ii])
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])
    thdulist.writeto(molecfit_file_root/name,overwrite=True)
    print(f'Spectrum {ii} written to {str(molecfit_file_root/name)}')
    return(0)










class molecfit_gui(object):
    """This class defines most of the behaviour of the GUI to inspect the molecfit output."""
    def __init__(self,wls,fxc,trans):
        import matplotlib.pyplot as plt
        import math
        self.wls=wls
        self.fxc=fxc
        self.trans=trans
        self.i = 0#The current spectrum.
        self.N = len(wls)#total number of spectra.
        self.fig,self.ax = plt.subplots(2,1,sharex=True,figsize=(14,6))
        self.maxboxes = 20.0
        self.nrows = math.ceil(self.N/self.maxboxes)#number of rows
        plt.subplots_adjust(left=0.05)#Create space for the interface (#rhyme comments ftw <3)
        plt.subplots_adjust(right=0.75)
        plt.subplots_adjust(bottom=0.1+0.05*self.nrows)
        plt.subplots_adjust(top=0.95)
        self.set_spectrum(self.i)
        self.ax[0].set_title('Corrected / uncorrected s1d #%s/%s' % (self.i,self.N-1))#start counting at 0.
        self.ax[1].set_title('Transmission spectrum')
        self.img1a=self.ax[0].plot(wls[self.i],fxc[self.i]*trans[self.i],alpha=0.5)
        self.img1b=self.ax[0].plot(wls[self.i],fxc[self.i],alpha=0.5)
        self.img2 =self.ax[1].plot(wls[self.i],trans[self.i])
        self.ax[0].set_ylim(0,self.img_max)
        self.selected=[]#list of selected spectra. Is empty when we start.
        self.crosses=[]#List of checkboxes that have crosses in them. Starts empty, too.

    def set_spectrum(self,i):
        """This modifies the currently active spectrum to be plotted."""
        import numpy as np
        import tayph.functions as fun
        from tayph.vartests import typetest
        import matplotlib.pyplot as plt

        typetest(i,int,'i in molecfit_gui/set_order')
        self.wl = self.wls[i]
        self.spectrum = self.fxc[i]
        self.Tspectrum= self.trans[i]
        self.img_max = np.nanmean(self.spectrum[fun.selmax(self.spectrum,0.02,s=0.02)])*1.3

    def update_plots(self):
        """This redraws the plot planels, taking care to reselect axis ranges and such."""
        import matplotlib.pyplot as plt
        import numpy as np
        import tayph.functions as fun
        import copy
        self.img1a[0].set_xdata(self.wl)
        self.img1a[0].set_ydata(self.spectrum*self.Tspectrum)
        self.img1b[0].set_xdata(self.wl)
        self.img1b[0].set_ydata(self.spectrum)
        self.img2[0].set_xdata(self.wl)
        self.img2[0].set_ydata(self.Tspectrum)
        self.ax[0].set_title('Corrected / uncorrected s1d #%s/%s' % (self.i,self.N-1))
        self.ax[0].set_ylim(0,self.img_max)
        self.fig.canvas.draw_idle()

    def slide_spectrum(self,event):
        #This handles the spectrum-selection slide bar.
        self.i = int(self.spectrum_slider.val)
        self.set_spectrum(self.i)
        self.update_plots()

    def previous(self,event):
        #The button to go to the previous spectrum.
        self.i -= 1
        if self.i <0:#If order tries to become less than zero, loop to the highest spectrum.
            self.i = self.N-1
        self.set_spectrum(self.i)
        self.spectrum_slider.set_val(self.i)#Update the slider value.
        self.update_plots()#Redraw everything.

    def next(self,event):
        #The button to go to the next spectrum. Similar to previous().
        self.i += 1
        if self.i > self.N-1:
            self.i = 0
        self.set_spectrum(self.i)
        self.spectrum_slider.set_val(self.i)
        self.update_plots()

    def cancel(self,event):
        #This is a way to crash hard out of the python interpreter.
        import sys
        print('------Canceled by user')
        sys.exit()

    def save(self,event):
        #The "save" button is actually only closing the plot. Actual saving of things
        #happens after plt.show() below.
        import matplotlib.pyplot as plt
        print('------Closing GUI')
        plt.close(self.fig)


    def draw_crosses(self):
        import math
        for c in self.crosses:
            self.selec.lines.remove(c)
        self.crosses=[]#Delete the references and start again.
        self.fig.canvas.draw_idle()
        for s in self.selected:
            cx = s % (self.maxboxes)
            cy = self.nrows-math.floor(s/self.maxboxes)-0.5
            x = [float(cx)-0.5,float(cx)+0.5]
            y1 = [float(cy)-0.5,float(cy)+0.5]
            y2 = y1[::-1]
            self.crosses.append(self.selec.plot(x,y1,color='red')[0])
            self.crosses.append(self.selec.plot(x,y2,color='red')[0])
        self.fig.canvas.draw_idle()





def check_fit_gui(wls,fxc,trans):
    """This code initializes the GUI that plots the telluric-corrected spectra
    from Molecfit. The user may select spectra to be re-fit manually via the Molecfit GUI. Note that
    since molecfit takes between a few and 10 minutes to run on a single spectrum,
    this becomes arduous when more than a few spectra are selected in this way.
    It quickly becomes worthwile to redo the entire sequence with different inclusion
    regions overnight. The code returns the list of spectra that need to be done
    manually via the Molecfit GUI.

    Input: The list of wl axes, each of which was returned by a call to molecfit;
    and similarly the corrected spectra fxc and the transmission spectra.

    """


    import sys
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
    import tayph.functions as fun
    import numpy as np
    print('Starting visual inspection GUI')
    M = molecfit_gui(wls,fxc,trans)

    #The slider to cycle through orders:
    rax_slider = plt.axes([0.8, 0.2, 0.1, 0.02])
    rax_slider.set_title('Exposure #')
    M.spectrum_slider = Slider(rax_slider,'', 0,M.N-1,valinit=0,valstep=1)#Store the slider in the model class
    M.spectrum_slider.on_changed(M.slide_spectrum)

    #The Previous order button:
    rax_prev = plt.axes([0.8, 0.1, 0.04, 0.05])
    bprev = Button(rax_prev, ' <<< ')
    bprev.on_clicked(M.previous)

    #The Next order button:
    rax_next = plt.axes([0.86, 0.1, 0.04, 0.05])
    bnext = Button(rax_next, ' >>> ')
    bnext.on_clicked(M.next)

    #The save button:
    rax_save = plt.axes([0.92, 0.1, 0.07, 0.05])
    bsave = Button(rax_save, 'Continue')
    bsave.on_clicked(M.save)

    #The cancel button:
    rax_cancel = plt.axes([0.92, 0.025, 0.07, 0.05])
    bcancel = Button(rax_cancel, 'Cancel')
    bcancel.on_clicked(M.cancel)

    #This is to rescale the x-size of the checkboxes so that they are squares.
    bbox = M.fig.get_window_extent().transformed(M.fig.dpi_scale_trans.inverted())
    width, height = bbox.width*M.fig.dpi, bbox.height*M.fig.dpi


    M.selec=plt.axes([0.05,0.03,0.7,0.05*M.nrows])
    M.selec.spines['bottom'].set_color('white')
    M.selec.spines['top'].set_color('white')
    M.selec.spines['left'].set_color('white')
    M.selec.spines['right'].set_color('white')
    vlines = fun.findgen(M.N-1)+0.5

    row = M.nrows
    offset = 0
    for i in range(M.N):
        #print(i,float(i)-offset)

        if float(i)-offset > M.maxboxes-1.0:
            row -= 1
            offset += M.maxboxes
        M.selec.plot(float(i)-offset+np.array([-0.5,-0.5,0.5,0.5,-0.5]),[row,row-1,row-1,row,row],color='black')
        M.selec.text(float(i)-offset,row-0.5,'%s' % i,color='black',horizontalalignment='center',verticalalignment='center')

    M.selec.set_xlim(-0.55,M.maxboxes-1.0+0.55)#A little margin to make sure that the line thickness is included.
    M.selec.set_ylim(-0.05,1.0*M.nrows+0.05)
    M.selec.xaxis.set_tick_params(labelsize=8)
    M.selec.yaxis.set_tick_params(labelsize=8)



    def select_spectrum_box(event):

            #This handles with a mouseclick in either of the three plots while in add mode.
        if event.inaxes in [M.selec]:#Check that it occurs in one of the subplots.
            cc = event.xdata*1.0#xdata is the column that is selected.
            cr = event.ydata*1.0
            spectrum = np.round(cc)+np.round((M.nrows-cr-0.5))*M.maxboxes
            if spectrum < M.N:
                if spectrum in M.selected:
                    M.selected.remove(spectrum)
                    print('---Removed spectrum %s from manual' % spectrum)
                else:
                    M.selected.append(spectrum)
                    print('---Added spectrum %s to manual' % spectrum)
            M.draw_crosses()
    M.click_connector = M.fig.canvas.mpl_connect('button_press_event',select_spectrum_box)#This is the connector that registers clicks

    plt.show()
    print('Closed GUI, returning.')
    return(M.selected)







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
    print(f'------Reading telluric transmission from {inpath}')
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
