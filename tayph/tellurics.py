__all__ = [
    "execute_molecfit",
    "remove_output_molecfit",
    "retrieve_output_molecfit",
    "write_file_to_molecfit",
    "molecfit_gui",
    "write_telluric_transmission_to_file",
    "read_telluric_transmission_from_file",
    "apply_telluric_correction",
    "set_molecfit_config",
    "test_molecfit_config",
    "get_molecfit_config",
    "shift_exclusion_regions"
]

def shift_exclusion_regions(inpath,instrument,v):
    import astropy.constants as const
    import csv
    import tayph.util as ut
    from pathlib import Path
    import shutil
    import numpy as np
    """This reads the molecfit wavelength exclusion file of an instrument and shifts all boundaries
    by a certain number of km/s. This is to be used when the exclusion regions were designed to fall
    on stellar lines in one night of data; and the BERV has shifted these to another wavelength.
    This way you can use the same exclusion regions with changing BERV without having to click too
    much. Note that the exclusion file will be overwritten, but also backed up to a file in case
    of trouble."""

    inpath=ut.check_path(inpath,exists=True)
    outpath= Path(inpath)/f'wavelength_exclusion_{instrument}.dat'

    try:
        shutil.copy(outpath,Path(inpath)/f'wavelength_exclusion_{instrument}_velocity_shift_'
        'backup.dat')
    except:
        raise Exception('ERROR in copying wavelength exclusion file prior to velocity '
        'shift Aborting.')

    with open(outpath, 'r') as f_input:
        csv_input = csv.reader(f_input, delimiter=' ', skipinitialspace=True)
        x = []
        y = []
        for cols in csv_input:
            x.append(float(cols[0]))
            y.append(float(cols[1]))

    x=np.array(x)*(1+v/const.c.to('km/s').value)
    y=np.array(y)*(1+v/const.c.to('km/s').value)

    with open(outpath, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x,y))


def remove_output_molecfit(path,name):
    """This cleans the molecfit project folder"""
    import os
    from pathlib import Path
    root = Path(path)
    files = ['_out.fits','_out_gui.fits','_out_molecfit_out.txt','_out_fit.par','_out_fit.atm',
    '_out_apply_out.txt','_out_fit.atm.fits','_out_fit.res','_out_tac.asc','_TAC.fits',
    '_out_fit.res.fits','_out_fit.asc','_out_fit.fits']

    for f in files:
        try:
            os.remove(root/(name+f))
        except:
            pass
    return


def retrieve_output_molecfit(path):
    """
    This collects the actual transmission model created after a pass through molecfit is completed
    """
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

def execute_molecfit(molecfit_prog_root,molecfit_input_file,gui=False,alias='python3'):
    """This actually calls the molecfit command in bash"""
    import os
    from pathlib import Path
    import warnings
    if gui == False:
        command = str(Path(molecfit_prog_root)/'molecfit')+' '+str(molecfit_input_file)
        command2 = str(Path(molecfit_prog_root)/'calctrans')+' '+str(molecfit_input_file)
        os.system(command)
        os.system(command2)
    if gui == True:
        command = alias+' '+str(molecfit_prog_root/'molecfit_gui')+' '+str(molecfit_input_file)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.system(command)
    #python3 /Users/hoeijmakers/Molecfit/bin/molecfit_gui /Users/hoeijmakers/Molecfit/share/molecfit/spectra/cross_cor/test.par

def write_file_to_molecfit(molecfit_folder,name,headers,waves,spectra,ii,plot=False):
    """This is a wrapper for writing a spectrum from a list to molecfit format.
    name is the filename of the fits file that is the output.
    headers is the list of astropy header objects associated with the list of spectra
    in the spectra variable. ii is the number from that list that needs to be written (meaning
    that this routine is expected to be called as part of a loop).

    The user must make sure that the wavelength axes of these spectra are in air, in the observatory
    rest frame (meaning not BERV_corrected). Tayphs read_e2ds() function should have done this
    automatically.
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
    import matplotlib.pyplot as plt
    typetest(ii,int,'ii in write_file_to_molecfit()')
    molecfit_folder=ut.check_path(molecfit_folder,exists=True)
    wave = waves[int(ii)]
    spectrum = spectra[int(ii)]
    npx = len(spectrum)


    #Need to un-berv-correct the s1d spectra to go back to the frame of the Earths atmosphere.
    #This is no longer true as of Feb 17, because read_e2ds now uncorrects HARPS, ESPRESSO and
    #UVES spectra by default.
    # if mode == 'HARPS':
    #     berv = headers[ii]['HIERARCH ESO DRS BERV']
    # elif mode == 'HARPSN':
    #     berv = headers[ii]['HIERARCH TNG DRS BERV']
    # elif mode in ['ESPRESSO','UVES-red','UVES-blue']:
    #     berv = headers[ii]['HIERARCH ESO QC BERV']
    # wave = copy.deepcopy(wave*(1.0-(berv*u.km/u.s/const.c).decompose().value))
    spectrum[spectrum<=0]=np.nan
    err = np.sqrt(spectrum)
    # spectrum[np.isnan(spectrum)]=0
    # err[np.isnan(err)]=0
    if plot:
        plt.plot(wave,spectrum)
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.show()
        plt.plot(wave,err)
        plt.xlabel('Wavelength')
        plt.ylabel('Error')
        plt.show()
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
    thdulist.writeto(molecfit_folder/name,overwrite=True)
    ut.tprint(f'Spectrum {ii} written to {str(molecfit_folder/name)}')
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
    vlines = np.arange(0.5, M.N-0.5) # fun.findgen(M.N-1)+0.5
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
def write_telluric_transmission_to_file(wls,T,fxc,outpath):
    """This saves a list of wl arrays and a corresponding list of transmission-spectra
    to a pickle file, to be read by the function below."""
    import pickle
    import tayph.util as ut
    ut.check_path(outpath)
    print(f'------Saving teluric transmission to {outpath}')
    with open(outpath, 'wb') as f: pickle.dump((wls,T,fxc),f)

def read_telluric_transmission_from_file(inpath):
    import pickle
    import tayph.util as ut
    print(f'------Reading telluric transmission from {inpath}')
    ut.check_path(inpath,exists=True)
    pickle_in = open(inpath,"rb")
    return(pickle.load(pickle_in))#This is a tuple that can be unpacked into 2 elements.


def apply_telluric_correction(inpath,list_of_wls,list_of_orders,list_of_sigmas,parallel=False):
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
    from tayph.vartests import dimtest,postest,typetest,nantest,notnegativetest
    import copy
    if parallel: from joblib import Parallel, delayed

    T = read_telluric_transmission_from_file(inpath)
    wlT=T[0]
    fxT=T[1]

    typetest(list_of_wls,list,'list_of_wls in apply_telluric_correction()')
    typetest(list_of_orders,list,'list_of_orders in apply_telluric_correction()')
    typetest(list_of_sigmas,list,'list_of_sigmas in apply_telluric_correction()')
    typetest(wlT,list,'list of telluric wave-axes in apply_telluric_correction()')
    typetest(fxT,list,'list of telluric transmission spectra in apply_telluric_correction()')


    No = len(list_of_wls)#Number of orders.
    x = np.arange(No, dtype=float) #fun.findgen(No)
    Nexp = len(wlT)


    #Test dimensions
    if No != len(list_of_orders):
        raise Exception('Runtime error in telluric correction: List of wavelength axes and List '
        'of orders do not have the same length.')
    if Nexp != len(fxT):
        raise Exception('Runtime error in telluric correction: List of telluric wls and telluric '
        'spectra read from file do not have the same length.')
    if Nexp !=len(list_of_orders[0]):
        raise Exception(f'Runtime error in telluric correction: List of telluric spectra and data'
        f'spectra read from file do not have the same length ({Nexp} vs {len(list_of_orders[0])}).')

    def telluric_correction_order(i):
        order = list_of_orders[i]
        order_cor = order*0.0
        error  = list_of_sigmas[i]
        error_cor = error*0.0
        wl = copy.deepcopy(list_of_wls[i])#input wl axis, either 1D or 2D.
        #If it is 1D, we make it 2D by tiling it vertically:
        if wl.ndim == 1: wl=np.tile(wl,(Nexp,1))#Tile it into a 2D thing.
        #If read (2D) or tiled (1D) correctly, wl and order should have the same shape:
        dimtest(wl,np.shape(order),f'Wl axis of order {i}/{No} in apply_telluric_correction()')
        dimtest(error,np.shape(order),f'errors {i}/{No} in apply_telluric_correction()')
        for j in range(Nexp):
            T_i = interp.interp1d(wlT[j],fxT[j],fill_value="extrapolate")(wl[j])
            notnegativetest(T_i,f'T-spec of exposure {j} in apply_telluric_correction()')
            nantest(T_i,f'T-spec of exposure {j} in apply_telluric_correction()')
            T_i[T_i<0]=np.nan
            order_cor[j]=order[j]/T_i
            error_cor[j]=error[j]/T_i #I checked that this works because the SNR before and after
            #telluric correction is identical.

        return (order_cor, error_cor)

    # executing all No jobs simultaneously
    if parallel:
        list_of_orders_cor, list_of_sigmas_cor = zip(*Parallel(n_jobs=No)(delayed(
        telluric_correction_order)(i) for i in range(No)))
    else:
        list_of_orders_cor, list_of_sigmas_cor = zip(*[telluric_correction_order(i)
        for i in range(No)])
    return(list_of_orders_cor,list_of_sigmas_cor)


def set_molecfit_config(configpath):
    import pkg_resources
    from pathlib import Path
    import os
    import subprocess
    import tayph.system_parameters as sp
    import tayph.util as ut

    #Prepare for making formatted output.
    # terminal_height,terminal_width = subprocess.check_output(['stty', 'size']).split()

    Q1 = ('In what folder are parameter files defined and should (intermediate) molecfit output be '
    'written to?')
    Q2 = 'In what folder is the molecfit binary located?'
    Q3 = 'What is your python 3.x alias?'

    # configpath=get_molecfit_config()
    configpath=Path(configpath)
    if configpath.exists():
        ut.tprint(f'Molecfit configuration file already exists at {configpath}.')
        print('Overwriting existing values.')
        current_molecfit_input_folder = sp.paramget('molecfit_input_folder',configpath,full_path=True)
        current_molecfit_prog_folder = sp.paramget('molecfit_prog_folder',configpath,full_path=True)
        current_python_alias = sp.paramget('python_alias',configpath,full_path=True)

        ut.tprint(Q1)
        ut.tprint(f'Currently: {current_molecfit_input_folder} (leave empty to keep current '
        'value).')

        new_input_folder_input=str(input())
        if len(new_input_folder_input)==0:
            new_molecfit_input_folder = ut.check_path(current_molecfit_input_folder,exists=True)
        else:
            new_molecfit_input_folder = ut.check_path(new_input_folder_input,exists=True)
        print('')
        ut.tprint(Q2)
        ut.tprint(f'Currently: {current_molecfit_prog_folder}')
        new_prog_folder_input=str(input())
        if len(new_prog_folder_input)==0:
            new_molecfit_prog_folder = ut.check_path(current_molecfit_prog_folder,exists=True)
        else:
            new_molecfit_prog_folder = ut.check_path(new_prog_folder_input,exists=True)
        print('')
        ut.tprint(Q3)
        ut.tprint(f'Currently: {current_python_alias}')
        new_python_alias_input = str(input())
        if len(new_python_alias_input)==0:
            new_python_alias = current_python_alias
        else:
            new_python_alias = new_python_alias_input
    else:#This is actually the default mode of using this, because this function is generally
    #only called when tel.molecfit() is run for the first time and the config file doesn't exist yet.
        ut.tprint(Q1)
        new_molecfit_input_folder = ut.check_path(str(input()),exists=True)
        print('')
        ut.tprint(Q2)
        new_molecfit_prog_folder = ut.check_path(str(input()),exists=True)
        print('')
        ut.tprint(Q3)
        new_python_alias = str(input())


    with open(configpath, "w") as f:
        f.write(f'molecfit_input_folder   {str(new_molecfit_input_folder)}\n')
        f.write(f'molecfit_prog_folder   {str(new_molecfit_prog_folder)}\n')
        f.write(f'python_alias   {str(new_python_alias)}\n')

    ut.tprint(f'New molecfit configation file successfully written to {configpath}')


def test_molecfit_config(molecfit_config):
    """This tests the existence and integrity of the system-wide molecfit configuration folder."""
    import tayph.util as ut
    import tayph.system_parameters as sp
    from pathlib import Path
    import sys

    try:
        molecfit_input_folder = Path(sp.paramget('molecfit_input_folder',molecfit_config,full_path=True))
        molecfit_prog_folder = Path(sp.paramget('molecfit_prog_folder',molecfit_config,full_path=True))
        python_alias = sp.paramget('python_alias',molecfit_config,full_path=True)
    except:
        err_msg = (f'ERROR in initialising Molecfit. The molecfit configuration file '
        f'({molecfit_config}) exists, but it does not contain the right keywords. The required '
        'parameters are molecfit_input_folder, molecfit_prog_folder and python_alias. '
        'Alternatively, check that you have set the configfile parameter in run.molecfit() '
        'correctly.')
        ut.tprint(err_msg)
        sys.exit()

    if molecfit_input_folder.exists() == False:
        err_msg = (f"ERROR in initialising Molecfit. The molecfit configuration file "
            f"({molecfit_config}) exists and it has the correct parameter keywords "
            f"(molecfit_input_folder, molecfit_prog_folder and python_alias), but the "
            f"molecfit_input_folder path ({molecfit_input_folder}) does not exist. Please run "
            f"tayph.tellurics.set_molecfit_config() to resolve this. Alternatively, check that "
            f"you have set the configfile parameter in run.molecfit() correctly.")
        ut.tprint(err_msg)
        sys.exit()

    if molecfit_prog_folder.exists() == False:
        err_msg = (f"ERROR in initialising Molecfit. The molecfit configuration file "
            f"({molecfit_config}) exists and it has the correct parameter keywords "
            f"(molecfit_input_folder, molecfit_prog_folder and python_alias), but the "
            f"molecfit_prog_folder path ({molecfit_prog_folder}) does not exist. Please run "
            f"tayph.tellurics.set_molecfit_config() to resolve this. Alternatively, check that "
            f"you have set the configfile parameter in run.molecfit() correctly.")
        ut.tprint(err_msg)
        sys.exit()
    binarypath=molecfit_prog_folder/'molecfit'
    guipath=molecfit_prog_folder/'molecfit_gui'

    if binarypath.exists() == False:
        err_msg = (f"ERROR in initialising Molecfit. The molecfit configuration file "
                f"({molecfit_config}) exists and it has the correct parameter keywords "
                f"(molecfit_input_folder, molecfit_prog_folder and python_alias), but the molecfit "
                f"binary ({binarypath}) does not exist. Please run "
                f"tayph.tellurics.set_molecfit_config() to resolve this. Alternatively, check that "
                f"you have set the configfile parameter in run.molecfit() correctly.")
        ut.tprint(err_msg)
        sys.exit()

    if guipath.exists() == False:
        err_msg = (f"ERROR in initialising Molecfit. The molecfit configuration file "
                f"({molecfit_config}) exists and it has the correct parameter keywords "
                f"(molecfit_input_folder, molecfit_prog_folder and python_alias), but the molecfit "
                f"gui binary ({guipath}) does not exist. Please run "
                f"tayph.tellurics.set_molecfit_config() to resolve this. Alternatively, check that "
                f"you have set the configfile parameter in run.molecfit() correctly.")
        ut.tprint(err_msg)
        sys.exit()

    if ut.test_alias(python_alias) == False:
        err_msg = (f'ERROR in initialising Molecfit. The molecfit configuration file '
                f'({molecfit_config}) exists and it has the correct parameter keywords '
                f'(molecfit_input_folder, molecfit_prog_folder and python_alias), but the python '
                f'alias ({python_alias}) does not exist. Please run '
                f'tayph.tellurics.set_molecfit_config() to resolve this. Alternatively, check that '
                f"you have set the configfile parameter in run.molecfit() correctly.")
        ut.tprint(err_msg)
        sys.exit()



def get_molecfit_config():
    """
    This is the central place where the location of the molecfit configuration file is defined.
    """
    import pkg_resources
    from pathlib import Path
    configpath=Path(pkg_resources.resource_filename('tayph',str(Path('data')/Path(
        'molecfit_config.dat'))))
    return(configpath)
