__all__ = [
    "get_term_width",
    "test_alias",
    "statusbar",
    "start",
    "end",
    "check_path",
    "save_stack",
    "writefits",
    "readfits",
    "read_binary_kitzmann",
    "read_wave_from_e2ds_header"
]

def tprint(msg,margin=5):
    """This is a wrapper used to print long soft-wrapped messages to the terminal, such as
    error messages and warnings, or longer explanations. The marin is the margin at the right
    of the terminal window between the terminal edge and the wrapping limit. This gives a bit of
    air to the terminal output.
    """

    import textwrap
    try:
        terminal_width=get_term_width()
        print(textwrap.fill(msg,width=int(terminal_width)-margin))
    except:
        print(msg)


def get_term_width():
    import subprocess
    terminal_height,terminal_width = subprocess.check_output(['stty', 'size']).split()
    return(terminal_width)


def test_alias(alias):
    """This tests whether an alias exists on the system. Introduced for testing whether the
    user-supplied python alias exists, for calling Molecfit with."""
    import subprocess
    import platform
    import os
    if platform.system() == "Windows":
        cmd = "where"
    else:
        cmd = "which"

    try:
        location = str(subprocess.check_output([cmd,alias]),'utf-8')
    except:
        # print('The alias does not appear to exist.')
        return(False)
    return(True)

def statusbar(i,x):
    """
    This provides a little status indicator for use in long forloops.
    i is the counter (integer or float) of the forloop.
    x is either the total number of iterations (int/float) or
    the array through which is looped.

    Parameters
    ----------
    i : int, float
        Counter (integer or float) of the forloop.

    x : int, float or array-like
        Either the total number of iterations (int/float) or
        the array through which is looped.
    """
    if (type(x) == int or type(x) == float):
        print('  '+f"{i/(float(x)-1)*100:.1f} %", end="\r")
    elif len(x)>1:
        print('  '+f"{i/(len(x)-1)*100:.1f} %", end="\r")#Statusbar.



def start():
    """
    Short-hand for starting a timing measurement.
    """
    import time
    return(time.time())



def end(start,id='',silent=False,quiet=False):
    """
    Short-hand for ending a timing measurement and printing the elapsed time.

    Parameters
    ----------
    start : float
        Generated by time.time()

    id : str
        Description or numeral to identify the clock associated with the start time.

    Returns
    -------
    elapsed : float
        The time elapsed since start.
    """


    from tayph.vartests import typetest
    typetest(start,float,'start time in utils.end()')
    typetest(id,str,'id/descriptor in utils.end()')
    import time
    end=time.time()
    if silent+quiet == 0:
        print('Elapsed %s: %s' % ('on timer '+id,end-start))
    return end-start


def check_path(filepath,varname='filepath in check_path()',exists=False):
    """This is a short function that handles file paths when input to other functions.
    It checks that the proposed file path is either a string or a pathlib Path object, and
    converts to the latter if its a string. If the exists keyword is set to true, it will
    check that the path (either a file or a folder) exists, and raise an exception if it doesn't.
    All your filepath needs wrapped up in one :)

    This function tests the dimensions and shape of the input array var.
    Sizes is the number of elements on each axis.

    Parameters
    ----------
    filepath : str, Path object
        The path that needs to be vetted. This can be a folder or a filepath.

    varname : str
        Name or description of the variable to assist in debugging.

    exists : bool
        If set to True, the file/folder needs to exist in order to pass the test.
        If False, the routine only checks whether the variable provided is in fact
        a string or a path object.
    """


    import pathlib
    from tayph.vartests import typetest
    typetest(filepath,[str,pathlib.PosixPath,pathlib.WindowsPath],varname)#Test that we are dealing with a path.
    typetest(exists,bool)
    typetest(varname,str)
    if isinstance(filepath,str) == True:
        filepath=pathlib.Path(filepath)
    if exists == True and ((filepath.is_dir()+filepath.is_file()) == False):
        raise FileNotFoundError(str(filepath)+' does not exist.')
    else:
        return(filepath)

def save_stack(filename,list_of_2D_frames):
    """This code saves a stack of fits-files to a 3D cube, that you can play
    through in DS9. For diagnostic purposes.

    Parameters
    ----------
    filename : str, Path
        Output filename/path.

    list_of_2D-frames : list
        A list with 2D arrays

    Returns
    -------
    elapsed : float
        The time elapsed since start.

    """
    import astropy.io.fits as fits
    import numpy as np
    import pathlib
    from tayph.vartests import typetest
    from tayph.vartests import dimtest
    import warnings

    filename=check_path(filename,'filename in save_stack()')
    typetest(list_of_2D_frames,list,'list_of_2D_frames in save_stack()')#Test that its a list
    typetest(list_of_2D_frames[0],[list,np.ndarray],'list_of_2D_frames in save_stack()')
    for i,f in enumerate(list_of_2D_frames):
        typetest(f,[list,np.ndarray],'frame %s of list_of_2D_frames in save_stack()'%i)

    base = np.shape(list_of_2D_frames[0])
    N = len(list_of_2D_frames)

    dimtest(base,[2],'shape of list_of_2D_frames in save_stack()')#Test that its 2-dimensional
    for i,f in enumerate(list_of_2D_frames):
        dimtest(f,base,varname='frame %s of list_of_2D_frames in save_stack()'%i)#Test that all have the same shape.

    N = len(list_of_2D_frames)

    if N > 0:
        out = np.zeros((base[0],base[1],N))
        for i in range(N):
            out[:,:,i] = list_of_2D_frames[i]
        fits.writeto(filename,np.swapaxes(np.swapaxes(out,2,0),1,2),overwrite=True)
    else:
        warnings.warn("List_of_2D_frames has length zero. No output was generated by save_stack().", RuntimeWarning)


def writefits(filename,array):
    """
    This is a fast wrapper for fits.writeto, with overwrite enabled.
    """
    import astropy.io.fits as fits
    from tayph.vartests import typetest
    from tayph.vartests import dimtest
    import pathlib
    import numpy as np
    filename=check_path(filename,'filename in writefits()')
    # base = np.shape(array)
    # dimtest(base,[2],'shape of array in writefits()')#Test that its 2-dimensional
    fits.writeto(filename,array,overwrite=True)


def readfits(filename,i=0):
    """
    This is a replacement for fits.getdata, closing the file after reading to prevent too many open
    file errors. Set i to an extension if you wish to return another extension than the first one.
    """
    import astropy.io.fits as fits
    from tayph.vartests import typetest
    from tayph.vartests import dimtest
    import pathlib
    import numpy as np
    filename=pathlib.Path(filename,exists=True)
    with fits.open(filename) as hdul:
        D=hdul[i].data
    del hdul
    return(D)


def read_binary_kitzmann(inpath,double=True):
    """This reads a binary model spectrum (those created by Daniel Kitzmann)
    located at path inpath."""

    import struct
    from tayph.vartests import typetest
    if double == True:
        nbytes = 8
        tag = 'd'
    else:
        nbytes = 4
        tag = 'f'

    check_path(inpath,exists=True)

    r = []

    f = open(inpath,'rb')
    while True:
        seq = f.read(nbytes)
        if not seq:
            break
        else:
            r.append(struct.unpack(tag,seq)[0])#I put an index here because it was making a list of tuples.
            #I hope that this still works when double=True!
    f.close()
    return(r)


def read_wave_from_e2ds_header(h,mode='HARPS'):
    """
    This reads the wavelength solution from the HARPS header keywords that
    encode the coefficients as a 4-th order polynomial.
    """
    import numpy as np
    import tayph.functions as fun

    if mode not in ['HARPS', 'HARPSN', 'HARPS-N', 'UVES', 'FIES']:
        raise ValueError("in read_wave+from_e2ds_header: mode needs to be set to HARPS, HARPSN or UVES.")

    if mode in ['HARPS', 'HARPSN', 'HARPS-N', 'UVES']:

        npx = h['NAXIS1']
        no = h['NAXIS2']
        x = np.arange(npx, dtype=float)  # fun.findgen(npx)
        wave = np.zeros((npx, no))

        if mode == 'HARPS':
            coeffkeyword = 'ESO'
        if mode in ['HARPSN', 'HARPS-N']:
            coeffkeyword = 'TNG'
        if mode == 'UVES':
            delt = h['CDELT1']
            for i in range(no):
                keystart = h[f'WSTART{i + 1}']
                # keyend = h[f'WEND{i+1}']
                # wave[:,i] = fun.findgen(npx)*(keyend-keystart)/(npx-1)+keystart
                wave[:, i] = np.arange(npx, dtype=float) * delt + keystart  # fun.findgen(npx)*delt+keystart
                # These FITS headers have a start and end, but (end-start)/npx does not equal
                # the stepsize provided in CDELT (by far). Turns out that keystart+n*CDELT is the correct
                # representation of the wavelength. I do not know why WEND is provided at all and how
                # it got to be so wrong...

        else:
            key_counter = 0
            for i in range(no):
                l = x * 0.0
                for j in range(4):
                    l += h[coeffkeyword + ' DRS CAL TH COEFF LL%s' % key_counter] * x ** j
                    key_counter += 1
                wave[:, i] = l
        wave = wave.T

    if mode == "FIES":
        import re

        # Following code block loops through all of the fits headers and extracts the headers relavent to the wavelength axis
        compiled_wave_length_info = ""
        for i in range(1, len(h)):
            if i < 10:
                header_label = "WAT2_00" + str(i)
            elif i < 100:
                header_label = "WAT2_0" + str(i)
            else:
                # The WAT2_### header can vary in length dependening on observation. Code checks where an error will occur
                # befor executing and will end the loop if one does
                try:
                    header_label = "WAT2_" + str(i)
                    error_to_check = h[header_label]  # Command which would create if the error
                except KeyError:
                    break
            # Since headers a delimited by a " " those that end with are white space are ignored by python. This add a dummy
            # character so the length of the string of each row remains maintained
            if len(h[header_label]) == 67:
                h[header_label] = h[header_label] + "-"

            # All headers are combined into a single list and then split into a list using the character "
            wave_length_info = h[header_label]
            compiled_wave_length_info = compiled_wave_length_info + wave_length_info

        compiled_wave_length_info = compiled_wave_length_info.split("\"")

        # There are a few elements in the list which don't contain wavelength infomration following block removes this
        rows = []
        for i in range(0, len(compiled_wave_length_info)):
            if "s" in list(compiled_wave_length_info[i]):
                continue
            elif len(compiled_wave_length_info[i]) == 0:
                continue
            else:
                rows.append(compiled_wave_length_info[i])

        # Final block extracts the relevent wavelenght information and generates a grid for each eschelle order
        wave = []
        for i in range(0, len(rows)):
            string = rows[i]
            wave_length_info = re.split(' |-', string)
            wave_length_start = float(wave_length_info[3])
            order_steps = float(wave_length_info[4])
            pixel_number = int(wave_length_info[5])
            wave_length_grid = np.linspace(wave_length_start,
                                           wave_length_start + pixel_number * order_steps,
                                           pixel_number)

            wave.append(wave_length_grid)

    return np.array(wave)



def read_wave_from_makee_header(h):
    import numpy as np
    npx = h['NAXIS1']
    no = h['NAXIS2']
    x = np.arange(npx, dtype=float) #fun.findgen(npx)
    wave=np.zeros((no,npx))

    for i in range(no):
        l = x*0.0
        #Coefficient in non-linear MIKEE mode come in 2 lines per order. I read these, attach
        #them together, split them on the spaces in-between coefficients, and convert to floats
        #to evaluate the polynomial.
        coeff_line = str(h['WV_0_'+str(i+1).zfill(2)])+str(h['WV_4_'+str(i+1).zfill(2)])
        #I can add them together without introducing a space in between because each line starts
        #with a space. I will remove that space on the next line before splitting:
        coeffs = np.flip(np.array(coeff_line.split(),dtype=np.float32)) #in descending
        #order.

        wave[i] = np.poly1d(coeffs)(x)
    return(wave)
