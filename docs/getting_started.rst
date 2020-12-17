.. _getting_started:

**********************
Demonstration tutorial
**********************

Getting started with tayph
##########################

Along with the core package of Tayph , a package with demo data is made available containing a part of the HARPS data
that was first used to find iron in the transmission spectrum of the UHJ KELT-9 b. This package also contains the necessary
configuration files and templates to obtain cross-correlations reminiscent of Hoeijmakers et al. (2018), but without the
application of telluric correction.

To get started, download Tayph, open a terminal in the root folder and run::

    python3 setup.py install.

This installs Tayph as a python package on your system.

Create an empty directory somewhere on your system. This will be the working directory that contains your dataset(s), models,
templates, and Tayph output. For the purpose of this walk-through, we will assume that folder is located at
:code:`'/Users/tayph/xcor_project/'`

Open a python interpreter and call::

    import tayph.run
    tayph.run.make_project_folder(p)

where you have set :code:`'p'` to a string describing the filepath of the empty directory you just created. This creates the necessary folder structure Tayph uses.

Download the dummy data, located here [URL]. You may also download any other pipeline-reduced HARPS dataset from the ESO archive.
A pipeline-reduced dataset will consist of a number of files for each exposure, i.e. e2ds_A/B, s1d_A/B, blaze_A/B files, etc.
Tayph does *not* work on raw echelle data. For the purpose of this walk-through, we will assume that this folder is located at
:code:`'/Users/tayph/downloads/HARPS_data/'`. Take care to download and process only *one* transit at a time. Do not put the observations
of multiple transits in the same download folder, because Tayph will treat them as a single time-series. Although certain
use-cases may exist where this is desirable, it is non-standard from the point of view of present literature.

Open a python interpreter in your working directory (i.e. :code:`'/Users/tayph/xcor_project/'`) and call::

    import tayph.read
    tayph.read.read_e2ds('/Users/tayph/downloads/HARPS_data/','KELT-9/night1','',nowave=True,molecfit=False,mode='HARPS',ignore_exp=[])

to convert the pipeline-reduced data to the format used by Tayph, and place it in the data folder in your working directory.
The input parameters are structured in the following way::

    tayph.read.read_e2ds('input_folder','output_folder','location_of_runfile',nowave=True,molecfit=False,mode='HARPS',ignore_exp=[])

- :code:`'input_folder'`: The first parameter is the location of your downloaded data. This is typically a dedicated folder in your project or even your downloads folder. 
- :code:`'output_folder'`: The second is the name of your dataset, as a folder name. Typically, this takes the form of system_name_b, or system_name_b/night_n if multiple transits of the same system are available.
- :code:`'location_of_runfile'`: The third is the location of your runfile (see below). This is only needed in case Molecfit is used (because the runfile points Tayph to where Molecfit is installed). This is left blank for now because we are not planning to use Molecfit at the moment.
- :code:`nowave=True`: The nowave keyword indicates whether a wavelength file is present in the downloaded data. This would be the case if you had run the HARPS pipeline yourself, but the wavelength file is typically not present for data downloaded from the archive. We therefore set this keyword to True, telling Tayph to take the wavelength solution from the FITS headers instead.
- :code:`molecfit=False`: Molecfit is set to False because we ignore it for the time being. For most metals in the optical, telluric correction is not crucially important, at least in first instance.
- :code:`mode='HARPS`:The mode keyword can be used to switch between HARPS, ESPRESSO, UVES-red and UVES-blue modes. In this case, we are dealing with HARPS data.
- :code:`ignore_exp=[]`: ignore_exp can be set to a list of integers, which allows you to ignore exposures in the time series. This can be useful if you find out, after reading in the data that some exposures are bad for whatever reason; and you want to ignore these from the analysis without going around deleting files. Of course you'd want to do this inspection before running Molecfit (if you were going to), because Molecfit will need to be run again after ignore_exp is changed, and typically takes a long time.

This has produced a new folder :code:`'/Users/tayph/xcor_project/data/KELT-9/night1/'`, in which the various files are located. This has
not created a configuration file, which we will typically need to make ourselves, but one is provided in the prepackaged template.

The configuration file
**********************

The configuration file is a 2-column tab-separated table with keywords in the first column and corresponding values in the second
column. The configuration file for this dataset may look like this::

    P              1.4811235
    a              0.03462
    aRstar         3.153
    Rp             1.891
    Mp             2.48
    K              0.275
    RpRstar        0.08228
    vsys           -18
    RA             20:31:26.4
    DEC            +39:56:20
    Tc             2457095.68572
    duration       235.0
    resolution     110000.0
    inclination    86.79
    vsini          111.0
    air            True

which describe the orbital period in days, the semi-major axis in AU, the mass/radius of the planet relative to Jupiter, the radial
velocity semi-amplitude of the star in km/s, the radius-ratio of the planet and star, the systemic velocity in km/s, the RA and DEC
coordinates, the transit center time, the spectral resolution of the instrument, the transit duration in minutes, the orbital
inclination in degrees (close to 90 if the planet is transiting), the projected equatorial rotation velocity of the stellar disk,
whether or not the wavelength solution is in air.

The library file
****************

After the data is reformatted and a configuration file is created, we need to point Tayph to a set of models that are going to be used as
cross-correlation templates and (optionally) for model injection-comparison. Models are located in the :code:`'/Users/tayph/xcor_project/models/'` directory,
with optional subdirectories for different sets of models. In most use-cases, the user will have multiple sets of models to choose from, which
may or may not be similar in their naming or content. To be able to access different sets of models, Tayph assumes that models are organised
in so-called libraries, which are ASCII tables that act as dictionaries with which the user can refer to model files saved in subfolders using short-hand names or labels.

The library file and template name/label are passed to Tayph at runtime, and the library files are structured as 2-column ASCII tables in the models/
directory. A typical library file called :code:`'KELT-9-models.dat'` may look as follows::

  FeI     KELT-9/4000K_Fe.fits
  FeII    KELT-9/4000K_Fe_p.fits
  TiI     KELT-9/4000K_Ti.fits
  TiII    KELT-9/4000K_Ti_p.fits
  TiO     KELT-9/3000K_TiO.fits
  H2O     KELT-9/3000K_H2O.fits

Individual models are assumed to be saved in FITS files, in subdirectories starting in the :code:`'/Users/tayph/xcor_project/models/'` directory.
In this example, the FITS files are located at in the :code:`'/Users/tayph/xcor_project/models/KELT-9'` directory. Each FITS file is a 2-row FITS image, with
wavelength (in nm) on the first row, and flux on the second row. In the case of transit spectra, this flux will typically be the expected transit radius of the 
planet as a function of wavelength. To convert models into cross-correlation templates, Tayph (optionally) performs a continuum subtraction (controlled by the
c_subtract switch below).

Examples of a model/template library file and associated model files are prepackaged along with the dummy data. Place these in the models subfolder of the working
directory.

A second library file located at :code:`'/Users/tayph/xcor_project/models/WASP-123-models.dat'` relevant to a different exoplanet system may take the following
form::

  FeI_2k      WASP-123/2000K_FeI.fits
  FeI_3k      WASP-123/3000K_FeI.fits
  FeII_3k     WASP-123/3000K_FeII.fits
  FeI_2k      WASP-123/2000K_TiI.fits
  FeI_3k      WASP-123/3000K_TiI.fits
  FeII_3k     WASP-123/3000K_TiII.fits
  TiO         WASP-123/2000K_TiO.fits
  H2O         WASP-123/2000K_H2O.fits

For each run of Tayph, only one model library or template library may be specified, so the user should organise their library files according to what models and
templates they wish to run in batches.

The run file 
************

Finally, we proceed by creating a run-file that specifies the working settings of our cross-correlation run. This file is again a 2-column ASCII table with
keywords in the first column and values in the second. This may look like below. The entries in the second column may be followed by commentary that
explains keywords or choices that are not self-descriptive or that you wish to remember.::

    molecfit_input_folder     /Users/username/Molecfit/share/molecfit/spectra/cross_cor/
    molecfit_prog_folder      /Users/username/Molecfit/bin/
    datapath                  data/KELT-9/night1  #The path to your test data.
    template_library          models/KELT-9-models   #The path to your library of models to be used as templates.
    model_library             models/KELT-9-models   #The path to your library of models to be used as injection models.
    model                     FeI                 ##A comma-separated list of templates as defined in your library file.
    template                  FeI,FeII,TiI,TiII   #A comma-separated list of templates as defined in your library file.
    c_subtract                True    #Set to True if your templates are not already continuum-subtracted.
    do_telluric_correction    True
    do_colour_correction      True
    do_xcor                   True    #Set this to True if you want the CCF to be recomputed. Set to False if you have already computed the CCF in a previous run, and now you just want to alter some plotting, cleaning or doppler shadow parameters. CCFs need to be rerun when masking, orbital parameters, velocity corrections, injected models or telluric corrections are altered.
    inject_model              False
    plot_xcor                 True
    make_mask                 False   #Don't be enthusiastic in making a mask. Once you change things like BERVs and airtovac corrections, the mask wont be valid anymore. Make 100% sure that these are correct first.
    apply_mask                True
    do_berv_correction        True
    do_keplerian_correction   True
    make_doppler_model        True     #Make a new doppler model (True) / use the previously generated one (False). If multiple templates are provided, the GUI to make a model will only be called on the first template. Make sure that is a template with strong stellar lines, i.e. FeI or FeII.
    skip_doppler_model        False    #This is skipping the application of the doppler model altogether.
    RVrange                   1000.0   #Extent of the CCF velocity excursion. Linearly increases computation time.
    drv                       2.0      #Cross-correlation step size in km/s.
    f_w                       60.0     #Cross-correlation filter width in km/s. Set to zero to disable hipass filter.
    shadowname                shadow_FeII     #This is the name of the file containing the doppler model shadow that is to be made or loaded. This file is located in the data folder, along with the spectral orders, telluric correction files, etc.
    maskname                  generic_mask    #Same, for the mask.



This file is typically saved in the working directory (i.e. as a file :code:`'/Users/tayph/xcor_project/testrun.dat'`), and is the primer for initialising
a cross-correlation run by calling::

    import tayph.run
    run.start_run('testrun.dat')


Use molecfit for telluric corrections
#####################################


So far we have not used molecfit in order to correct for telluric lines. We did avoid using molecfit by setting the molecfit-parameter to :code:`False`. 
If you want to use molecfit for telluric corrections, these are the necessary steps you have to take: 

- A parameter file for your instrument has to be created. An example of this parameter file is shown below. 
- The paths in your runfile have to be set correctly for molecfit to be executed. An example is shown below. 
- Exchange some files in the molecfit program folder.
- The read-call of tayph has to be executed with the right path indication for the runfile. An example is shown below. 


The parameter file
******************

For each instrument a parameter file has to be created. To work with the given example of KELT-9 b data, the parameter file can be downloaded here (add URL). 
For the purpose of this example we assume this file to be located in your molecfit-folder (i.e. :code:`'/Users/username/Molecfit/share/molecfit/spectra/cross_cor/'`. The following inputs have to be adapted to your system.

- :code:`user_workdir`: The user directory has to be set to the path of your project in our example case here we use :code:`user_workdir: /Users/tayph/xcor_project`. 
- :code:`filename`: The filename of the fits file that is created during the molecfit run has to be set. This file shall be named after your parameter file. Hence in this example: :code:`'filename: /Users/username/Molecfit/share/molecfit/spectra/cross_cor/HARPS.fits'`. 



The run file
************

Within the run file two paths have to be indicated. One of the paths is supposed to point at the folder where the parameter file is located. 
The other path indicates the position where the :code:`molecfit_gui` is located. These two paths are in the run file as given above. Here as a reminder::

    molecfit_input_folder     /Users/username/Molecfit/share/molecfit/spectra/cross_cor/
    molecfit_prog_folder      /Users/username/Molecfit/bin/

We are here assuming that molecfit is installed in the directory :code:`/Users/username/`. 


Exchange of molecfit files
**************************

In order to correct for an error in a code line of a molecfit python file, as well as making molecfit executable with python3, several changes have to be made. 
The necessary files including the file structure are given here (insert URL). 

Be aware that when replacing the file :code:`SM02GUI_Main.py` will lose its alias, which is the molecfit_gui in another folder. Make sure to create this alias again, name it molecfit_gui and replace the broken version in the bin folder (i.e. :code:`/Users/username/Molecfit/bin/`).


The read call
*************

In order to call tayph with molecfit, the following command has to be executed instead of the old read-command::

    tayph.read.read_e2ds('/Users/tayph/downloads/HARPS_data/','KELT-9/night1','/Users/tayph/xcor_project/testrun.dat',nowave=True,molecfit=True,mode='HARPS',ignore_exp=[])
    
Don't forget to set :code:`molecfit=True`.
