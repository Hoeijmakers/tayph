.. _getting_started:

**************************
Getting started with Tayph
**************************

This page provides a tutorial for how to get started with Tayph from scratch, covering installation
and the application of Tayph's standard workflow to a set of demo data. When finishing this tutorial,
you should have enough information to use Tayph for the analysis of standard high-resolution datasets.

Requirements
############

Tayph is written in python 3, and requires multiple packages to function. All of these can be installed
using standard package managers (e.g. pip or Anaconda).

Python version 3.x
Matplotlib
Numpy
Scipy
Astropy
lmfit
joblib

Installation
############


To get started, download or clone Tayph from the `GitHub page <https://github.com/Hoeijmakers/tayph>`_,
open a terminal in the root folder and run::

    python3 setup.py install.

This installs Tayph as a python package on your system. Verify that Tayph can be imported by opening a
python interpreter and importing Tayph::

  import tayph

If no errors are raised, Tayph has been successfully installed on your system.
For the scope of this tutorial we will assume, Tayph is installed in the following path :code:`'/usr/local/src/tayph/tayph'`.

.. note::
  You will not have to remember this path for the scope of running through the tutorial without molecfit. 
  Molecfit will require some path information.


Setting up Tayph with demo data
###############################


Tayph works within a folder structure in which your data, models, templates and Tayph output results are
organised. We call this the 'project folder' or 'working directory'. Create an empty directory somewhere on your system.
For the purpose of this walk-through, we will assume that this folder is called
:code:`'/Users/tayph/xcor_project/'`. Open a python 3 interpreter and call::

    import tayph.run
    p = '/Users/tayph/xcor_project/'
    tayph.run.make_project_folder(p)
    exit()

where you have set :code:`'p'` to a string describing the filepath of the empty directory you just created.
This creates the necessary folder structure Tayph uses. 

Along with the core package of Tayph, a package with demo data is made available containing the
HARPS-N spectra that were first used to find iron in the transmission spectrum of the exoplanet
KELT-9 b. This package also contains the necessary configuration files and templates to obtain
cross-correlations reminiscent of Hoeijmakers et al. (2018), but without the application of
telluric correction (for those see section below).

Download the dummy data, located `here <https://drive.google.com/file/d/1OMHXvCJ626oecP1j_0BYvHRQA_MCE0ec/view?usp=sharing>`_ .
You may also download any other pipeline-reduced HARPS
or HARPS-N dataset from the ESO or TNG archives (Tayph does *not* work on
raw echelle data). A pipeline-reduced dataset will consist of a number of files for each exposure,
i.e. e2ds_A/B, s1d_A/B, blaze_A/B files, etc. For the purpose of this walk-through, we will assume
that this folder is located at :code:`'/Users/tayph/downloads/demo_data/'`. If downloading your own
data, take care to download and process only *one* transit at a time. Do not put the observations
of multiple transits in the same download folder, because Tayph will treat them as a single
time-series.  Although certain use-cases may exist where this is desirable, it is non-standard from
the points of view of present literature and of this demonstration.

To continue with the demo data, move to your working directory in the terminal
(i.e. :code:`'cd /Users/tayph/xcor_project/'`), open a python 3 interpreter and call::

    import tayph.run as run
    run.read_e2ds('/Users/tayph/downloads/demo_data/kelt-9-spectra','KELT-9/night1',mode='HARPSN',config=True)

This converts the pipeline-reduced data to the format used by Tayph, places it in the data
folder in your working directory, and executes a preliminary cross-correlation to measure the
radial velocities of the stitched, wide-band (1D) spectra and the indivual echelle orders (2D).
This preliminary cross-correlation is quite time consuming, but it helps to judge whether the
data is read properly and how to set Tayph to correctly deal with velocity corrections and the
wavelength solution.

Read_e2ds is meant to provide you with a quick gateway to handling pipeline reduced echelle spectra,
and is designed to work for multiple different spectrographs out of the box; explicitly to try to
protect you against confusion regarding wavelength solutions and velocity corrections
(where different pipelines have different conventions). The input parameters of read_e2ds are
structured in the following way::

    tayph.run.read_e2ds('input_folder','output_name',mode='HARPSN',measure_RV=True,star='hot',config=True)

- :code:`'input_folder'`: The first parameter is the location of your downloaded data. This is typically a dedicated folder in your project or even your downloads folder.
- :code:`'output_name'`: The second is the name of your dataset, as a folder name. Typically, this takes the form of system_name_b, or system_name_b/night_n if multiple transits of the same system are available.
- :code:`mode='HARPS'`:The mode keyword can be used to switch between HARPS, HARPSN (or HAPRS-N), ESPRESSO, UVES-red,UVES-blue, CARMENES-VIS and CARMENES-NIR modes. In this case, we are dealing with HARPS-N data.
- :code:`measure_RV=True`: Set to True if, after reading in the data, let Tayph perform quick cleaning and correlation with a PHOENIX model and an Earth telluric model. 
At the end, Tayph will plot the 1-dimensional and 2-dimensional spectra as well as the two models, to give you a good sense of whether one or the other are barycentric corrected or not, 
and whether wavelength solutions are in air or vaccuum. These will influence how you configure Tayph later so it is recommended to run read_e2ds with measure_RV=True when starting out.
- :code:`star='solar'`: If measure_RV is set to True, the PHOENIX model used will either match that of the sun (code:`star='solar'`), that of a 9000K A-star (code:`star='hot'`) or a cool 4000K K-dwarf (code:`star='cool'`).
- :code:`config=True`: If set, Tayph will create an empty configuration file with some values filled in, depending on the instrument mode.


Read_e2ds has produced a new folder :code:`'/Users/tayph/xcor_project/data/KELT-9/night1/'` in
which the various files are located, including a dummy configuration file called
:code:`'config_empty'`. The user would now need to proceed by filling in this configuration
and renaming it from :code:`config_empty`: to :code:`config`:. However, a finished configuration
file has been provided along with the prepackaged demo data (in
:code:`'/Users/tayph/downloads/demo_data/configuration_files/config'`), so for the purpose of this
tutorial, you should proceed by copying this file to the data folder instead.



The configuration file
**********************

The configuration file is a 2-column tab-separated table with keywords in the first column and
corresponding values in the second column. The configuration file for this dataset may look like
this, describing HARPS-N observations of KELT-9 b::


      P	          1.4811235
      a	          0.03462
      aRstar	    3.153
      Mp	        2.48
      Rp          1.891
      K           0.275
      RpRstar     0.08228
      vsys        -18
      RA          20:31:26.4
      DEC         +39:56:20
      Tc          2457095.68572
      resolution  110000.0
      inclination	86.79
      vsini	      111.0
      long	      -17.8850
      lat         28.7573
      elev        2396.0
      air         True

which describe the orbital period in days, the semi-major axis in AU, the mass/radius of the planet
relative to Jupiter, the radial velocity semi-amplitude of the star in km/s, the radius-ratio of
the planet and star, the systemic velocity in km/s, the RA and DEC coordinates, the transit centre
time, the spectral resolution of the instrument, the orbital inclination in degrees (close to 90 if
the planet is transiting), the projected equatorial rotation velocity of the stellar disc, the
geographical location of the observatory and whether or not the wavelength solution is in air.
When running supported instruments, instrument-specific information will have been filled in
automatically.

.. note::
  When setting the configuration file, the transit duration is derived from the combination of
  transit parameters (a/Rstar, period and the inclination). This duration is used to inject models
  into the data, but also to select which spectra are to be co-added in the rest-frame of the
  planet. The accuracy of these parameters therefore has an effect on how the spectra are treated.



Model and template library files
********************************

After the data is reformatted and a configuration file is created, we need to point Tayph to a set
of model spectra that are going to be used as cross-correlation templates and (optionally) for model
injection-comparison. Models may be located in the :code:`'/Users/tayph/xcor_project/models/'`
directory, with optional subdirectories for different sets of models. In most use-cases, the user
will have multiple sets of models to choose from, which may or may not be similar in their naming
or content. To be able to access different sets of similar models, Tayph assumes that models are
organised in so-called libraries, which are ASCII tables that act as dictionaries through which the
user can refer to model files saved in subfolders using short-hand names (i.e. labels).

The library files are structured as 2-column ASCII tables in the models/ directory. A library file
called :code:`'kelt-9-model-library.dat'` is provided along with the demo data, and is as follows::

    FeI_4k     KELT-9/4000K_1_Fe.fits
    FeII_4k    KELT-9/4000K_1_Fe_p.fits
    MgI_4k     KELT-9/4000K_1_Mg.fits
    NaI_4k     KELT-9/4000K_1_Na.fits
    ScII_4k    KELT-9/4000K_1_Sc_p.fits
    CrII_4k    KELT-9/4000K_1_Cr_p.fits
    TiII_4k    KELT-9/4000K_1_Ti_p.fits
    YII_4k     KELT-9/4000K_1_Y_p.fits

Individual models are to be saved as FITS files, which are assumed to be located in subdirectories
starting in the :code:`'/Users/tayph/xcor_project/models/'` directory. Absolute paths (e.g.
:code:`'/Users/tayph/xcor_project/models/KELT-9/4000K_1_Fe.fits'`) may also be provided.
Each FITS file is a 2-row FITS image, with wavelength (in nm) on the first row, and flux on the
second row. In the case of transit spectra, this flux will typically be the expected transit radius
of the planet as a function of wavelength. To convert models into cross-correlation templates,
Tayph (optionally) performs a continuum subtraction (controlled by the c_subtract switch below).

.. note::
  For Tayph to correctly work, the template needs to have a continuum of zero, either a priori or
  after application of the continuum subtraction option included in Tayph. In addition, absorption
  lines need to be in the negative direction. Otherwise, built-in routines that deal with the cross-
  correlation functions may mis-interpret the results.

In this example, the FITS files of the cross-correlation templates are to be located in the
:code:`'/Users/tayph/xcor_project/models/KELT-9'` directory, and an example of a library file and
associated model files are prepackaged along with the dummy data. Create a subfolder
:code:`'KELT-9'` in the :code:`'/Users/tayph/xcor_project/models/'` directory, place the
template FITS files from the demo package inside (located in
:code:`'/Users/tayph/downloads/demo_data/templates'`), and finally place the pre-packaged library
file (:code:`'/Users/tayph/downloads/demo_data/configuration_files/KELT-9-model-library'`) in the
the :code:`'/Users/tayph/xcor_project/models/'` directory. The library file and template name/label
are going to be passed to Tayph at runtime, allowing Tayph to find the model template files.

Later, when the user wishes to analyse a dataset of a different planet, a second library file
located at :code:`'/Users/tayph/xcor_project/models/WASP-123456-models'` may be placed in the
:code:`'models/'` directory as well, pointing to different (but perhaps similar) models, e.g. as
follows::

  FeI_2k      WASP-123456/2000K_FeI.fits
  FeI_3k      WASP-123456/3000K_FeI.fits
  FeII_3k     WASP-123456/3000K_FeII.fits
  FeI_2k      WASP-123456/2000K_TiI.fits
  FeI_3k      WASP-123456/3000K_TiI.fits
  FeII_3k     WASP-123456/3000K_TiII.fits
  TiO         WASP-123456/2000K_TiO.fits
  H2O         WASP-123456/2000K_H2O.fits

For each run of Tayph, only one model library or template library may be specified, so the user
should organise their library files according to what models and templates they wish to run in
batches.



The run file
************

The final step is to create a run-file that controls the working parameters of our
cross-correlation run. This file is again a 2-column ASCII table with keywords in the first column
and values in the second. This may look like below. The entries in the second column may be
followed by commentary that explains keywords or choices that are not self-descriptive or that you
wish to remember for yourself.::

    datapath                  data/KELT-9/night1  #The path to your test data.
    template_library          models/KELT-9-model-library.dat   #The path to your library of models to be used as templates.
    model_library             models/KELT-9-model-library.dat   #The path to your library of models to be used as injection models.
    model                     FeI_4k                 #A comma-separated list of templates as defined in your library file.
    template                  FeII_4k,FeI_4k  #A comma-separated list of templates as defined in your library file.
    c_subtract                True    #Set to True if your templates are not already continuum-subtracted. True for demo data.
    do_telluric_correction    False   #Molecfit has not been run for the demo data.
    do_colour_correction      True
    do_xcor                   True    #Set this to True if you want the CCF to be recomputed. Set to False if you have already computed the CCF in a previous run, and now you just want to alter some plotting, cleaning or doppler shadow parameters. CCFs need to be rerun when masking, orbital parameters, velocity corrections, injected models or telluric corrections are altered.
    inject_model              False
    plot_xcor                 True
    make_mask                 False   #Don't be enthusiastic in making a mask. Once you change things like BERVs and airtovac corrections, the mask wont be valid anymore. Make 100% sure that these are correct first.
    apply_mask                False
    do_berv_correction        True
    do_keplerian_correction   True
    make_doppler_model        False   #Make a new doppler model (True) / use the previously generated one (False). If multiple templates are provided, the GUI to make a model will only be called on the first template. Make sure that is a template with strong stellar lines, i.e. FeI or FeII.
    skip_doppler_model        True    #This is skipping the application of the doppler model altogether.
    RVrange                   300.0   #Extent of the CCF velocity excursion. Linearly increases computation time.
    drv                       1.0     #Cross-correlation step size in km/s.
    f_w                       0.0     #Cross-correlation filter width in km/s. Set to zero to disable hipass filter.
    shadowname                shadow_FeII     #This is the name of the file containing the doppler model shadow that is to be made or loaded. This file is located in the data folder, along with the spectral orders, telluric correction files, etc.
    maskname                  generic_mask    #Same, for the mask.



This file is typically saved in the working directory, although it can be placed anywhere in your
system (make sure to adjust the paths correspondingly!). The demo package contains a pre-made run file for the KELT-9 dummy data, located at
:code:`'/Users/tayph/downloads/demo_data/configuration_files/demorun.dat'`). Place it into your
working directory, and from the working directory, initialise a cross-correlation run by calling::

    import tayph.run
    tayph.run.start_run('demorun.dat')

This initialises the processing cascade of Tayph. Cross-correlation output is saved in the
output directory :code:`'/Users/tayph/xcor_project/output/KELT-9/night1/'`, with a subfolder
for each template library (a dataset can be cross-correlated with templates of different
libraries), in which there are subfolders for each template. The CCF data is stored in separate
FITS files, with the main output being :code:`'ccf_cleaned.fits'`. If this file was successfully
generated, you should see a slanted dark streak in the CCFs of Fe I and Fe II, which is the
signature of these atoms in the atmosphere of KELT-9 b.

Congratulations! You have now successfully installed and executed Tayph!


Interactive processing
**********************

The functionality of Tayph includes two GUI interfaces. The first allows users to interactively
specify bad regions in their spectral orders. This is activated by setting the make_mask and
apply_mask. parameters in the run file to True. After cross-correlation, a second GUI can be
opened to allow the user to fit the Doppler shadow feature with a single or double-gaussian model.
This is activated by setting make_doppler_model to True and skip_doppler_model to False.
After having been run once, the mask files and doppler model files are saved in the data folder
with names as specified by the shadowname and maskname parameters in the run file.


Using molecfit for telluric corrections
#######################################

So far we have not used molecfit in order to correct for telluric lines.
If you want to use molecfit for telluric corrections, these are the necessary steps you have to take:

- You need to install the standalone version of Molecfit on your system.
- Replace some files within molecfit to make it exectutable.
- A parameter file for your instrument has to be created. Parameter files for the supported instruments packaged in the demo data package, but you need to modify these to work on your system (see below).
- You need to use Tayph create a configuration file for molecfit.


Install molecfit on your system
*******************************

You can find the all the required Molecfit files `here <https://drive.google.com/file/d/1GU--4UFYxmWPW1zOHzFT9bnzAhZGUR95/view?usp=sharing>`_ .
It includes a manual on how to install molecfit on your system. 
Additional notes for installation on Mac (Catalina) can be found here (add link)

For the rest of theses tutorial, we assume your molecfit installation to be located at i.e. :code:`'/usr/local/src/Molecfit'`.


Exchange of molecfit files (this is not done yet)
*************************************************

In order to correct for an error in a code line of a molecfit python file, as well as making molecfit executable with python3, several changes have to be made.
The necessary files including the file structure are given here (insert url)

- bin changes 
.. note::
    When replacing the file :code:`SM02GUI_Main.py`, it will lose its alias, which is the molecfit_gui in another folder. Make sure to create this alias again, name it molecfit_gui and replace the broken version in the bin folder (i.e. :code:`/usr/local/src/Molecfit/bin/`).


The parameter files
*******************

For each instrument a parameter file has to be created. To work with the given example of KELT-9 b data, the parameter file can be found in your downloaded dummy data.
For the purpose of this example we assume this file to be located here (:code:`'/Users/tayph/xcor_project/models/molecfit/'`. 
The following inputs have to be adapted to your system.

- :code:`user_workdir`: The user directory has to be set to the path of your project. This is necessary for molecfit to find your files. We use :code:`user_workdir:user_workdir: /Users/tayph/xcor_project/`.
- :code:`filename`: The filename of the fits file that is created during the molecfit run has to be set. This file shall be named after your parameter file. Hence in this example: :code:`'filename: user_workdir: /Users/tayph/xcor_project/models/molecfit/HARPSN.fits'`.
- :code:`output_dir`: The output directory for intermediate molecfit output has to be defined. We define it to be the same folder as the input directory where store out parameter file. We use  :code:`'output_dir = user_workdir: /Users/tayph/xcor_project/models/molecfit/'`


The molecfit config file
************************

For molecfit to successfully run through, a config file has to be adapted. 
Tayph produces a config file per default (see :code:`'tayph/tayph/data/molecfit_config.dat'`), but requires you to set the parameters yourself.

To set the parameters, navigate to your project folder, open a python 3 interpreter and call::

    import tayph.tellurics as tellurics
    tellurics.set_molecfit_config('/usr/local/src/tayph/tayph/data/molecfit_config.dat')

You will be asked to enter the following information:

    - In what folder are parameter files defined and should (intermediate) molecfit output be written to?

    This is going to be the location of your parameter file, i.e. :code:`'/Users/tayph/xcor_project/models/molecfit/'`.

    - In what folder is the molecfit binary located?

    This is within your molecfit installation, i.e. :code:`'/usr/local/src/Molecfit/bin'`
    
    - What is your python 3.x alias?

    python



The run call
*************

Now we are almost there. Now you only need to execute molecfit from the terminal before running the cross-correlation. 
To do so, you navigate into your project folder, open a python3 interpreter and call::

    import tayph.run as run
    run.molecfit('/Users/tayph/xcor_project/data/KELT-9/night1', mode='GUI',instrument='HARPSN')

This will open the molecfit GUI for you to choose your fitting regions, continuum normalisation, etc and save the files in the output directory we indicated in the parameter file. 
Now we want to apply this correction to all obtained spectra, subsequently calling::

    run.molecfit('/Users/tayph/xcor_project/data/KELT-9/night1', mode='batch',instrument='HARPSN')

If you want to execute the GUI and apply the correction immediately, you can call::

    import tayph.run as run
    run.molecfit('/Users/tayph/xcor_project/data/KELT-9/night1', mode='GUI',instrument='HARPSN')


.. note::
    The GUI requires screen access, so remember to add -X when logging into an external server. The batch process runs through without interaction. 
    So if you want to run tayph on a server, it is recommended to call those two tasks separately and execute the batch process on a screen, for example over night.



