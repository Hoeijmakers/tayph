High Resolution Spectroscopy of Exoplanet Atmospheres
-----------------------------------------------------
THIS IS A WORK IN PROGRESS AND ONLY SEGMENTS OF THE PROJECT ARE CURRENTLY FUNCTIONAL
------------------------------------------------------------------------------------

[![Astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org)
[![ReadTheDocs](https://readthedocs.org/projects/tayph/badge/?version=latest)](https://tayph.readthedocs.io/en/latest/?badge=latest)
[![Travis](https://travis-ci.org/Hoeijmakers/tayph.svg?branch=master)](https://travis-ci.org/Hoeijmakers/tayph)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
![ViewCount](https://views.whatilearened.today/views/github/Hoeijmakers/tayph.svg)


Tayph is envisioned as a package to help you analyze high-resolution spectral time-series of exoplanets using the Cross-correlation technique introduced by [Snellen et al. 2010](https://www.nature.com/articles/nature09111) in a *robust*, *transparent*, *flexible* and *user-friendly* way.

- **Robust**: The analysis of high-resolution spectral time series can be challenging, and normally involves a number of analysis steps that may depend strongly on the type of data, planet, and circumstances. The steps applied in the analysis cascade have been developed and tested over the course of a number of published analyses. The code itself is thoroughly tested at the build and execution levels.
- **Transparent**: Tayph aims to be transparent in its capabilities, assumptions and limitations. To this end, the entire package is open-source, and the analysis steps have been well described in published literature. The code is documented following the Astropy style, and thoroughly commented in-line. Ongoing issues, bugs and improvements are maintained and documented in public on [Tayphs GitHub page](https://github.com/Hoeijmakers/tayph).
- **Flexible**: Tayph is designed to work on spectral time series observed with commonly used High-resolution Echelle spectrographs that observe optical wavelengths. When applied to pipeline-reduced data of HARPS and ESPRESSO, Tayph will run out of the box with minimal effort required. For other instruments, Tayph requires the user to format their spectral time-series into an accessible, human-readable and easy-to-understand format. The routines and analysis steps employed by Tayph can be adjusted easily to allow for the analysis of other wavelength ranges (e.g. the NIR/MIR). The modular nature provided by the python package ecosystem and the transparency of the data structure will allow the user to insert their own modules whenever needed. 
- **User-friendly**: Tayph provides meaningful status reports and errors. In addition, because high-resolution spectrographs can produce large, unwieldly spectra; considerable effort has been directed to providing meaningful plots and GUIs for the user to inspect, understand and interact with their data.

Despite the aims set out above, both the methods used by the community to analyze high-resolution observations of exoplanet atmospheres as well as Tayph are likely never perfect, and subject to approximations, mistakes or bugs. Secondly, the code being focused on transparency and flexibility, Tayph is not designed to make *optimal* use of computer resources. The entire package is written in Python, and there are many instances where factors can be gained in terms of computation and memory efficiency. Making Tayph more nimble is hence an ongoing effort. I hope that you, *the user* will benefit from this package, and share your questions, feedback, ideas or pull requests. I will strive to use your input to better Tayph.




License
-------

This project is Copyright (c) Jens Hoeijmakers and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the Astropy package template <https://github.com/astropy/package-template>
which is licensed under the BSD 3-clause license. See the licenses folder for
more information.

