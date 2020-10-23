.. _introduction:

***************
Introduction
***************

Cross-correlation
=================


The cross-correlation technique leverages the combination of the following four facts to detect
species in the atmospheres of short-period exoplanets:
* The radial velocity of a short-period exoplanet can change by tens of km/s in a timespans of hours.
* Ground-based high-resolution spectrographs can resolve individual spectral lines with an instantaneous
resolution of a few km/s.
* Although the lines of the planet atmosphere are small compared to the photon noise created by the star
(both when seen in transmission or emission/reflection), for some species these lines are plentiful,
especially in the case of molecules.
* High-resolution ground-based echelle spectrographs cover wide wavelength ranges.

[FIGURE OF CHANGING RADIAL VELOCITY, PLANET IN-TRANSIT BLUESHIFT/REDSHIFT?]

By averaging many spectral lines in the planet atmosphere, the overwhelming stellar photon-noise
that dominates at all wavelengths can be beat down. Furthermore, by taking a time-series, the
changing velocity of the planet can be traced as a (predictable) shift in the average spectral line,
allowing the signal to be discriminated from that of the star or the Earth's atmosphere. If the time-series
spans a transit, the spectrum in-transit can be compared to those out of transit, providing a transmission spectrum
which is another way to discriminate between the planet's atmosphere and any other effects.

To average spectral pixels, the cross-correlation makes use of a template (or mask), which acts to select the spectral
pixels at which lines are (supposed to be) located, and prescribes with what weight these are added. The template is
shifted over a range of Doppler velocities; to map out the planet's average line-strength as a function of radial velocity.
The signal should only appear at the velocity of the planet at the time of observation, taking into account the combination of
all radial velocity components, such as the motion of the Earth around the Sun, the systemic velocity and the projected orbital
velocity of the planet. Most of the time, all these components are known, meaning that the location of any signal can be
confidently predicted ab initio, adding additional confidence that a signal candidate is actually real. Generally, a template
will have non-zero values at wavelengths of lines of interest, and is zero in-between (i.e. it has no continuum). However,
the difference between line and continuum can be fuzzy in the case of molecular species with rich absorption spectra, where
lines are blended and a continuum is formed by a forest of weak lines.

[TEMPLATE VS DATA CCF ANIMATION]


Various implementations of the cross-correlation function (CCF) exist in the literature. The function by `Snellen et al. (2010) <https://www.nature.com/articles/nature09111>`_
is a normalised function between -1 and 1, corresponding to perfect anti-correlation and perfect correlation respectively.
This quantity is unit-less, and relies on normalisation by the variance in the data and the cross-correlation template.

Tayph uses an alternative implementation that is equivalent to a weighted average of spectral pixels. In this case, the
template is scaled such that its integrated area is normalised to unity. The resulting 'CCF' is the average depth by which the
spectral lines protrude through the continuum of the planet's spectrum. In the case of transmission spectroscopy, this
quantity can be expressed as a transit radius, which can be converted to an altitude difference (in e.g. km) between where
the continuum and the line cores (on average) become optically thick. Naturally, the interpretation of this quantity depends
on the choice of template and wavelength range. The purpose of Tayph is to perform necessary formatting, correction and cleaning
steps, in order to yield accurate measurements of these average lines. It is the job of the astronomer to provide these with
scientifically meaningful interpretations.

To aide in the interpretation, an astronomer will often wish to try and compare many templates, and Tayph is designed to allow
quick looping through lists of templates. In addition, one may wish to compare the obtained cross-correlation functions to
model data. To facilitate this, Tayph optionally allows the injection of model spectra, which will be treated with the real
data in parallel. Although this does not readily allow for model parameter optimization, it does allow the astronomer
to judge whether a given line average is stronger or weaker than expected from some model in a qualitative sense; and this
is routinely applied in our recent publications, as it already greatly help the interpretations of obtained lined depths given
that our current understanding of exoplanet atmospheres is still relatively limited.

In summary, the objective of Tayph is to de-mystify and standardise the cross-correlation technique, to allow astronomers to
spend time thinking about templates, models and line depths and shapes, rather than the technical details needed to develop
their own cross-correlation codes.
