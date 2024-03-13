# Power Spectrum Computation

Leader: Chia Min
Second: Bradley

## Overview

The power spectrum computation is the process to convert dedispersed time series from a given observation into 
power spectra. A set of pre- and post-processing will be perform on the data. These are : 
- Normalising the time series prior to Fourier Tranform
- Barycentre the observation.
- Padding the data to have length of 2 ** 20 to improve Fourier Transform performance.
- Replace bad frequency bins in the power spectra with zeroes.
- Remove rednoise from the power spectra.

The final product from this process is a set of power spectra ready for the searching and stacking processes.

## Interfaces


### Input:

- DedispersedTimeSeries
  - A 2D array of time series dedispersed to various different DM. The DM spacing between adjacent time series 
    is currently fixed at 0.125 pc/cc.
  - A 1D array of the DM values of each dedispersed time series.
  - RA and Dec of the pointing.
  - The MJD start time of the observation.
  - The ID of the observation.

### Output:

- PowerSpectra
  - A 2D array of the power spectra from each dedispersed time series.
  - A 1D array of the DM value of each power spectrum.
  - A 1D array of the values of each frequency bin in the power spectra.
  - RA and Dec of the pointing.
  - A list of datetimes in the power spectra. Note that a power spectra can be either from a single 
    observation or from a set of observations post stacking.
  - Number of days of spectra stacked in the power spectra. 1 for newly formed power spectra.
  - The barycentric correction factor of the pointing if it consists of only a single observation.
  - A list of list of bad frequency indices for the observations in the pointing as produced by the periodic rfi 
    removal code.
  - A list of observation IDs of all the power spectra stacked. 1 ID for newly formed spectra.

### Database:

- The list of bad frequency indices for the given observation is saved into the database.

### Metrics:

- Fraction of bad frequencies in the data.
