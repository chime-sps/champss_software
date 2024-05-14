# Power Spectrum Stacking

Leader: Chia Min
Second: Bradley

## Overview

The power spectrum stacking is the process to add the power spectra produced for a given observation from a given 
pointing to the stack consisting of all power spectra produced for a given pointing. 

The stacking process itself is very simple, it is just a straight summing of the normalised single day power spectra 
into the stack currently preserved.

The current plan is to have two separate stack, a monthly stack that contains power spectra from observations done over 
the past 30 days, and a cumulative stack that consists of power spectra from the very beginning of the survey.

A quality control process is planned to ensure the power spectra from a given day is good enough to not contaminate the 
power spectra stack. The actual algorithm to do so is yet to be decided. 

## Interfaces


### Input:

- PowerSpectra from a single day
  - A 2D array of the power spectra from each dedispersed time series.
  - A 1D array of the DM value of each power spectrum.
  - A 1D array of the values of each frequency bin in the power spectra.
  - RA and Dec of the pointing.
  - A list of datetimes in the power spectra. It should only have one entry here.
  - Number of days of spectra stacked in the power spectra. This should be one.
  - The barycentric correction factor of the pointing if it consists of only a single observation.
  - A list of list of bad frequency indices for the observations in the pointing as produced by the periodic rfi 
    removal code.
  - A list of observation IDs of all the power spectra stacked. It should only have one entry here.
  
- PowerSpectra stack
  - A 2D array of the power spectra from each dedispersed time series.
  - A 1D array of the DM value of each power spectrum.
  - A 1D array of the values of each frequency bin in the power spectra.
  - RA and Dec of the pointing.
  - A list of datetimes in the power spectra.
  - Number of days of spectra stacked in the power spectra.
  - The barycentric correction factor of the pointing if it consists of only a single observation.
  - A list of list of bad frequency indices for the observations in the pointing as produced by the periodic rfi 
    removal code.
  - A list of observation IDs of all the power spectra stacked.

### Output:

- PowerSpectra stack added with latest single day spectra.
  - A 2D array of the power spectra from each dedispersed time series.
  - A 1D array of the DM value of each power spectrum.
  - A 1D array of the values of each frequency bin in the power spectra.
  - RA and Dec of the pointing.
  - A list of datetimes in the power spectra, plus the datetime of the newest observation.
  - Number of days of spectra stacked in the power spectra, plus one
  - The barycentric correction factor of the pointing if it consists of only a single observation.
  - A list of list of bad frequency indices for the observations in the pointing as produced by the periodic rfi 
    removal code.
  - A list of observation IDs of all the power spectra stacked, plus the ID of the newest observation.


### Database:

- A PsStack table is used to record the power spectra stack. Each pointing will have it's own unique entry. The 
properties saved are :
  - ID of the pointing
  - path to the monthly and cumulative stack
  - datetimes of the observations included in the stack
  - number of days in the monthly and cumulative stack.

### Metrics:
