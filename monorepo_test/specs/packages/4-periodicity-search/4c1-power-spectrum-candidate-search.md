# Power Spectrum Candidate Search

Leader: Chia Min
Second: Bradley, Kathryn


## Overview

The power spectrum candidates search employs the harmonic summation method to improve the sensitivity towards pulsars 
with small duty cycle. The harmonic sum code will look for the indices of a spectrum that is closest to the harmonic 
to be added to the power spectrum. The harmonic sum code will work on 1, 2, 4, 8, 16 and 32 harmonics added together, 
which then a script will compute the power-to-sigma conversion for particular harmonics being summed toegther. A sigma 
threshold is used to identify the frequencies where the power value is above the sigma threshold, and the harmonic 
information of the detection is duly recorded. 

To speed up the harmonic summing processes, the candidate search code precomputes the indices required to run the 
harmonic summation process and applies it on the data. The final product will be a list of detections along with 
information useful to identify pulsar candidates from RFI/noise.

## Interfaces


### Input:

- PowerSpectra
  - A 2D array of the power spectra from each dedispersed time series.
  - A 1D array of the DM value of each power spectrum.
  - A 1D array of the values of each frequency bin in the power spectra.
  - RA and Dec of the pointing.
  - A list of datetimes in the power spectra.
  - Number of days of spectra stacked in the power spectra. This will effect the power-to-sigma calculation.
  - The barycentric correction factor of the pointing if it consists of only a single observation.
  - A list of list of bad frequency indices for the observations in the pointing as produced by the periodic rfi 
    removal code.
  - A list of observation IDs of all the power spectra stacked.

### Output:

- PowerSpectraDetection
  - A list of all the detection made by the candidates search process. Each detection comes with : 
    - The spin frequency of the detection.
    - The DM of the detection.
    - The number of harmonics summed.
    - A 2 x num_harm structure array of the raw power from original spectra and the frequencies of the harmonics.
    - The sigma of the detection.
  - The spacing between adjacent frequency bins of the power spectra.
  - RA and Dec of the pointing where detections are made.
  - The minimum sigma of the search process.
  - The list of observation IDs within the pointing searched.

### Database:

- No database interaction is expected at this stage.

### Metrics:

- No metric is expected to be recorded here. The candidates information will be recorded in candidates processing step.