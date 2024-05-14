# Bonsai dedispersion

Leader : Kendrick
Second : Chia Min

## Overview

Dedispersion is applied on the sky-beam spectra. The max DM to be dedispersed to
is controlled by the database with the list of sky positions and their LoS max
DM to search for. Use the tracking database to determine if there is data to
process. The trial DM step size is 0.125 pc/cc

### Prepsubband

Currently, the presto routine `prepsubband` is being used for dedispersion process. 
The input for running `prepsubband `

### Bonsai

Eventually, bonsai will be used for dedispersion

## Interfaces

### Input:
- Beamformer output: sky-beams of length T with N channels depending on sky position, indexed by their RA, Dec
  - Sky coordinate (RA, Dec)
  - Start timestamp
  - nchan
  - ntime
  - Spectra[nchan][ntime]
  - For `prepsubband`, input is a filterbank file described in 3a
- DM model control:
  - Sky coordinate
  - Max LoS DM
  - `ndm` number of DM trials

### Output:
- `ndm` number of DM trials dedispersed time series of length `ntime` for a sky-beam
  - Sky coordinate
  - Start, end timestamps
  - ntime
  - ndm
  - Spectra[ndm][ntime]: at each time step, a reduction of all frequencies (mean, normalized)
  - For `prepsubband`, output is written out as M number of dedispersed time series at each DM trial with
    filenames <working_directory>/YYYY/MM/DD/<beam_no>/<RA>_<Dec>_DM<trial_dm>.dat

### Requirements:
- List of sky position with LoS max DM/channel numbers required.
- database tracking pointings that are formed.

### Database:
- for each (Sky coordinate, start time, end time, observation date)
  - Filename recording the samples

### Metrics:
- Information about the sky position where dedispersion is done

