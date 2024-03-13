# Hhat computation

Leader : Alex, Chia Min
Second : -

## Overview

1. The hhat computation is then run on each dedispersed time series, in order
   from zero DM up to max DM.
2. The hhat computation will run on each time series, with a fixed duty cycle,
   dc and a fixed nphi value that determines the trial frequency steps.
3. The output from each computation is an 1-D array of trial frequencies, f and
   their hhat value. At nphi=32 on 10 minutes long time series, this will
   produce 240,000 frequency trials between 0.01 and 50 Hz.
4. The process will repeat on each time series with 12 different trial dc value
   between 0.01 and 0.50
5. The hhat values for each different trial dc will then be appended to form a
   2-D array of f, dc for a trial DM value.

## Interfaces

### Input: 
- M x DM trials dedispersed time series of length T for a sky-beam
- Hhat process control:
  - Sky coordinate
  - Start, end timestamps

### Output:
- M x DM trials of 2-D array of hhat values indexed by f,dc. 
  - Sky coordinate
  - Start, end timestamps
  - Ndm: # trial DM steps
  - Dm[Ndm]: trial DM values
  - Nfreq: # trial frequencies
  - Freq[Nfreq]: trial frequency values
  - Ncycle: # trial duty cycle values
  - Cycle[Ncycle]: duty cycle values
  - Hhat[Ndm][Nfreq][Ncycle]

### Database:
- for each (Sky coordinate, DM, Freq, DC, start time, end time)
  - Filename recording the samples

### Metrics:
