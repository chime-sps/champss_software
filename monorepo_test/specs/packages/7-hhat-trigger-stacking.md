# Hhat trigger and stacking

Leader : Chia Min
Second : -

## Overview

The hhat trigger and stacking takes in the hhat arrays that passed the quality control to trigger for candidates and adding to monthly stack.
For single day hhat stack, the process is as of follow:
1. The hhat quality control (QC) process will inform whether a 2-D hhat array indexed by frequency and duty cycle at a given DM passed the QC process.
2. If passed, the hhat trigger process will receive the 2-D hhat array, apply the hhat mask and search for any (f, dc) pair with hhat value above a threshold and record the hhat value at the (RA, Dec, DM, f, dc) index.
3. If failed, the hhat QC process will inform the database of the failed DM value and the hhat trigger and stacking process will ignore the hhat array at the problematic DM.
4. The hhat stacking process will then takes in the 2-D hhat arrays that have been searched for, normalises based on the sensitivity for the day and adds into the most recent monthly stack using a logsum addition.
5. The monthly stack is normalised based on the combined sensitivity of the daily stacks added. This normalisation factor is recorded.

For monthly stacking process, the monthly stack will be first added to the cumulative stack before the trigger process starts
a. The monthly stack might have a different approach to QC which could affect how the incoming hhat slices from the monthly stack.
b. The QC can be done on any of the 3 axis so it is not necessary a (f, dc) slice at a DM.

## Interfaces

### Input: 

1. Daily stacking process
- Daily Hhat value
  - M x DM trials of 2-D Hhat array indexed by (f, dc)
- Monthly hhat stack
- Hhat RFI mask identified by "Hhat QC" for each DM trial

2. Monthly stacking process
- Monthly hhat stack indexed by (DM, f, dc)
- Cumulative hhat stack indexed by (DM, f, dc)
### Output:

1. Daily stacking process
- Hhat value at a (RA, Dec, DM, f, dc) indices that are triggered.
- Monthly hhat stack added with the latest daily hhat arrays
  - Sky coordinates
  - Ndm : # trial DM steps
  - Dm[Ndm]: trial DM values
  - Nfreq: # trial frequencies
  - Freq[Nfreq]: trial frequency values
  - Ncycle: # trial duty cycle values
  - Cycle[Ncycle]: duty cycle values
  - Hhat[Ndm][Nfreq][Ncycle]

2. Monthly stacking process
- Hhat value at a (RA, Dec, DM, f, dc) indices that are triggered.
- Cumulative hhat stack added with the latest monthly hhat stack
  - Sky coordinates
  - Ndm : # trial DM steps
  - Dm[Ndm]: trial DM values
  - Nfreq: # trial frequencies
  - Freq[Nfreq]: trial frequency values
  - Ncycle: # trial duty cycle values
  - Cycle[Ncycle]: duty cycle values
  - Hhat[Ndm][Nfreq][Ncycle]

### Database:

- for each (Sky coordinate, DM, Freq, DC, start time, end time)
  - Filename recording the samples
  - The current normalisation factor of the hhat stacks

### Metrics:

- for each (Sky coordinate, DM, Freq, DC, start time, end time)
  - The updated normalisation factor of the hhat stacks
