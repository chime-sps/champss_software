# Periodic RFI Removal

Leader: Sujay
Second: Bradley


## Overview
Periodic terrestrial signals (birdies) will be a major issue, and are expected
to dominate the periodicity search candidates even if the impulsive RFI
mitigation is perfect. To reduce the number of RFI events we can inspect the
power spectrum of the zero-DM (i.e., dispersed) time series to look for large
signal power. At 0 DM, the brightest signals will be periodic RFI that should be
ignored when candidate processing begins. This process can nominally occur just
prior to, or in parallel with, the search statistic computation, but must be
complete before the search statistic can be inspected to produce candidates. It
is also possible to compare the results of this first-stage periodicity
rejection with that of other non-adjacent beams, where any commonality is almost
certainly not astrophysical.

While the topocentric time series should be used to detect the RFI signals, we
still require the respective barycentric correction factor to convert the
frequencies so that the correct (barycentric) power spectrum bins are flagged at
the candidate selection step.

## Interfaces


### Input:
A single zero-DM power spectrum with the relevant metadata.

In total, there are 3 quantities needed:
  - power spectrum intensity, `[nfreq]`
  - power spectrum frequency labels, `[nfreq]`
  - barycentric correction factor (`float`)

### Output:
A list of indices corresponding to (barycentric) power spectrum bins that
should be flagged.


### Database:
Currently there is no database interaction.


### Metrics:
- number of frequency bins flagged
