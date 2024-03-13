# Bonsai dedispersion

Leader : Kendrick
Second : Chia Min

## Overview

Dedispersion is applied on the sky-beam spectra. The max DM to be dedispersed to
is controlled by the database with the list of sky positions and their LoS max
DM to search for. Use the tracking database to determine if there is data to
process. The trial DM step size is 0.125 pc/cc

## Interfaces

### Input:
- Beamformer output: sky-beams of length T with N channels depending on sky position, indexed by their RA, Dec
  - Sky coordinate (RA, Dec)
  - Start, end timestamps
  - Nfreq
  - Ntime
  - Spectra[Nfrq][Ntime]
- DM model control:
  - Sky coordinate
  - Start, end timestamps
  - Max DM LOS
  - Trial DMs

### Output:
- M x DM trials dedispersed time series of length T for a sky-beam
  - Sky coordinate
  - Start, end timestamps
  - Ntime
  - Ndm
  - Spectra[Ndm][Ntime]: at each time step, a reduction of all frequencies (mean, normalized)

### Requirements:
- List of sky position with LoS max DM/channel numbers required.
- database tracking pointings that are formed.

### Database:
- for each (Sky coordinate, start time, end time)
  - Filename recording the samples

### Metrics:
- Information about the sky position where dedispersion is done

