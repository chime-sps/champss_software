# RFI excision tool

Leader : Bradley

Second : Kathryn, Chitrang

## Overview

The SPS RFI excision pipeline can use a number of methods to remove RFI.
Typically, the best performance is realised when incorporating the CHIME/FRB L1
RFI mask and weights in addition to the other cleaning processes here. The RFI
excision requires data chunks of shape `[Nfreq][Ntime]` and promotes the
nominal 3-bit intensity data sent from L1 to 32-bit floats for calculations.
The RFI mask computed at each different cleaning step is combined with the mask
from CHIME/FRB L1 to realise a new time- and frequency-dependent sample mask.
The mask is currently stored separately from the data (i.e., no data are
modified in this process, other than the promotion to 32-bit floats). In the
current iteration, it is recommended that the CHIME/FRB L1 mask be applied,
along with the known bad channel list and then the spectral kurtosis filter.
The other cleaners are slow and rarely improve the mask significantly.

### Current processing steps
- Create a `RFIPipeline` object by passing four things:
  - a list (even if a single file) of file paths to load
  - the FRB beam number
  - a dictionary noting which cleaners to run
  - an integer (observation ID) or dictionary (database object)
    - default = 1 (i.e. first database entry)
- Read data from given file list into a contiguous `numpy.ma.MaskedArray` of
shape `[Nfreq][Ntime]`
  - if data are read from `.msg` or `.msgpack` files, the CHIME/FRB L1 weights
  and RFI masks are automatically applied
  - if data are read from a `.hdf5` file, it is assumed the format is as
  described below in the "Output" section
- Depending on what configuration was specified on the command-line, run any
number of cleaning procedures and update the data mask in-place
  - i.e., each successive cleaning method works on the masked output of the
  previous by using `numpy`'s `MaskedArray` algorithms
  - current order that methods apply:
    1. CHIME/FRB L1 mask application (no compute required)
    2. known bad channel mask applied  (no compute required)
    3. spectral kurtosis (channel-independent, operates on 1024 time samples)
- Write output to either: HDF5 file (preferred), or filterbank file (requires
  additional processing to replace masked data)

## Interfaces

### Input:
For a single beam:
- Data pack containing the L1 3-bit intensity data
  - Also contains missing samples / RFI mask
  - Can be ~any length, but **1024 must divide the number of time samples**
- Known bad channel mask **(Requires re-evaluation)**

### Output:
- Raw intensity data `[Nfreq][Ntime]`
- Bad sample mask `[Nfreq][Ntime]`: 1 - masked; 0 - unmasked
- Metadata:
  - Beam number
  - Start, end timestamps
  - `Nfreq`: number of frequency channels
  - `Ntime`: number of time samples

For the prototyping phase, files will be written between each successive step.
The RFI processing pipeline will output one file:
- 1 HDF5 file containing two (2) datasets and metadata, named
`[startunixtime]_[enduntixtime].hdf5` (times truncated to integers)
  - `intensity` (raw unmasked data)
  - `mask` (an integer mask for the intensity data, 1 = masked)
  - metadata stored as HDF5 file attributes:
    - `beam_number` (FRB beam number, int)
    - `start` (start unix time of first sample, float)
    - `end` (end unix time of last sample, float)
    - `nchan` (number of frequency channels, int)
    - `ntime` (number of time samples, int)

### Database:
Does not currently interact (read/write) from database, but may need to in the
future, thus the basic skeleton infrastructure is there to keep a track of
which observation/pointing is being processed.

### Metrics:
- For each cleaner:
  - run time (total and per-channel)
  - fraction of data masked by that specific cleaner
- Total cleaning/processing time (not super helpful if part of a chain)
- Total masked fraction of input data (before/after also possible)

### Usage example
The `rfi-excision` package should solve its own dependency problems, but it
relies on:
  - `sps_common`
  - `sps_databases` (technically not used, but here for posterity)
  - `ch_frb_l1`

Here is a basic Python script example of how to use the `RFIPipeline` object:
```python
from rfi_mitigation.pipeline import RFIPipeline

masking_dict = dict(
weights=True,  # apply msgpack weights
l1=True,  # apply L1 RFI mask
badchan=True,  # apply known bad channel mask
kurtosis=False,  # run temporal kurtosis filter
mad=False,  # run median absolute deviation outlier filter
sk=True,  # run spectral kurtosis filter
powspec=False,  # run a power spectrum based filter
dummy=False  # run a dummy filter that does nothing
)

msgpack_list = [...]  # nominally a list of N paths to msgpack data files
frb_beam_number = 0  # the FRB beam number
obs_id = 1  # observation key (corresponding to entry in database)

# if the msgpack files are archived data (suffix = .msg) rather than
# callback data (suffix = .msgpack)
is_callback = False

rfiPipe = RFIPipeline(msgpack_list, frb_beam_number, masking_dict, obs_id)
rfiPipe.read_chunk(is_callback=is_callback)
rfiPipe.clean()
rfiPipe.write_to_hdf5()
```
