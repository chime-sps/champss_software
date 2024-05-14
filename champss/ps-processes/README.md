# ps-processes
![Test suite status](https://github.com/chime-sps/ps-processes/workflows/Tests/badge.svg)

Power spectrum creation and stacking from the CHIME/SPS online search

Version 1.0 : revamped the scripts to have the various processing class object to run on 
dedispersed time series and power spectra from all DM values in one go, instead of processing
each DM separately

## Installation
Assuming you are in the top-level directory:
```
python setup.py install [--user]
```
or
```
pip install . [--user]
```
## Python API

The docstrings of `PowerSpectraCreation`, `PowerSpectraSearch`, `PowerSpectraStack` provides the various processing 
options for the class objects.

### Creating power spectra from a given set of dedispersed time series of an observation
```python
from ps_processes.processes.ps import PowerSpectraCreation
from ps_processes.interfaces import DedispersedTimeSeries

# to load a set of dedispersed time series in presto .dat file format in a specific directory
dedisp_ts = DedispersedTimeSeries.read_presto_datfiles("/path/to/dat/files/", obs_id)
# initialise the class object to create power spectra from dedispersed time series
ps_creation = PowerSpectraCreation()
# returns a PowerSpectra class from the dedispersed time series
power_spectra = ps_creation.transform(dedisp_ts)
# save the spectra in an hdf5 file
power_spectra.write_hdf5_file("/path/to/power/spectra/power_spectra.hdf5")
```

### Searching a power spectrum
```python
from ps_processes.processes.ps_search import PowerSpectraSearch
from ps_processes.interfaces import PowerSpectra

# to load a power spectra
power_spectra = PowerSpectra.read_hdf5_file("/path/to/power/spectra/power_spectra.hdf5")
# initialise the class object to search the power spectra
power_spectra_search = PowerSpectraSearch()
# returns a PowerSpectraDetections class object from the search process
power_spectra_detections = power_spectra_search.search(power_spectra)
# save the detections in an hdf5 file
power_spectra_detections.write_hdf5_file("/path/to/power/spectra/detections/power_spectra_detections.hdf5")
```

### Stacking power spectra
```python
from ps_processes.processes.ps_stack import PowerSpectraStack
from ps_processes.interfaces import PowerSpectra

# to load the power spectra to stack and the power spectra stack
power_spectra = PowerSpectra.read_hdf5_file("/path/to/power/spectra/power_spectra.hdf5")
power_spectra_stack = PowerSpectra.read_hdf5_file("/path/to/power/spectra/power_spectra_stack.hdf5")
# initialise the class object to stack power spectra
stack_power_spectra = PowerSpectraStack()
# stack the power spectra to the stack
power_spectra_stack = stack_power_spectra.stack(power_spectra, power_spectra_stack)
# save the power spectra stack in an hdf5 file
power_spectra_stack.write_hdf5_file("/path/to/power/spectra/stack/power_spectra_stack.hdf5")
```

### Command line pipeline script
New pipeline script `run_ps_processes` can be used to processes dedispersed data of all DM from 
presto to create stacked power spectra and single day candidates. To use:
```
run_ps_processes [options]
```
### Required arguments
`--obs` : ID number of the observation to be processed.

`--basepath` : Path to the base directory of the SPS data products

`--subdir` : Path to the sub directory of the SPS data products. It should be in the structure YYYY/MM/DD/<ra>_<dec>

### Optional arguments
`--no-ps` : Do not run the fft process to create power spectra from dedispersed time series

`--no-search` : Do not run the search process to find detections from the power spectra

`--stack` : Option to stack the daily power spectra to the monthly stack. The monthly stack is written to '<basepath>/stack/<ra>_<dec>_power_spectra_stack.hdf5'

`--write_spectra` : Option to write the daily power spectra to '<basepath>/<subdir>/<ra>_<dec>_power_spectra.hdf5'.

`--no-write-detections` : Option to NOT write the power spectra detection to a '<basepath>/<subdir>/<ra>_<dec>_power_spectra_detections.hdf5'

`--no-update-db` : Option to NOT update the SPS databases of the power spectra processes

`--tsamp` : Sampling time of the dedispersed timeseries in seconds (Default : 0.00098304 s).

`--no-norm` : Option to NOT normalise the dedispersed time series.

`--no-bary` : Option to NOT barycentre the dedispersed time series.

`--padded-length` : padded length of the dedispersed time series. (Default : 1048576)

`--no-qc` : Option to NOT run quality control to remove bad frequency bins.

`--no-zero-replace` : Option to NOT doing a zero replacement of the bad freqeuncy bins that are being flagged

`--no-rednoise` : Option to NOT remove rednoise from the FFTs.

`--harm` : Number of harmonics to search the FFTs to (Default = 32).

`--sigma` : The minimum sigma of the candidates (Default = 5.0).

`--precompute-harm` : Precomputes the indices used for harmonic summing prior to the harmonic search operation.