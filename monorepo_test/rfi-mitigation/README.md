# rfi-mitigation
![Test suite status](https://github.com/chime-sps/rfi-mitigation/workflows/Tests/badge.svg)

A repository containing the RFI excision pipeline for the CHIME Slow Pulsar Search 
project.

## Overview
The SPS RFI excision pipeline can use a number of methods to remove RFI. The 
best performance is realised when incorporating the CHIME/FRB L1 RFI mask and weights 
in addition to some of the filters defined in this package. The RFI excision requires 
data "portraits" of shape `[Nfreq][Ntime]` and promotes the 3-bitized, encoded intensity
data sent from the special SPS L1 module to 32-bit floats for calculations. The RFI mask 
computed independently for each channel over a specific time scale (nominally 1024 1-ms 
samples) and is combined with the mask from CHIME/FRB L1 to realise a new time- and 
frequency-dependent sample mask. The mask is stored separately from the data (i.e., no 
data are modified in this process, other than the promotion to 32-bit floats). In the 
current iteration, it is recommended that the CHIME/FRB L1 mask and weights be applied, 
along with the known bad channel list and then the spectral kurtosis filter. The other 
cleaners are slow and rarely improve the mask significantly.


## Installation
Assuming you are in the top-level directory:
```
python setup.py install
```
or 
```
pip install .
```
(Add the `--user` flag if you want a user-local install.)

The package is now available for import like
```python
import rfi_mitigation
```

If you want support for reading in raw FRB L1 msgpack data (as opposed to the 
Huffman-encoded version), you will need to explicity ask for its installation as 
follows:
```
pip install .[l1-msgpack-support]
```
where the square-braces are included on the command-line. 


## Running tests
Assuming you are in the top-level directory, and that you have successfully installed 
the package, simply run
```
pytest
```

## Building documentation
Assuming you are in the top-level directory:
```
cd docs
make html
cd build
# open contents.html in your favourite browser 
# (e.g. > firefox contents.html)
```

The `docutils` package is required for this process, and can either be installed 
directly
```
pip install docutils
```
or via this package's extras
```
pip install .[docs]
```

## Running the RFI mitigation pipeline
### From within another script
```python
from rfi_mitigation.pipeline import RFIPipeline
from rfi_mitigation.reader import DataReader

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

# Initialise the cleaning RFIPipeline object
rfiPipe = RFIPipeline(masking_dict)
 
# If reading data from files on disk, a DataReader instance also need to 
# be initialised.
reader = DataReader(apply_msgpack_weights=True, apply_l1_mask=True)

# Read the data from disk into special SlowPulsarIntensityChunk objects
# which are the interfaces between the onloading data step, cleaning 
# stage and the subsequent beamforming component.
# NOTE: there are keyword arguments to specify whether the data are 
# callback/raw FRB intensity data, but the default is to assume standard 
# SPS quantized data.
chunks = reader.read_files(msgpack_list)  # a list of SlowPulsarIntensityChunk objects

# Clean the data chunks using the filters configured at initialisation. 
# The output of this step is a list same length as `chunks`, but now the 
# data have had their masks updated. 
cleaned_chunks = rfiPipe.clean(chunks)

# If writing the output to disk, the interface has an easy `write` method, e.g.,
for c in cleaned_chunks:
    c.write()  # write data into an HDF5 file format
    
# Otherwise, you would hand off this list of SlowPulsarIntensityChunk objects to
# the beamformer process (or an intermediate shepherding process that collects the 
# required cleaned data segments for a given beamforming operation).
```
Note that the code will naively just load ALL data into a contiguous array, so there 
will have to be some external logic that splits up data files into manageable sets. 

### From the command line
An example of how a user can utilise the `cli.py` script to clean SPS intensity data.
```bash
srcname="B1919+21"  # the source name  (for book-keeping)
srcpos="19:21:44.815 +21:53:02.25"  # the source coordinates (TODO: can probably remove)
beam="1059"  # the FRB beam number (for bookkeeping)
s=2290  # start processing on this file number
e=2291  # end proceessing on this file number
length=1  # combine this many files worth into one "chunk"

# create a folder structure in the current working dir
odir="${srcname}/${beam}/2021-03-28"
mkdir -p $odir
cd $odir

# assuming you are logged in to sps-st1
fpath="/data/chime/sps/raw/2021/03/28/1059"
echo cli.py -n $srcname -p "$srcpos" -b $beam -i $s $e -l $length --badchan --ps $fpath
cli.py -n $srcname -p "$srcpos" -b $beam -i $s $e -l $length --badchan --ps $fpath
cd -
```
