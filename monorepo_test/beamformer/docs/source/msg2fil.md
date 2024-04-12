# msgpack2fil

## Overview

This prototype of beamformer converts 4 FRB beams in a line into a beamformed tracking beam pointing towards a position in sky. The script takes in msgpack files from FRB L1 as input and outputs a filterbank file of the whole transit.

## Usage

```
python msg2fil_beamformed.py <arguments>
```

The arguments are :

- `--beam0glob`, `--beam1glob`, `--beam2glob`, `--beam3glob`
  - compulsory arguments to point to the directory where the msgpack files of the
    FRB beams. The number represents which column of beams it points to.
- `--beamno`
  - compulsory argument to the beam number in terms of row (between 0 and 255).
- `--chunksize`
  - optional argument to determine the size of chunks to work on in no of
    msgpack files.
- `--psr`
  - compulsory argument for pulsar name to put into filterbank header. Can be
    anything.
- `-o`
  - compulsory argument for the output directory and filterbank filename.
- `--fscrunch`
  - optional argument for the scrunch factor in frequency of the output
    filterbank file from 16384 channels.
- `--listonly`
  - optional argument to output a list of msgpack corresponding to the pointing.
    Only works on beams to be stiched.
- `--radec`
  - optional argument to set the RA and Dec of the pointing. Input in degrees.
    Otherwise the script computes the central RA, Dec of the msgpack files.
- `--add`
  - optional argument to apply a sensitivity weighted addition of all FRB beams
    with relative sensitivity of above 0.5 of the most sensitive beam towards
    the direction of the pointing. Otherwise a stitching method choosing the
    most sensitive beam is used instead.

The formula for the weighted addition is as of follow :

- For each channel, calculate relative sensitivity s0, s1, s2 and s3. One of them will have the value 1.
- Compute the weighting to the intensity data where `w0 = sqrt(s0/(s0+s1+s2+s3))` and so forth.
- This weighting is then applied to the intensity of each msgpack chunk on a per-channel basis.
- A straightforward sum of all the weighted intensity data is computed for the final spectra.
