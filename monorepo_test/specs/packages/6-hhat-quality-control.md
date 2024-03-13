# Hhat quality control

Leader : Kathryn
Second : Bradley, Chitrang

## Overview

While the hope is that the RFI excision process will remove all RFI presence in
the data, it will not be perfect and RFI can sneak into the hhat array formed.
Hence, we need to run a quality control on the hhat stack to ensure that we do
not add RFI contaminated single-day hhat stack into the monthly and subsequently
cumulative stack.

1. There will be weak periodic RFI signals present in the hhat stack that will
   appear in non-contiguous sky positions. We can cross check between hhat
   arrays at zero DM to determine these frequencies to be masked.
2. We can also identify known bad frequency values with known RFI and masked
   them in the hhat array by masking them.
3. Based on first tests of hhat, transient RFI-contaminated hhat array at zero
   DM seems to show a steady rise in hhat values at low f values. This can be
   used to inform the presence of contamination in the hhat. If the
   contamination is above a certain threshold, then the hhat array for the day
   is discarded and the discarded pointing is recorded.
4. Further tests needed to identify RFI are being tested/thought up

5. The methods should be quick since we expect the 2-D hhat array to be stacked
   onto the retrieved monthly stack as soon as possible to reduce memory cost.
6. After the initial frequency zapping, a trigger is done on the 2-D hhat array
   with hhat value of above 7.0, recording the hhat value at a (RA, Dec, DM, f,
   dc) tuple. These triggers will be sent to the candidates processor.
7. After the triggers, the daily hhat arrays are then added to the monthly
   stack.

## Interfaces

### Input:
- M x DM trials of 2-D hhat arrays of every pointing for one RA
- Monthly hhat stack of a pointing
- known bad frequencies, their widths and number of harmonics to zap - a "birdies" file
- bad frequencies identitified during the rfi stage
- DM0 topocentric hhat for each pointing
- Hhat process control:
  - Sky coordinate
  - Start, end timestamps

### Output:
- Hhat value at a (RA, Dec, DM, f, dc) indices that are triggered.
- Monthly hhat stack added with the latest daily hhat arrays.
- Hhat RFI mask?
- Monthly hhat stack "count" - how much data has been added to each (RA, Dec, DM, f, dc) point

### Requirements: 
1. Information about the current hhat array to retrieve the relevant monthly hhat stack.
2. Information about the number of days in the monthly hhat stack.

### Database:
- for each (Sky coordinate, DM, Freq, DC, start time, end time)
  - ??

### Metrics:
1. Information about rejected hhat stack (day and pointing).
2. Hhat mask
3. Number of days in the current monthly hhat stack.
