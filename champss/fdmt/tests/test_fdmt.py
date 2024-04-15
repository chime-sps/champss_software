'''test'''

import numpy as np
from fdmt.cpu_fdmt import FDMT

def test_fdmt():
    # Random data with 1024 frequency channels and 
    # 40960 time samples
    data = np.random.normal(size=(1024, 40960))
    fdmt = FDMT()  # outputting 2048 time series, the default

    # code call
    time_dt_series = fdmt.fdmt(data, retDMT=True)

    # Done. time_dt_series is an array of 2048 dedispersed time series.
