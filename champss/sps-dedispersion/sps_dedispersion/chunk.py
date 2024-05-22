'''Reads in a single binary .dat waterfall from stdin and writes out the
dispersion transform.'''

import sys
import numpy as np
from sps_dedispersion.fdmt.cpu_fdmt import FDMT

# The following is for memory inspection
import linecache
import os
import tracemalloc
from datetime import datetime
from queue import Queue, Empty
from resource import getrusage, RUSAGE_SELF
from threading import Thread
from time import sleep



def do_fdmt_from_stdin(fdmt, nbytes, nsamps, maxsamps, dtype,
                    verbosity=0):
    '''Do FDMT on binary data from stdin.

    Parameters:
        fdmt:       an initialized FDMT instance
        nbytes:     the number of bytes per data point in the file
        nsamps:     the number of time samples per chunk beyond the minimum
                    (which minimum is equal to the number of time samples'
                    delay at maxdm).
        dtype:      the numpy data type
        verbosity:  0 for no output; 1 for basic output; 2 for obnoxiously
                    loud
        memory_check:   0 for nothing; 1 to output line-by-line memory usage
    '''
    if verbosity >= 1:
        sys.stderr.write(f'beginning file...' + '\n')
    if verbosity >= 2:
        sys.stderr.write(f'initializing helper arrays...' + '\n')
    # ingredients for padding the inputs. we pad the inputs with an 
    # extra maxdm time samples so that we can combine the dm transform
    # arrays nicely.
    zeroes = np.zeros((fdmt.maxDT, fdmt.nchan), dtype=dtype)
    pad_outarr = np.empty((nsamps + fdmt.maxDT, fdmt.nchan), dtype=dtype)

    # we hold onto the last maxDT time samples of the most recently
    # processed dm transform array (dmt). then, after we process the next 
    # dmt, we add these maxtd time samples to the first maxtd time 
    # samples of the newly processed dmt, and write out the result, 
    # except we hold onto the last maxtd time samples without writing it 
    # out so that we can add it to the next dmt.
    prev_dmt_tail = np.zeros((fdmt.maxDT, fdmt.maxDT), dtype=dtype)

    if verbosity >= 2:
        sys.stderr.write('done initializing arrays' + '\n')

    totalread = fdmt.nchan * nbytes * nsamps
    indata = sys.stdin.buffer.read(totalread)
    idx = 0
    while indata != b'' and (maxsamps == -1 or idx * nsamps < maxsamps):
        if verbosity >= 2:
            sys.stderr.write(f'iteration {idx}/{maxsamps // nsamps}\n')
        outarr = np.frombuffer(indata, dtype=dtype).reshape(-1, fdmt.nchan)
            
        # outarr is just a nchans * nsamps float32 array
        # pad outarr by an extra nchans zeroes so that we
        # can combine it with other fdmt outputs
        try:
            np.concatenate((outarr, zeroes), axis=0, out=pad_outarr) 
        except ValueError:
            pad_outarr = np.concatenate((outarr, zeroes), axis=0)

        if verbosity >= 1:
            sys.stderr.write('starting FDMT...\n')
        fdmt.reset_ABQ()
        dmt = fdmt.fdmt(pad_outarr.T, retDMT=True).T

        # write out the dmts.
        if idx == 0:
            # if this is the first chunk, delete the first maxDT samples.
            dmt = dmt[fdmt.maxDT:]
        else:
            dmt[:fdmt.maxDT] += prev_dmt_tail
        prev_dmt_tail = dmt[-fdmt.maxDT:]
        sys.stdout.buffer.write(dmt[:-fdmt.maxDT].tobytes())

        indata = sys.stdin.buffer.read(totalread)
        idx += 1
        
        # # if we have read the entire file, we need to write out 
        # # the final parts of the dmt, so as not to lose it. 
        # if indata == b'':
        #     sys.stdout.buffer.write(prev_dmt_tail.tobytes())
        

# From https://stackoverflow.com/questions/552744/
def display_top(snapshot, key_type='lineno', limit=3, verbosity=0):

    if verbosity >= 1:
        out = sys.stdout
    else:
        out = os.devnull

    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit, file=out)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line, file=out)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024), file=out)
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024), file=out)


def memory_monitor(command_queue: Queue, poll_interval=1, verbosity=0):

    if verbosity >= 1:
        out = sys.stderr
    else:
        out = os.devnull

    tracemalloc.start()
    prev_max = -1
    snapshot = None
    while True:
        try:
            command_queue.get(timeout=poll_interval)
            if snapshot is not None:
                print(datetime.now(), file=out)
                display_top(snapshot, verbosity=verbosity)

            return
        except Empty:
            max_rss = getrusage(RUSAGE_SELF).ru_maxrss
            if max_rss != prev_max:
                prev_max = max_rss
                snapshot = tracemalloc.take_snapshot()
                print(datetime.now(), 'max RSS', max_rss, file=out)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='''Perform FDMT dedispersion on one stream of binary
        intensity data, with the result returned as multiple time series to
        STDOUT. Read in the intensity data chunk by chunk rather than all at
        once so as not to hold all the data in memory at once.''')


    parser.add_argument('nchans', type=int, help='the number of frequency '
                        'channels in the data in the file (default=1024)')
    parser.add_argument('nbytes', type=int, help='the number of bytes '
                        'per data point in the file (4)')
    parser.add_argument('nsamps', type=int, help='the number of samples '
                        'to read at a time (8192)')
    parser.add_argument('maxdm', type=int, help='the maximum DM to which '
                       'to dedisperse, in units of time steps dispersed. '
                        '(2048)')
    parser.add_argument('maxsamps', type=int, help='the maximum number '
                        'of samples to read; or -1 to read the entire '
                        'file. (-1)')
    parser.add_argument('minfreq', type=float, help='min frequency')
    parser.add_argument('maxfreq', type=float, help='max frequency')
    parser.add_argument('--verbose', '-v', action='count', help='verbosity '
                        '(two levels)', dest='verbosity', default=0)
    parser.add_argument('--memory-check', action='store_true', default=False,
                        help='if present, check FDMT memory usage '
                        'periodically')

    args = parser.parse_args()
    
    fdmt_init_dict = dict(
      fmin=args.minfreq,
      fmax=args.maxfreq,
      nchan=args.nchans,
      maxDT=args.maxdm,
    )
    fdmt = FDMT(**fdmt_init_dict)

    # other useful things
    dtype = np.dtype(f'f{args.nbytes}')

    if args.memory_check:
        queue = Queue()
        poll_interval = 0.05
        monitor_thread = Thread(target=memory_monitor, args=(queue,
                                                             poll_interval,
                                                             args.verbosity))
        monitor_thread.start()

    try:
        do_fdmt_from_stdin(fdmt, args.nbytes, args.nsamps,
                           args.maxsamps, dtype, args.verbosity)
    finally:
        if args.memory_check:
            queue.put('stop')
            monitor_thread.join()

