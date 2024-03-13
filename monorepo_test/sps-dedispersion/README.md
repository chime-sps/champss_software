# sps-dedispersion
FDMT (and Cordes tree) dedispersion code.

## FDMT code
- sps-dedispersion/FDMT is the FDMT code.
- sps-dedispersion/chunk.py does dedispersion chunk-by-chunk. Run it like
`python chunk.py [arguments] < [filename_in] > [filename_out]`.

## Cordes tree code
- sps-dedispersion/cordes-priv is the Cordes tree code.
- sps-dedispersion/linearize.py adds frequency channels spaced so that the f^2 dispersion
is linearized.
- To run, set the `$dm\_dir` environment variable to be the directory where the output 
should go.
- Run them like `python linearize.py < [filename_in] | dedisp.float [arguments]`, 
passing the binary data to linearize.py's STDIN. 
