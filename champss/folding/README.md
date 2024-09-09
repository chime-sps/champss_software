# folding
Follow-up of CHAMPSS candidates and known pulsars, including folding and timing tools

# Main Folding Script

The main processing script is fold_candidates.py, which can fold on a candidate on a certain day. The inputs are either a given ra, dec, f0, dm, a known pulsar using --psr flag, or a database id for the FollowUpSources database collection, which keeps track of automatic folding, and has the path to an ephemeris for each candidate/known_pulsar.  Example usages:

```
fold_candidate --psr 'B2217+47' --date 20220621
```

```
fold_candidate --ra 334.95 --dec 47.91 --f0 1.8571 --dm 43.5 --date 20220621
```

```
fold_candidate --fs_id '661014b27139411ac7bc5266' --date 20240330 --db-name 'sps-processing' --db-port 27017 --db-host 'sps-archiver'
```

This code find the nearest pointing from the beamforming strategist, beamforms the data, saves as a filterbank, folds with dspsr, creating archive files and a diagnostic plot.

# Automatic Processing

The automatic daily processing folds sources and candidates daily based on the contents of the FollowUpSources collection in the CHAMPSS mongodb.  Daily candidates are filtered, and added to the database using filter_mpcandidates.py, and the automatic queueing of these jobs is handled with Workflow by processing.py in the main pipeline repo

New and known sources can be added to the FollowUpSources database using the tools in sps-databases.

# Phase-coherent search for stack candidates

The power spectrum stack candidates do not recover the spin parameters precisely enough to phase-connect between days.  A phase-coherent search to confirm stack candidates is done with confirm_cand.py, which searches a grid of f0, f1 values surrounding the candidate, and computes the chi-squared statistic.

This is now queued through workflow, and uses the FollowUpSources database collection.  The multiday pipeline takes a stack candidate, adds it to the database, folds up to N days, then run the coherent search, and just needs the candidate path as an input:

```
multidayfold_pipeline --candpath '/data/lkuenkel/mp_sanity_check/new_stack_tests/mp_runs/new_0625_euclidean_no_pos/candidates/Multi_Pointing_Groups_f_2.684_DM_32.788_class_Astro.npz' --db-name "sps-processing"
```

# Timing

Automatic timing tools are a work in progress..
