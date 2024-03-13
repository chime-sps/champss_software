# folding
Folding of candidates and known pulsars

To filter through all candidates for a given day:
'''python
python Candidate Filter.py “year month day”
'''
To just fold on a single pulsar:
'''python
python Candidate Filter.py “year month day -k dm f0 ra dec”
'''

After the npz file is created, run the folding script:
'''python
python fold_dspsr.py “year month day candfile.npz”
'''

It will output an ephemeris, filterbank file, archive file, and candidate plot png
