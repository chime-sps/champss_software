# multi-pointing

Process candidates from multiple pointings

This script can either process canidadates by providing a path to candidate files or by reading the paths to the candidate files from a databases.

Candidate paths: `spsmp --file-path [PATH-TO-CANDIDATE-FILES]` or `python -m sps_multi_pointing --file-path [PATH-TO-CANDIDATE-FILES]`.

From db: `spsmp --use-db --db-port 27017 --date 2023/08/01 --ndays 1`

### Training a classifier

There are currently two classifier algorithm available to train a classifier model : Support Vector 
Machine and Multi Layer Perceptron, accessible via the `SvmTrainer` and `MlpTrainer` classes in 
`sps_multi_pointing.classifier.trainer`. To train a classifier : 
```python
from sps_multi_pointing.data_reader import read_multi_pointing_candidates
from sps_multi_pointing.classifier.trainer import MlpTrainer

# To read a set of MultiPointingCandidate from files in a directory
mp_cands = read_multi_pointing_candidates("/path/to/multi/pointing/candidates/")

# Ensure that the candidates have labels attached to it :
print(mp_cands[0].classification)
# yield `CandidateClassification` class object with label and grade
# The label should be `CandidateClassificationLabel` with either RFI, Ambiguous or Astro

# To train a classifier
mlp_trainer = MlpTrainer(**kwargs)
# where kwargs is a dict with various properties of the algorithm 
# see scikit-learn documentation for list of arguments : 
# https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
mlp_trainer.train(mp_cands, compute_metrics=True, save_model=True, filename="./mlp_classifier.pickle")
# This will save the classifier in `mlp_classifier.pickle`, with the metrics, features used and python
# package versions in a mlp_classifier_metadata.txt file
```

### Updating the known source database

The known source database is a MongoDB created from two scripts. The first is `write_psrcat_to_ksdb.py` and it uses `psrqpy` to grab the latest version of the ATNF catalogue database file. The second is `write_psrscraper_to_ksdb.py`. This script grabs known sources from David Kaplan's Pulsar Scraper, which contains unpublished pulsars, often acquired by 'scraping' pulsar survey websites where new discoveries are posted. 

To populate a known source database, run `write_psrcat_to_ksdb.py`, followed by `write_psrscraper_to_ksdb.py`. Specify the database name, host and port. The update option is also needed if you want to only update the database with new sources or known sources that have updated ephemerides. Otherwise, all new sources will be added to the database and any existing sources in the databse will be replaced. There is a check for duplicate sources performed in `write_psrscraper_to_ksdb.py` to ensure the same pulsar is not added multiple times. 

```
python3 write_psrcat_to_ksdb.py --db-host sps-archiver1 --db-name test --update

python3 write_psrscraper_to_ksdb.py --db-host sps-archiver1 --db-name test --update
```


