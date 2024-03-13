# multi-pointing
![Test suite status](https://github.com/chime-sps/multi-pointing/workflows/Tests/badge.svg)

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
