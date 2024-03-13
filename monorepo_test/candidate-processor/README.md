# candidate-processor
![Test suite status](https://github.com/chime-sps/candidate-processor/workflows/Tests/badge.svg)

Makes and processes candidates from H-hat arrays/ power spectrum candidates.

## Installation

This code requires `sps-common` as a dependency (https://github.com/chime-sps/sps-common).
Once you have `sps-common` installed:

```
>>> git clone https://github.com/chime-sps/candidate-processor.git
>>> cd candidate-processor
>>> pip install -e .
```
## Test
```
>>> pytest [-sv]
```
## Runing CLI instructions

```
>>> python cli.py -h
usage: cli.py [-h] [-f F] [-g G] [--eps EPS] [--max-harm MAX_HARM] [--min-harm MIN_HARM] [--min-freq MIN_FREQ]
              [--min-dm MIN_DM] [-o O] [-o_hrcs O_HRCS] [--config CONFIG]

optional arguments:
  -h, --help           show this help message and exit
  -f F                 hdf5 file name containing the power spectra detections
  -g G                 min group size to be considered a cluster
  --eps EPS            the eps value for the DBSCAN clustering process
  --max-harm MAX_HARM  the maximum harmonic to match to a cluster
  --min-harm MIN_HARM  the minimum fractional harmonic to match to a cluster
  --min-freq MIN_FREQ  the minimum frequency to consider a detection cluster as a candidate
  --min-dm MIN_DM      the minimum dm to consider a detection cluster as a candidate
  -o O                 output filename to store the single pointing candidate collection npy file
  -o_hrcs              output npz filename to store the harmonically related clusters, if wish to write them
  --config             features configuration file, will default to the example yaml in candidate-processor
```

* Typical example:
```
>>> python cli.py -f /path/to/data/pointing_power_spectra_detections.hdf5 -g 5 -o /path/to/data/pointing_power_spectra_detections.npy
```

* Note:
The HarmonicallyRelatedClusters objects are an intermediary step. You can choose to write them out with the `-o_hrcs` argument.
In normal operations this will not be necessary but is useful for debuggin/testing when coding a new feature

## Overview/What do all these class names mean?
`PowerSpectraDetections` input from `ps_processes`, points in frequency-DM space where the power spectra's significance was above some threshold sigma.  

-> clustering ->  

`SinglePointingDetectionClusters` - clusters of detections, found by DBSCAN, in freq-DM space. This object contains all the clusters for one pointing.

-> harmonic filter ->  

`HarmonicallyRelatedClusters` - a group of clusters found to be harmonically-realted to each other, based on the max-sigma frequency in each cluster  
(each group has a `main_cluster` which is the highest sigma cluster which also has a DM > some threshold and a frequency > some threshold)  
(many may packaged together into a `HarmonicallyRelatedClustersCollection` for reading/writing)  

-> feature generation ->

`SinglePointingCandidate` - made features from the raw information in a `HarmonicallyRelatedClusters`, so the raw freq-DM clusters themselves are not included in these.  
(commonly packaged together into a `SinglePointingCandidateCollection` for reading/writing)  

e.g. 
* start with 30,000 points over the threshold sigma, in a `PowerSpectraDetections` instance
* DBSCAN finds 30,000 clusters in these points, all 3,000 of these clusters are contained in a `SinglePointingDetectionClusters` instance
* the harmonic filter sorts these into 300 groups of clusters. You have 300 `HarmonicallyRelatedClusters` instances, the first contains 32 clusters, then second contains 5 clusters, etc
* features are extracted from each individual group and a `SinglePointingCandidate` is made from each `HarmonicallyRelatedClusters`. You have 300 `SinglePointingCandidate`s


## Features
### Configuration file
Features to be generated are specified in a configuration yaml file which is of the form
```yaml
<datacode1>:
  - feature: <specific_feature_class_name1>
    flags:
      - <flag1>
      - <flag2
  - feature: <specific_feature_class_name2>
    options:
      <option_key1>: <option_value1>
      <option_key2>: <option_value2>

<datacode2>:
  - feature: <specific_feature_class_name3>
```
Specific example - your desired features are the skewness of the DM, weighted by sigma, the kurtosis of the DM weighted by sigma and calculated about the point of peak sigma, and the mean of the DM:
```yaml
DM:
  - feature: Skewness
    flags:
      - weighted
  - feature: Kurtosis
    flags:
      - weighted
      - about_peak
  - feature: Mean
```

The datacode specifies what data you want to derive the feature from, `- feature: ` lines specify the specific feature you want, e.g. Mean, and it must match a class name in `utilities/features.py`, `flags:` then specifies any True/False flags you wish to pass in and `options:` sets any optional arguments you wish to specify.

Note with this yaml formatting, when read, each datacode will have an entry which is a list, with one entry per feature, that list contains `feature`, a list of `flags` any are specfied, and an `options` dictionary if any are specified.
(in yaml formating "-" means it's a list entry)

The exception to the above formatting is the `Property` datacode, which can get any property of a `HarmonicallyRelatedClusters` instance and its section of the yaml file should look like:
```yaml
Property:
  - <property1>
  - <property2>
  - <property3>
```
Specific example - you wish to extract the `max_dm` and `unique_freqs` properties:
```yaml
Property:
  - max_dm
  - unique_freqs
```

#### Currently accepted configurations

| feature type | datacodes   | flags                | option: key                 | option: possible values | option: default |
|--------------|-------------|----------------------|-----------------------------| ----------------------- | ------------------- |
| Property     | Property    |
| Stat         | DM <br /> DCoverF | weighted <br /> about_peak |  
| Fit          |             |                      | amalgamate  <br /> max_iters| max_sigma, mean <br /> any int | max_sigma <br /> 10 |

(DCoverF being duty cycle divided by frequency)

##### Property features currently (01/06/21) implemented
Can be any property of `HarmonicallyRelatedClusters` if it's not an iterable. E.g. `num_unique_freqs` is fine but `unique_freqs` is not.
Currently: num_harmonics, num_unique_freqs, num_unique_dms, size, max_dm, min_dm

##### Stat features currently (01/06/21) implemented
Any subclass of `Stat` in `utilities/features.py`
Currently: Mean, Variance, MAD (Mean Absolute Deviation), StandardDeviation, Skewness, Kurtosis, Min, Max, Range

##### Fit features currently (01/06/21) implemented
None

### General code flow and structure
Each datacode has corresponding `DataGetter` and `FeatureGenerator` subclasses defined in `feature_generator.py`.

The `DataGetter` stores information about the type of data which needs to be extracted from any `HarmonicallyRelatedClusters` instance and has a `get(self, hrc: HarmonicallyRelatedClusters)` method which gets the correct data for the feature you wish to derive.

The `FeatureGenerator` stores information about the feature you wish to derive and has a `make` method which takes in the output from its corresponding `DataGetter`'s `get` and makes the feature, returning, at minimum, a value for the feature and its type.

For example, in the example above where you wished to calculate the kurtosis of the DM, weighted by sigma and calculated about the point with the maximum sigma:
* `Kurtosis` is a subclass of `Stat` so we will be making one `StatDataGetter` instance and one `StatGenerator` instance
* The `StatDataGetter` will store that you want to get DM, and that it should be weighted and calculated about the peak sigma point, e.g. you could manually set it up like:
  ```python
  stat_dat_get = StatDataGetter(datacode="DM", weighted=True, about_peak=True)
  ```
* The `StatGenerator` will store that you want to use the `Kurtosis` class to do the calculation e.g. you could manually set it up like:
  ```python
  import candidate_processor.utilities.features as feat
  kurtosis_class = getattr(feat, "Kurtosis")
  stat_gen = StatGenerator(statclass=kurtosis_class)
  ```
  the second line here is because you need to pass in the class to `StatGenerator`, passing in the class's name as a string will not work.
* To get the value and type of the feature, from a `HarmonicallyRelatedClusters` instance `hrc`, you would then do
  ```python
  val, dt = stat_gen.make(stat_dat_get.get(hrc))
  ```


To automate this process we have the `Features` class. You can make an instance from a configuration file set up as described above with
```python
with open(config_filename) as f:
        features_config = yaml.safe_load(f)

fg = Features.from_config(features_config)
```
This will parse your configuration and store a list of appropriate `DataGetter` subclass instances and a list of their corresponding `FeatureGetter`s.
Running the `make_single_pointing_candidate` method takes in a `HarmonicallyRelatedClusters` instance, gets the appropriate data and makes all of the initialised features, puts all your features into a numpy structured array with appropriate names (e.g. `DM_Mean_weighted`) and returns a `SinglePointingCandidate` corresponding to the input `HarmonicallyRelatedClusters` object.
```python
spc = fg.make_single_pointing_candidate(hrc)
```

And the `make_single_pointing_candidate_collection` method will take in a list of HarmonicallyRelatedClusters objects, run `make_single_pointing_candidate` on each one, and output the resulting candidates as a `SinglePointingCandidateCollection`
```python
spcc = fg.make_single_pointing_candidate_collection(hrc_list)
```

### features in a SinglePointingCandidate
For `spc`, a `SinglePointingCandidate`, its features can be accessed/interacted with via
```python
spc.features
```
This will give you a (single layer - no nesting) numpy structured array.
To get a list of what features `spc` contains, use
```python
spc.features.dtype.names
```
which will give something like this:
```
('DM_Mean', 'DM_Mean_weighted', 'DM_Variance_weighted_about_peak', 'DCoverF_Variance', 'unique_freqs', 'unique_dms', 'size', 'max_dm')
```
Note that anything of the form `spc.features["feature_name"]` will also output an array.

**Example 1**

```python
spc.features['DM_Mean']
```
gives
```
array(2.28169014)
```

**Example 2**

If the feature could not be made (e.g. a power spectrum search does not search in duty cycle so anything with datacode `DCoverF` doesn't work) it is returned as a numpy.nan and so

```python
spc.features['DCoverF_Variance']
```
gives
```
array(nan)
```

### Coding a new feature
This section should help if you wish to write a new feature. If it falls under the `Fit` or `Stat` umbrella some structure is already in place - those classes (in `utilities/features.py`) and the info here should be helpful.
If not and you add a new `DataGetter` and `FeatureGenerator` (you will also need to add something in `Features.make`) please add details here and update the [Configuration file](#configuration-file) section above.

**Note** Each feature (aka each entry in the structured array) must be a single value - no lists etc. - to work with the next stage of the pipeline without issues.

**Note2** How to handle an invalid feature - log a warning and make it return `np.nan, float` as the `value, dtype`
In general, we don't want `Features.make` to raise Errors if the feature can't be made for a particular `HarmonicallyRelatedClusters` instance, but could for another one (DCoverF is a prime example of this - with power spectra clusters it's not valid, but with hhat ones it is).

**Note3** Each DataGetter and FeatureGenerator must also have a `make_name()` method - this should be formatted to be compatible with `make_combined_name` in `feature_generator.py`.

#### Property
(mostly here for completeness - to add a new Property you need to add a new property to the `HarmonicallyRelatedClusters` class)
**DataGetter** - `PropertyDataGetter`
  * Attributes
    * datacode (str)
  * `get(hrc: HarmonicallyRelatedClusters)` returns the hrc

**FeatureGenerator** -'PropertyGenerator`
  * Attributes
    * property (str)
  * key line in `make` anything new must work with: `value = getattr(hrc, self.property)`

#### Stat
**DataGetter** -`StatDataGetter`
  * Attributes
    * datacode (str) - see [Currently accepted configurations](#currently-accepted-configurations) above
    * weighted (bool) - weight statistic by sigma
    * about_peak (bool) - if statistic is calculted about a point, use the point of maximum sigma (it's usually the mean instead)
  * `get(hrc: HarmonicallyRelatedClusters)` returns `(data, weights, point)` where `data` and `weights` are numpy 1D arrays of the same length and `point` is a number

**FeatureGenerator** - 'StatGenerator`
  * Attributes
    * statclass (Stat subclass)
  * key line in `make` anything new must work with: `stat = self.statclass.compute(data, weights, point)`

#### Fit
(currently a skeleton)
**DataGetter** -`FitDataGetter`
**FeatureGenerator** - 'FitGenerator`

### OUTDATED
## Using the process_cand function directly

```
from candidates_processor import process_cands

# If the input is an hhat array
s, g = process_cands.process_full_hhat(min_group_size, noise_threshold, hhat_array_hdf5, bool_to_plot,
output_directory, bool_to_harmonic_filter, observation_id, bool_to_update_database)

# If the input is a list of power spectrum candidates
s, g = process_cands.process_power_spec(min_group_size, noise_threshold, list_of_ps_candidates_hdf5, bool_to_plot,
output_directory, bool_to_harmonic_filter, observation_id, bool_to_update_database)

```
