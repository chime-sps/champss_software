# CHAMPSS Software

This software contains the software used for running the CHIME All-Sky Multiday Pulsar Stacking Search (CHAMPSS). The Survey was previously called the Slow Pulsar Search (SPS) which is still refelected in the code.

First paper: [Preprint](https://arxiv.org/abs/2504.16293)

## Installation

The full software package can be installed using
```
pip install .[beam-model]
```
or
```
poetry install -E beam-model
```
This installs the CHIMEFRB beam-model repository which is not public and requires being a member of the CHIMEFRB github organization. Without this repository the main CHAMPSS pipeline will not work.
Without access to this repository the software package can still be installed with
```
pip install .
```
or
```
poetry install
```

When using `poetry` activating the environment can be achieved by either using `poetry shell` or `poetry env activate` depending on the poetry version.

When working with `pip` and an editable install is wanted, this needs to be performed in the individual package folders. For example:
```
cd champss/beamformer
pip install -e . --no-deps
```

Due to the relative paths in the pyproject.toml the full software suite cannot be installed as a dependency (when using poetry at least). But sps-common can be installed as dependency in other projects.

### Running Workflow Scripts

Before running any scripts that call `schedule_workflow_job` outside of a container, you'll need to run:
```
workflow workspace set champss.workspace.yml
```

## Individual Packages

This repository contains multiple packages in the `/champss/` which were previously contained in inidvidual repositories. These individual packages are automatically installed.

The individual packages are (ordered roughly by how they appear in the processing chain):
- sps-common: Code that is shared between multiple packages.
- controller: Control data recording.
- sps-pipeline: Contains the `run-pipeline` and `run-stack-search-pipeline` pipelines which are the main component of CHAMPSS.
- sps-databases: Database methods used by the pipeline. The CHAMPSS pipeline requires a running MongoDB instance.
- scheduler: Schedule many pipeline jobs across a number of compute nodes using [Workflow](https://github.com/CHIMEFRB/workflow)
- beamformer: Create quasi-tracking beam from CHIMEFRB data
- spshuff: Methods for reading the huffman-encoded recorded data.
- rfi-mitigation: RFI removal methods in both the time and freqency domain.
- sps-dedispersion: Dedisperse the beamformed data using a fork of the [dmt](https://github.com/pravirkr/dmt) package.
- ps-processes: Create power spectra, search for detections, cluster detections and stack power spectra.
- candidate-processor: Process CHAMPSS single pointing candidates.
- multi-pointing: Create multi pointing candidates by comparing candidates resulting from different pointings.
- folding: Fold interesting candidates and search through folds of multiple observations.


## Developer Notes

### Pre-Commit

To automatically fix formatting issues and other possible coding issues we use `pre-commit` which can installed using
```
pre-commit install
```
This performs a number of checks each time a commit is created and may require you to run the commit command a second time if all issues were able to be fixed automatically.

If you are not able to fix the reported issues and still want to create a commit you can use `git commit -n` ( or use the `Commit All (No Verify)` ins VS Code) to skip the pre-commit check.

### Merging changes to main

In order to merge your development changes to the main branch, create a pull request and request a review. When merging the approved pull request, squash all commits to a single commit and edit the commit message so that it follows [Conventional Commit format](https://www.conventionalcommits.org/en/v1.0.0/). This keeps the commit history clean and enables a new release version to be automatically created.

### Testing Branch with Docker

If you want to test your branch's code with Docker or Workflow, you can use our GitHub Action to automatically build and push a Docker Image of your branch to DockerHub by including the keyword "[test]" in a commit message pushed to your branch. Then, check the Actions tab of this repository to see when it finishes (takes ~5-10 minutes). Now, your image will be available as chimefrb/champss_software:yourbranchname.


### Notes on apptainer images

When running our software on Narval using apptainer your job will not have internet access which can mess with astropy if it can't access cached files properly.
In order to successfully run our jobs you may need to add `--fakeroot --no-home` to your `apptainer exec` command.
