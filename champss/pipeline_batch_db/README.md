# Slow Pulsar Search Pipeline
[![Continuous Integration](https://github.com/chime-sps/pipeline_batch_db/actions/workflows/continuous-integration.yml/badge.svg?branch=main)](https://github.com/chime-sps/pipeline_batch_db/actions/workflows/continuous-integration.yml)
[![Continuous Deployment](https://github.com/chime-sps/pipeline_batch_db/actions/workflows/continuous-deployment.yml/badge.svg?branch=main)](https://github.com/chime-sps/pipeline_batch_db/actions/workflows/continuous-deployment.yml)

## Scripts

### run-pipeline
```
run-pipeline [--date=DATE] RAJD DECJD [STEPS]
```
- RAJD, DECJD: sky coordinates of the pointing of interest. Uses the nearest location in the pointing map, and throws an error if one cannot be found.
- DATE: optional date of the pointing's transit (default: most recent one)
- STEPS: pipeline step to run (default: all)

The intensity data are expected to be in a subdirectory of `/data/chime/sps/raw`, in a hierarchy of `YYYY/mm/dd/beam`, and files named `chunkNNNN.msg`. All subsequent data products are written into a subdirectory of the directory from which the pipeline is being run by default unless specified: `YYYY/mm/dd/beam` for the quantization and RFI steps, and `YYYY/mm/dd/row` for beamforming and onward.

#### Example using J2111+2106 transit on May 4, 2020

Because J2111 has RAJD of 317.8880 and DECJD of 21.1019 (ref [psrcat](https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?version=1.63&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names=J2111%2B2106&ephemeris=long&submit_ephemeris=Get+Ephemeris&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&style=Long+with+last+digit+error&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query)), we use that as the argument (rounded):
```
run-pipeline --date=2020-05-04 318 21 all
```
### pointing-with-data
There is also `pointing-with-data` included in this package, to help finding which days/beams have data associated with them
```
Usage: pointing-with-data [OPTIONS]

  Script to determine the pointings with data to process in a given day.

  How to use it : pointing-with-data --date YYYYMMDD or YYYY-MM-DD or
  YYYY/MM/DD --beam <beam_no>


Options:
  --date [%Y%m%d|%Y-%m-%d|%Y/%m/%d]
                                  Date of data to process. Default = Today in
                                  UTC
  --beam INTEGER RANGE            Beam row to check for existing data. Can
                                  only input a single beam row. Default = All
                                  rows from 0 to 224  [0<=x<=255]
  -h, --help                      Show this message and exit.
```

## Installation

An in-depth tutorial to GitHub, Poetry, and all other tools that you need to know can be found in the CHIME Handbook (Guidelines section), which is continuously being updated: https://chimefrb.github.io/handbook/guidelines/poetry-intro/

To clone the repo, use
```
# With https
git clone https://github.com/chime-sps/pipeline_batch_db.git
# With ssh
git clone git@github.com:chime-sps/pipeline_batch_db.git
```

First, install Poetry to manage all dependencies and their versions, as well as to use isolated virtual environments
```
pip install poetry
```

You may have to run the following if Poetry cannot be found because it was installed in a place such as ~/.local/bin that is not in your PATH
```
export PATH="$HOME/.local/bin:$PATH"
```

Then, open the virtual environment to avoid conflict with your local pacakages, using
```
poetry shell
```

This will create your venv likely somewhere in  ~/.cache/pypoetry/virtualenvs/, however you can specify it to be created in your current directory, ./.venv/, by doing the following:
```
poetry config virtualenvs.in-project true
```

Make sure to install the project and all of its dependencies into the virtual environemnt
```
poetry install
```

Instead of opening the shell, you can prepend
```
poetry run
```
before each subsequent instruction, to use Poetry's virtual environment, which will contain all its dependencies

When you first clone this repo, you must also setup pre-commit, a package that will run checks on your code and format it accordingly automatically after you commit a change
```
pre-commit install
```

You can also install this package to your local device library without Poetry, using one of the following options
```
pip install .
```

```
pip install git+https://github.com/chime-sps/pipeline_batch_db.git
```

```
pip install git+ssh://git@github.com:chime-sps/pipeline_batch_db.git
```

## Development

After you finished your changes, use
```
cz c
```
to commit your changes with commitzen, which will help you interactively to build a standardized commit message, which is then importantly used with release versioning, changelog, and Docker image auto-generation

To add a dependency to the project, use
```
poetry add "<package_name>@<version>"
```
Use -D or -E flag to add a dependency only for dev or extras respectiely \
And use --dry-run "package_name@*" to find the most recent version of desired package that is compatible \
Then use the command above with the found version in the verbose

If you manually edit pyproject.toml, or if you want to update your private repository dependencies to their latest versions, you must also update the poetry.lock file accordingly with
```
poetry update --lock
```

## Build and Deploy

The Docker image is automatically pushed to DockerHub whenever a push to the main branch is made, so you can test your changes by pulling the image with
```
docker pull chimefrb/sps-pipeline:latest
```

You can view the container process using
```
docker ps -a
```

Alternatively, you can build the Docker image locally where you'd like using
```
docker build -f Dockerfile --platform=linux/amd64 --tag chimefrb/sps-pipeline --ssh github_ssh_id=<location_of_your_ssh_key> .
```
with the . representing the context Docker uses (the folder with all the code). Additionally, if you ssh'd into a server/node using
```
ssh -A some_server
ssh-add -L
```
then your github_ssh_id would be equal to $SSH_AUTH_SOCK

Now, you can view your image and its size with
```
docker image ls
```
