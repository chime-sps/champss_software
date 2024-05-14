# CHIME/SPS Beamformer
[![Continuous Integration](https://github.com/chime-sps/beamformer/actions/workflows/continous-integration.yml/badge.svg?branch=main)](https://github.com/chime-sps/beamformer/actions/workflows/continous-integration.yml)
[![Continuous Deployment](https://github.com/chime-sps/beamformer/actions/workflows/continous-deployment.yml/badge.svg?branch=main)](https://github.com/chime-sps/beamformer/actions/workflows/continous-deployment.yml)

## Installation

To clone the repo, use
```
# With https
git clone https://github.com/chime-sps/beamformer.git
# With ssh
git clone git@github.com:chime-sps/beamformer.git
```

First, install Poetry 1.5.1 to manage all dependencies and their versions, as well as to use virtual environments
```
pip install poetry==1.5.1
```

Then, open the virtual environment to avoid conflict with your local pacakages, using
```
poetry shell
```

Or alternatively, prepend
```
poetry run
```
before each subsequent instruction, to use Poetry's virtual environment, which will contain all needed dependencies

Now, install the project and all of its dependencies into the virtual environemnt
```
poetry install
```

Make sure to setup pre-commit, a package that will run checks on your code and format it accordingly automatically after you commit a change
```
pre-commit install
```

You can also install the whole package to your local device library with one of the following options
```
pip install .
```

```
pip install git+https://github.com/chime-sps/beamformer.git
```

```
pip install git+ssh://git@github.com:chime-sps/beamformer.git
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

If you want to update the dependencies to the latest compatiable versions, especially in the case of other dependent private repos having a new release version, use
```
poetry update
```

Or, if you are manually editing pyproject.toml, you must also update the poetry.lock file accordingly with
```
poetry update --lock
```

See more about using Poetry [here](https://python-poetry.org/docs/cli/)

## Build and Deploy

The Docker image is automatically created/updated in the sps-compute1 node whenever a push/pull-request to the main branch is made, so you can test your changes by logging in there and creating a Docker container to run the image with
```
docker run chimesps/beamformer:latest
```

You can view the container process using
```
docker ps -a
```

Alternatively, you can build the Docker image where you'd like using
```
DOCKER_BUILDKIT=1 docker build -f Dockerfile --platform=linux/amd64 --tag chimefrb/sps-beamformer:latest --ssh github_ssh_id=<location_of_your_ssh_key> .
```
with the . representing the context Docker uses (the folder with all the code). Additionally, if you ssh'd into a server/node using
```
ssh -a some_server
ssh-add
```
then your github_ssh_id would be equal to $SSH_AUTH_SOCK

Now, you can view your image and its size with
```
docker image ls
```

## Python API

### How to generate a new full sky a pointing map?
```python
from beamformer.strategist import mapper
sky = mapper.PointingMapper()
pointings = sky.get_pointing_map()
```

### How to find out what pointing maps are availaible?
```python
from beamformer import AVAILAIBLE_POINTING_MAPS
print(AVAILAIBLE_POINTING_MAPS)

['/some/path/beamformer/beamformer/data/pointings_map_v1-2.json',
 '/some/path/beamformer/beamformer/data/pointings_map_v1-3.json']
```

### For current pointing map
```python
from beamformer import CURRENT_POINTING_MAP
print(CURRENT_POINTING_MAP)
```

### How to get active beam pointings from a pointing map?
```python
import datetime
import time
import numpy as np
import pytz
from beamformer import AVAILAIBLE_POINTING_MAPS
from beamformer.strategist import strategist

# To list all pointing maps run
print(AVAILAIBLE_POINTING_MAPS)

# To create a list of pointings to beamform, using sps-databases
# Note that the list of pointings generated is only up to a sidereal day
strategy = strategist.PointingStrategist()
# if you do NOT have access to the database :
strategy = strategist.PointingStrategist(from_db=False)

# getting the list of pointings between now and 2 hours later for beams 128, 129, 130:
active_pointings = strategy.get_pointings(time.time(), time.time()+7200, np.asarray([128, 129, 130]))

# If you have a target ra, dec (say 300, 60) and want to find the data during transit for a given day
# for data taken in 2020-06-23
date = datetime.datetime(year=2020, month=6, day=23).replace(tzinfo=pytz.UTC)
active_pointing = strategy.get_single_pointing(300, 60, date)
```

### How to get SPS beamformed data based on active pointings?
```python
from beamformer.skybeam import skybeam
for pointing in active_pointings:
    # Create a Beamformer Class for each pointing
    beamformer = skybeam.SkyBeamFormer()
    # Beamform on the data
    skybeam = beamformer.form_skybeam(pointing)
    # write the beamformed data on a filterbank file
    skybeam.write_to_filterbank("/path/to/filterbank.fil")
```
