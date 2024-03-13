# CHIME/SPS Common Library
[![Continuous Integration](https://github.com/chime-sps/sps-common/actions/workflows/continuous-integration.yml/badge.svg?branch=main)](https://github.com/chime-sps/sps-common/actions/workflows/continuous-integration.yml)
[![Continuous Deployment](https://github.com/chime-sps/sps-common/actions/workflows/continuous-deployment.yml/badge.svg?branch=main)](https://github.com/chime-sps/sps-common/actions/workflows/continuous-deployment.yml)

## Installation

To clone the repo, use
```
# With https
git clone https://github.com/chime-sps/sps-common.git
# With ssh
git clone git@github.com:chime-sps/sps-common.git
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

You can also install this package to your local device library with one of the following options
```
pip install .
```

```
pip install git+https://github.com/chime-sps/sps-common.git
```

```
pip install git+ssh://git@github.com:chime-sps/sps-common.git
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
docker run chimesps/sps-common:latest
```

You can view the container process using
```
docker ps -a
```

Alternatively, you can build the Docker image where you'd like using
```
DOCKER_BUILDKIT=1 docker build -f Dockerfile --platform=linux/amd64 --tag chimefrb/sps-common:latest --ssh github_ssh_id=<location_of_your_ssh_key> .
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
