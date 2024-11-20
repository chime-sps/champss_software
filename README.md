# CHAMPSS Software

## Running Scripts

Before running any scripts that call `schedule_workflow_job` outside of a container, you'll need to run:
```
workflow workspace set champss.workspace.yml
```

## Merging changes to main

In order to merge your development changes to the main branch, create a pull request and request a review. When merging the approved pull request, squash all commits to a single commit and edit the commit message so that it follows [Conventional Commit format](https://www.conventionalcommits.org/en/v1.0.0/). This keeps the commit history clean and enables a new release version to be automatically created.

## Testing Branch with Docker

If you want to test your branch's code with Docker or Workflow, you can use our GitHub Action to automatically build and push a Docker Image of your branch to DockerHub by including the keyword "[test]" in a commit message pushed to your branch. Then, check the Actions tab of this repository to see when it finishes (takes ~5-10 minutes). Now, your image will be available as chimefrb/champss_software:yourbranchname.

## Seting Up Environment

Please first run:
```
pre-commit install
```
to add a local Git Hook which will automatically fix any formatting isues when commiting (no work required on your part).

There is a second pre-commit config, `.pre-commit-manual.yaml`, which will be automatically run when you create a pull request.
However, those pre-commit hooks need manual fixing on your local machine.
On your machine, these hooks can be run by adding `--config .pre-commit-manual.yaml` to pre-commit commands, such as `pre-commit run`.

To install the entire project, run:
```
poetry install
```
and then
```
poetry shell
```
to enter the virtual environment it has been installed into.

The sofware can also be installed using pip with:
```
pip install .
```

If wishing to do live edits with a pip install, you need to afterwards go to the inidividual sub-folders and run the installation there:
```
cd champss/beamformer
pip install -e . --no-deps
...
```
Otherwise, if using the Poetry virtual environment, everything should be in editable mode by default.


## Notes on apptainer images

When running our software on Narval using apptainer your job will not have internet access which can mess with astropy if it can't access cached files properly.
In order to successfully run our jobs you may need to add `--fakeroot --no-home` to your `apptainer exec` command.
