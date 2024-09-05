# CHAMPSS Software

## Running Scripts

Before running any scripts that call `schedule_workflow_job` outside of a container, you'll need to run:
```
workflow workspace set champss/pipeline_batch_db/champss.workspace.yml
```

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
