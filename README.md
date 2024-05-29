# CHAMPSS Software

Please first run:
```
pre-commit install
```
to add a local Git Hook which will automatically fix any formatting isues when commiting (no work required on your part).
There is a second pre-commit config `.precommit-manual.yaml` which will be automatically run when you create  pull request 
but those pre-commit hooks need manual fixing.
On your machine these hooks can be used by adding `--config .pre-commit-manual.yaml` to pre-commit commands.

To install the entire project, run:
```
poetry install
```
and then
```
poetry shell
```
to enter the venv it has been installed into.

The sofware can also be installed using pip with
```
pip install .
```
When an editable install is wanted, you need to afterwards go to the inidividual sub-folders and run the installation there.

For example:
```
cd champss/beamformer
pip install -e . --no-deps
```