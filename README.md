# CHAMPSS Software

Please first run:
```
pre-commit install --config .precommit-automatic.yaml
```
to add a local Git Hook which will automatically fix any formatting isues when commiting (no work required on your part)

Please make sure to commit using commitizen:
```
cz c
```
because "fix" or "feat" commmits will be detected to automatically create and bump release tag versions of this codebase