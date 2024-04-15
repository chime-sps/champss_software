#!/bin/bash

# This assumes you all have the CHIME SPS GitHub repositories under one folder on your local system
# and that you have no local uncommited changes, as it will checkout and pull to the main branch
# in order to push to it

# Also assumes a "dev" folder in this one folder to diffrentiate between CI/CD with/without dispatching tests
# (for controller and folding) and with/without building an image (for pipeline_batch_db), respectively

# Does not include "sps-ops" as it is the current repository

ALL_REPOS=("pipeline_batch_db" "controller" "sps-common" "sps-databases" "beamformer" "rfi-mitigation" "sps-dedispersion" "FDMT" "ps-processes" "candidate-processor" "multi-pointing" "folding")

for REPO in "${ALL_REPOS[@]}"; do
    if [[ $REPO == "controller" || $REPO == "folding" ]]; then
        cp ../dev/continuous-integration-no-dispatch.yml ../$REPO/.github/workflows/continuous-integration.yml
    else
        cp ../dev/continuous-integration-dispatch.yml ../$REPO/.github/workflows/continuous-integration.yml
    fi

    if [[ $REPO == "pipeline_batch_db" ]]; then
        cp ../dev/continuous-deployment-image.yml ../$REPO/.github/workflows/continuous-deployment.yml
    else
        cp ../dev/continuous-deployment-no-image.yml ../$REPO/.github/workflows/continuous-deployment.yml
    fi

    cp .pre-commit-config.yaml ../$REPO/.pre-commit-config.yaml
done

for REPO in "${ALL_REPOS[@]}"; do
    cd ../$REPO
    git checkout main
    git pull

    if REPO -ne "pipeline_batch_db"; then
        git rm Dockerfile
    fi

    git add .github/workflows/continuous-integration.yml .github/workflows/continuous-deployment.yml .pre-commit-config.yaml
    git commit -n -m "fix(continuous-integration.yml): update common files, e.g. continuous-integration.yml, continuous-deployment.yml, pre-commit-config.yaml"
    git push
done