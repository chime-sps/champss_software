#!/bin/bash

# WARNING!
# THIS WILL RESET THE ENTIRE MONOREPO TO THE STATE ALL OF THE SUBREPOS' BRANCHES ARE IN!

# The name of the monorepo module
MONOREPO_NAME="champss"

# The name of your current branch in the monorepo to add the subrepos's main branches to
MONOREPO_BRANCH="setup-monorepo"

# The URL of the monorepo that will contain all of the subrepos
MONOREPO_URL="https://github.com/chime-sps/monorepo_test.git"

# All of the subrepos and their desired 'main' branch that will be added to the monorepo
SUBREPO_URLS=(
    "https://github.com/chime-sps/beamformer.git#main"
    "https://github.com/chime-sps/candidate-processor.git#main"
    "https://github.com/chime-sps/controller.git#main"
    "https://github.com/chime-sps/fdmt.git#main"
    "https://github.com/chime-sps/folding.git#main"
    "https://github.com/chime-sps/multi-pointing.git#main"
    "https://github.com/chime-sps/pipeline_batch_db.git#main"
    "https://github.com/chime-sps/ps-processes.git#main"
    "https://github.com/chime-sps/rfi-mitigation.git#main"
    "https://github.com/chime-sps/spshuff.git#apr_slow_pulsar"
    "https://github.com/chime-sps/sps-common.git#main"
    "https://github.com/chime-sps/sps-databases.git#main"
    "https://github.com/chime-sps/sps-dedispersion.git#main"
    "https://github.com/chime-sps/sps-ops.git#main"
)

# Files to delete from the subrepos after they are added to the monorepo
FILES_TO_DELETE=".mypy_cache/ .pytest_cache/ .venv/ .github/ .git/ .DS_Store .pre-commit-config.yaml .gitignore .dockerignore Dockerfile poetry.lock LICENSE CHANGELOG.md MANIFEST.in"

# A global variable to contain all of the subrepo branches per subrepo
SUBREPO_BRANCHES=()

for SUBREPO_URL in "${SUBREPO_URLS[@]}"
do
  SUBREPO_BRANCH=$(echo "$SUBREPO_URL" | awk -F'#' '{print $2}')
  SUBREPO_URL_WITHOUT_BRANCH=$(echo "$SUBREPO_URL" | sed 's/#.*//')
  SUBREPO_NAME=$(basename "$SUBREPO_URL_WITHOUT_BRANCH" .git)
  echo $SUBREPO_BRANCH
  echo $SUBREPO_URL_WITHOUT_BRANCH
  echo $SUBREPO_NAME

  cd "$MONOREPO_NAME"
  rm -rf "$SUBREPO_NAME"
  git clone $SUBREPO_URL_WITHOUT_BRANCH

  cd "$SUBREPO_NAME"
  git checkout $SUBREPO_BRANCH

  SUBREPO_BRANCHES="$(git branch -r | grep -v $SUBREPO_BRANCH | grep -v HEAD | awk -F '/' '{print $2}' | tr '\n' ' ')"
  echo $SUBREPO_BRANCHES

  SUBREPO_BRANCHES_PER_SUBREPO+=("$SUBREPO_BRANCHES")
  echo $SUBREPO_BRANCHES_PER_SUBREPO

  for FILE_TO_DELETE in $FILES_TO_DELETE
  do
    if [ -e "$FILE_TO_DELETE" ]
    then
      rm -rf "$FILE_TO_DELETE"
    fi
  done

  sed -i.bak '/chime-sps/ s|git = "ssh://git@github.com/chime-sps/\(.*\)".*|path = "../\1"|; s/, rev = "[^"]*"//g' pyproject.toml
  sed -i.bak '/path = /s/$/}/' pyproject.toml
  rm -f pyproject.toml.bak

  cd ../..
done

for SUBREPO_URL in "${SUBREPO_URLS[@]}"
do
  SUBREPO_URL_WITHOUT_BRANCH=$(echo "$SUBREPO_URL" | sed 's/#.*//')
  SUBREPO_NAME=$(basename "$SUBREPO_URL_WITHOUT_BRANCH" .git)
  echo $SUBREPO_URL_WITHOUT_BRANCH
  echo $SUBREPO_NAME

  cd "$MONOREPO_NAME/$SUBREPO_NAME"

  poetry lock

  cd ../..
done

rm -f poetry.lock
poetry lock

git add .
git commit -n -m "Adding all modified versions of each subrepo's desired main branches to your monorepo branch: $MONOREPO_BRANCH"
git push

CLONE_SUBREPO_BRANCHES=$1

if [ "$CLONE_SUBREPO_BRANCHES" -eq 1 ]; then
  for ((index=0; index<${#SUBREPO_BRANCHES_PER_SUBREPO[@]}; index++))
  do
    SUBREPO_URL=${SUBREPO_URLS[index]}
    SUBREPO_URL_WITHOUT_BRANCH=$(echo "$SUBREPO_URL" | sed 's/#.*//')
    SUBREPO_NAME=$(basename "$SUBREPO_URL_WITHOUT_BRANCH" .git)
    echo $SUBREPO_URL
    echo $SUBREPO_URL_WITHOUT_BRANCH
    echo $SUBREPO_NAME

    SUBREPO_BRANCHES=${SUBREPO_BRANCHES_PER_SUBREPO[index]}
    echo $SUBREPO_BRANCHES

    for SUBREPO_BRANCH in $SUBREPO_BRANCHES
    do
      git checkout $MONOREPO_BRANCH

      NEW_BRANCH="$SUBREPO_NAME-$SUBREPO_BRANCH"
      echo $NEW_BRANCH

      git push origin -d "$NEW_BRANCH"
      git branch -D "$NEW_BRANCH"
      git checkout -b "$NEW_BRANCH"
      git push --set-upstream origin "$NEW_BRANCH"

      cd "$MONOREPO_NAME"
      rm -rf "$SUBREPO_NAME"
      git clone $SUBREPO_URL_WITHOUT_BRANCH

      cd "$SUBREPO_NAME"
      git checkout $SUBREPO_BRANCH

      for FILE_TO_DELETE in $FILES_TO_DELETE
      do
        if [ -e "$FILE_TO_DELETE" ]
        then
          rm -rf "$FILE_TO_DELETE"
        fi
      done

      sed -i.bak '/chime-sps/ s|git = "ssh://git@github.com/chime-sps/\(.*\)".*|path = "../\1"|; s/, rev = "[^"]*"//g' pyproject.toml
      sed -i.bak '/path = /s/$/}/' pyproject.toml
      rm -f pyproject.toml.bak

      poetry lock

      cd ../..

      git add .
      git commit -n -m "Adding $NEW_BRANCH branch's codebase to the monorepo"
      git push
    done
  done
fi