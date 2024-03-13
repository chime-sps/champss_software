#!/bin/bash

# Define the files and directories to delete
FILES_TO_DELETE=".mypy_cache/ .pytest_cache/ .venv/ .github/ .git/ .pre-commit-config.yaml .gitignore .dockerignore Dockerfile"

# Loop through each subdirectory
for dir in ./monorepo_test/*; do
  if [ -d "$dir" ]; then
    dir=${dir%*/}  # Remove trailing slash
    echo "Checking directory: $dir"
    
    # Loop through each file/directory to delete
    for file_to_delete in $FILES_TO_DELETE; do
      # Check if the file/directory exists in the current subdirectory
      if [ -e "$dir/$file_to_delete" ]; then
        echo "Deleting: $dir/$file_to_delete"
        rm -rf "$dir/$file_to_delete"
      fi
    done

    # Update the poetry.lock file
    # cd $dir
    # poetry lock
    # cd ../../
  fi
done

