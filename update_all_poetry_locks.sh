# The name of the monorepo module
MONOREPO_NAME="champss"

# All the names of the subrepo modules with poetry.lock files
SUBREPO_NAME=(
    "beamformer"
    "candidate-processor"
    "controller"
    "folding"
    "multi-pointing"
    "sps-pipeline"
    "ps-processes"
    "rfi-mitigation"
    "sps-common"
    "sps-databases"
    "sps-dedispersion"
)

for SUBREPO_NAME in "${SUBREPO_NAME[@]}"
do
  cd "$MONOREPO_NAME/$SUBREPO_NAME"

  echo $SUBREPO_NAME

  poetry lock

  cd ../..
done

poetry lock

poetry install